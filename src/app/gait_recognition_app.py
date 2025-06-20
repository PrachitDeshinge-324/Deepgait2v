"""
Main application class for gait recognition system
"""

import cv2
import time
import os
from tqdm import tqdm

from .component_manager import ComponentManager
from .track_manager import TrackManager
from .identification_manager import IdentificationManager
from .visualization_handler import VisualizationHandler
from .keyboard_handler import KeyboardHandler
from .database_handler import DatabaseHandler
from .statistics_reporter import StatisticsReporter

# Force optimization mode
FORCE_BATCH_PROCESSING = True
FORCE_DEVICE_OPTIMIZATION = True

class GaitRecognitionApp:
    """Main application class for gait recognition with quality control"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.frame_count = 0
        self.fps = 0
        self.prev_time = 0
        self.video_writer = None
        self.cap = None
        self.next_person_id = 1
        
        # Create component objects
        self.component_manager = ComponentManager(config)
        self.track_manager = TrackManager(config)
        self.database_handler = DatabaseHandler(config)
        self.keyboard_handler = KeyboardHandler(config)
        self.statistics_reporter = StatisticsReporter()
        
    def initialize(self):
        """Initialize components and resources"""
        print("Initializing application components...")
        
        # Initialize components
        self.component_manager.initialize()
        
        # Initialize database
        db_path = os.path.join(self.config.DATA_DIR, "person_database")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.next_person_id = self.database_handler.load_database(db_path)
        
        # Share database with identification manager
        self.identification_manager = IdentificationManager(
            self.config,
            self.database_handler,  # Pass the DatabaseHandler, not just person_db
            self.next_person_id
        )
        
        # Initialize visualization handler
        self.visualization_handler = VisualizationHandler(
            self.config,
            self.component_manager.visualizer,
            self.component_manager.face_extractor
        )
        
        # Open video capture
        self.cap = cv2.VideoCapture(self.config.VIDEO_PATH)
        if not self.cap.isOpened():
            print("Error: Could not open video file")
            return False
        
        # Initialize video writer if saving video
        if self.config.SAVE_VIDEO:
            self._setup_video_writer()
        
        # Setup frames directory if needed
        if self.config.SAVE_FRAMES:
            os.makedirs(self.config.OUTPUT_FRAMES_DIR, exist_ok=True)
            print(f"Saving frames to: {self.config.OUTPUT_FRAMES_DIR}")
            
        # Setup progress bar
        self._setup_progress_bar()
        
        print("Initialization complete!")
        return True
        
    def run(self):
        """Main processing loop"""
        print("Starting video processing...")
        
        # Process video frames
        while self.cap.isOpened() and self.frame_count < self.config.MAX_FRAMES:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Update progress bar
            self._update_progress_bar()
            
            # Calculate FPS
            self._calculate_fps()
            
            # Process frame
            self._process_frame(frame)
            
            # Handle keyboard input
            if self._handle_keyboard_input():
                break
                
        print("Video processing complete!")
        
    def _process_frame(self, frame):
        """Process a single video frame"""
        # Detect, track and extract silhouettes
        detections = self.component_manager.detector.detect(frame)
        tracks = self.component_manager.tracker.update(detections)
        active_track_ids = {t.track_id for t in tracks}
        
        # Extract silhouettes and face embeddings
        silhouette_sequences = self.component_manager.silhouette_extractor.extract_silhouettes(frame, tracks)
        face_embeddings = self._extract_face_embeddings(frame, tracks)
        
        # Update track data
        self.track_manager.update_tracks(
            silhouette_sequences, 
            face_embeddings, 
            active_track_ids, 
            self.frame_count
        )
        
        # Process tracks for quality assessment and identification
        self._process_tracks(active_track_ids)
        
        # Visualize results
        self._visualize_frame(frame, tracks, active_track_ids)
    
    def _extract_face_embeddings(self, frame, tracks):
        """Extract face embeddings if face extractor available"""
        face_embeddings = {}
        if self.component_manager.face_extractor is not None:
            try:
                return self.component_manager.face_extractor.extract_faces(frame, tracks)
            except Exception as e:
                print(f"Face extraction failed: {e}")
        return face_embeddings
        
    def _process_tracks(self, active_track_ids):
        """Process tracks for quality assessment and identification with batch optimization"""
        ready_tracks = self.track_manager.get_tracks_ready_for_processing()
        
        # Prepare batch data for efficient processing
        batch_data = []
        track_metadata = []
        
        for track_id in ready_tracks:
            # Get track data
            silhouettes = self.track_manager.get_track_silhouettes(track_id)
            
            # Skip if track is no longer active or has insufficient data
            if track_id not in active_track_ids or len(silhouettes) < self.track_manager.MIN_FRAMES_FOR_PROCESSING:
                continue
                
            # Quality assessment
            quality_result = self.component_manager.quality_assessor.assess_sequence_quality(silhouettes)
            self.track_manager.add_quality_assessment(track_id, quality_result)
            
            # Skip identification if quality is too low
            if not quality_result['is_acceptable']:
                continue
            
            # Add to batch for processing
            batch_data.append({
                'silhouettes': silhouettes,
                'parsings': None,  # Will be generated if needed
                'metadata': {'track_id': track_id}
            })
            track_metadata.append({
                'track_id': track_id,
                'quality_score': quality_result['overall_score']
            })
        
        # Process batch if we have data
        if batch_data:
            try:
                # Try batch processing first (more efficient)
                if hasattr(self.component_manager.gait_recognizer, 'recognize_batch') and len(batch_data) > 1:
                    embeddings = self.component_manager.gait_recognizer.recognize_batch(batch_data)
                    
                    # Process results
                    for i, (embedding, metadata) in enumerate(zip(embeddings, track_metadata)):
                        if embedding is not None:
                            self._process_single_identification(
                                metadata['track_id'], 
                                embedding, 
                                metadata['quality_score'],
                                active_track_ids
                            )
                else:
                    # Fallback to individual processing
                    for data, metadata in zip(batch_data, track_metadata):
                        embedding = self.component_manager.gait_recognizer.recognize(data['silhouettes'])
                        if embedding is not None:
                            self._process_single_identification(
                                metadata['track_id'], 
                                embedding, 
                                metadata['quality_score'],
                                active_track_ids
                            )
                            
            except Exception as e:
                print(f"Batch processing failed, using individual processing: {e}")
                # Fallback to individual processing
                for data, metadata in zip(batch_data, track_metadata):
                    try:
                        embedding = self.component_manager.gait_recognizer.recognize(data['silhouettes'])
                        if embedding is not None:
                            self._process_single_identification(
                                metadata['track_id'], 
                                embedding, 
                                metadata['quality_score'],
                                active_track_ids
                            )
                    except Exception as track_error:
                        print(f"Failed to process track {metadata['track_id']}: {track_error}")
    
    def _process_single_identification(self, track_id, embedding, quality_score, active_track_ids):
        """Process identification for a single track"""
        # Get face embedding if available
        face_embedding = self.track_manager.get_track_face_embedding(track_id)
        
        # Person identification
        self.identification_manager.process_identification(
            track_id, 
            embedding,
            face_embedding,
            quality_score,
            active_track_ids,
            self.track_manager.track_identities
        )
            
    def _visualize_frame(self, frame, tracks, active_track_ids):
        """Visualize the current frame with all annotations"""
        # Get data needed for visualization
        track_identities = self.track_manager.track_identities
        track_face_embeddings = self.track_manager.track_face_embeddings
        
        # Generate visualization
        vis_frame = self.visualization_handler.create_visualized_frame(
            frame, 
            tracks, 
            track_identities,
            track_face_embeddings,
            active_track_ids,
            self.database_handler.person_db,
            self.component_manager.gait_recognizer,
            self.frame_count,
            self.fps
        )
        
        # Display and/or save frame
        if self.config.SAVE_VIDEO and self.video_writer is not None:
            self.video_writer.write(vis_frame)
        
        if self.config.SAVE_FRAMES:
            frame_filename = f"{self.config.OUTPUT_FRAMES_DIR}/frame_{self.frame_count:04d}.jpg"
            cv2.imwrite(frame_filename, vis_frame)
        
        # Display frame if enabled
        if self.config.SHOW_DISPLAY:
            cv2.imshow('Gait Recognition with Quality Control', vis_frame)
    
    def _handle_keyboard_input(self):
        """Handle keyboard input, return True if should exit"""
        key = cv2.waitKey(1) & 0xFF if self.config.SHOW_DISPLAY else -1
        
        # Exit on 'q'
        if key == ord('q'):
            return True
            
        # Let keyboard handler process other commands
        self.keyboard_handler.handle_key(
            key, 
            self.component_manager.gait_recognizer,
            self.database_handler,
            os.path.join(self.config.DATA_DIR, "person_database")
        )
            
        return False
        
    def cleanup(self):
        """Clean up resources"""
        # Save database
        print("Saving person database...")
        db_path = os.path.join(self.config.DATA_DIR, "person_database")
        self.database_handler.save_database(db_path)
        
        # Close progress bar
        if hasattr(self, 'pbar') and self.config.SHOW_PROGRESS:
            self.pbar.close()
        
        # Release video resources
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
            print(f"Output video saved to: {self.config.OUTPUT_VIDEO_PATH}")
        
        cv2.destroyAllWindows()
        
    def report_statistics(self):
        """Report summary statistics"""
        track_identities = self.track_manager.track_identities
        person_db = self.database_handler.person_db
        
        self.statistics_reporter.report_statistics(
            self.frame_count,
            track_identities,
            person_db
        )
    
    def _setup_video_writer(self):
        """Set up video writer for saving output"""
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        os.makedirs(os.path.dirname(self.config.OUTPUT_VIDEO_PATH), exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.config.OUTPUT_VIDEO_PATH, 
            fourcc, 
            fps, 
            (frame_width, frame_height)
        )
        print(f"Saving output video to: {self.config.OUTPUT_VIDEO_PATH}")
    
    def _setup_progress_bar(self):
        """Set up progress bar for processing"""
        if self.config.SHOW_PROGRESS:
            cap_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frames = min(cap_frames, self.config.MAX_FRAMES) if cap_frames > 0 else self.config.MAX_FRAMES
            self.pbar = tqdm(total=max_frames, 
                           desc="Processing video", 
                           unit="frames",
                           dynamic_ncols=True,
                           leave=True,
                           miniters=1)
    
    def _update_progress_bar(self):
        """Update progress bar with current status"""
        if self.config.SHOW_PROGRESS and hasattr(self, 'pbar'):
            self.pbar.update(1)
            self.pbar.set_postfix({
                'FPS': f'{self.fps:.1f}',
                'Tracks': len(self.track_manager.track_silhouettes)
            })
            # Force refresh
            self.pbar.refresh()
    
    def _calculate_fps(self):
        """Calculate frames per second"""
        curr_time = time.time()
        if self.prev_time > 0:
            self.fps = 1/(curr_time - self.prev_time)
        self.prev_time = curr_time