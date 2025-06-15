"""
Main entry point for person detection, tracking, and gait recognition with quality control
"""

import cv2
import time
import config
import os
import numpy as np
import pickle
import sqlite3
import hashlib
import colorsys
from datetime import datetime
from tqdm import tqdm
from modules.detector import PersonDetector
from modules.tracker import PersonTracker
from modules.visualizer import Visualizer
from modules.silhouette_extractor import SilhouetteExtractor
from modules.gait_recognizer import GaitRecognizer
from modules.quality_assessor import GaitSequenceQualityAssessor
from utils.database import PersonEmbeddingDatabase  # Import the database class

def vprint(*args, **kwargs):
    """Verbose print - only prints if VERBOSE is enabled"""
    if config.VERBOSE:
        print(*args, **kwargs)

def get_color_for_id(id_value, saturation=0.8, brightness=0.9):
    """
    Generate a consistent color based on an ID value.
    Returns a tuple of (B, G, R) values for OpenCV.
    """
    # Use the hash of the string representation of the ID
    hash_object = hashlib.md5(str(id_value).encode())
    hash_hex = hash_object.hexdigest()
    
    # Convert part of the hash to an integer
    hash_int = int(hash_hex[:8], 16)
    
    # Use the hash to generate a hue value (0-360)
    hue = hash_int % 360
    
    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue/360, saturation, brightness)
    
    # Scale to 0-255 range for OpenCV
    return (int(b*255), int(g*255), int(r*255))

def enforce_unique_person_assignments(track_identities, active_track_ids, current_track_id, potential_match):
    """
    Ensures that each track gets assigned to a unique person ID by
    checking if the potential match is already assigned to another active track.
    
    Returns:
        bool: True if this person_id is available, False if already used by another track
    """
    person_id = potential_match[0]
    
    # Check if this person ID is already assigned to another track
    for track_id, identity in track_identities.items():
        if (track_id != current_track_id and 
            track_id in active_track_ids and 
            identity['person_id'] == person_id):
            # This person is already assigned to another active track
            print(f"Conflict: Person {person_id} already assigned to track {track_id}")
            return False
            
    # This person ID is not used by other tracks
    return True

def should_update_embedding(current_quality, stored_quality, similarity):
    """
    Determine if we should update the stored embedding based on quality and similarity.
    
    Args:
        current_quality: Quality score of current embedding
        stored_quality: Quality score of stored embedding
        similarity: Similarity between current and stored embedding
        
    Returns:
        bool: True if we should update, False otherwise
    """
    # Update if quality is significantly better
    if current_quality > stored_quality * 1.1:  # 10% better
        return True
    
    # Update if quality is better and similarity is high (same person, better view)
    if current_quality > stored_quality and similarity > 0.8:
        return True
        
    return False

def main():
    """Main function to run the tracking and recognition application with quality control"""
    
    # Initialize components
    detector = PersonDetector(config.MODEL_PATH)
    tracker = PersonTracker(config.TRACKER_CONFIG)
    visualizer = Visualizer(config.VISUALIZATION_CONFIG)
    silhouette_extractor = SilhouetteExtractor(config.SILHOUETTE_CONFIG, config.SEG_MODEL_PATH)
    
    # Initialize gait recognizer with enhanced toggle functionality
    vprint(f"Initializing GaitRecognizer with {config.GAIT_MODEL_TYPE} model...")
    gait_recognizer = GaitRecognizer(model_type=config.GAIT_MODEL_TYPE)
    
    # Show current model information
    model_info = gait_recognizer.get_model_info()
    vprint(f"Gait recognizer info: {model_info}")
    
    # Model is loaded based on config.GAIT_MODEL_TYPE setting
    
    quality_assessor = GaitSequenceQualityAssessor(config.QUALITY_ASSESSOR_CONFIG)
    
    # Initialize person database
    embedding_dimension = 256*16  # Adjust based on your actual embedding size
    person_db = PersonEmbeddingDatabase(dimension=embedding_dimension)
    
    # Initialize person counter for sequential IDs
    next_person_id = 1
    
    # Load existing database if available
    db_path = os.path.join(config.DATA_DIR, "person_database")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    if os.path.exists(db_path + ".index"):
        vprint(f"Loading database from {db_path}.index")
        try:
            person_db.load_from_disk(db_path)
            # Print detailed information to confirm loading worked
            print(f"Database loaded with {len(person_db.people)} persons")
            if hasattr(person_db, 'index') and hasattr(person_db.index, 'ntotal'):
                vprint(f"FAISS index size: {person_db.index.ntotal}")
            
            # Test the database with a random embedding to verify search works
            if hasattr(person_db, 'index') and person_db.index.ntotal > 0:
                vprint("Testing database search functionality...")
                test_emb = np.random.rand(1, embedding_dimension).astype(np.float32)
                test_results = person_db.identify_person(test_emb, top_k=5, threshold=0.1)
                vprint(f"Test search returned {len(test_results)} results")
            # Print a few entries to verify content
            vprint("Database contents:")
            for i, (pid, info) in enumerate(person_db.people.items()):
                if i < 3:  # Print first 3 entries for verification
                    vprint(f"  Person: {pid}, Name: {info['name']}, Quality: {info['quality']}")
                else:
                    break
                    
            # Extract the highest person ID number from existing entries
            existing_ids = []
            for person_id in person_db.people.keys():
                # Extract numeric part from IDs that follow the pattern "P123"
                if person_id.startswith('P') and person_id[1:].isdigit():
                    existing_ids.append(int(person_id[1:]))
            
            # Set next_person_id to highest value + 1 if any exist
            if existing_ids:
                next_person_id = max(existing_ids) + 1
                print(f"Starting person numbering at {next_person_id} based on existing database")
                
        except Exception as e:
            print(f"ERROR: Database loading failed: {e}")
            print("Creating a new database instead.")
            person_db = PersonEmbeddingDatabase(dimension=embedding_dimension)
    else:
        print(f"No existing database found at {db_path}.index. Creating a new one.")
    
    # Open video capture
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Initialize video writer if saving video
    video_writer = None
    if config.SAVE_VIDEO:
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(config.OUTPUT_VIDEO_PATH), exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
        print(f"Saving output video to: {config.OUTPUT_VIDEO_PATH}")
    
    # Create frames directory if saving individual frames
    if config.SAVE_FRAMES:
        os.makedirs(config.OUTPUT_FRAMES_DIR, exist_ok=True)
        print(f"Saving frames to: {config.OUTPUT_FRAMES_DIR}")
    
    # Variables for FPS calculation
    prev_time = 0
    curr_time = 0
    fps = 0
    
    # Dictionary to store silhouettes for each track
    track_silhouettes = {}
    track_embeddings = {}
    # Dictionary to store quality assessments
    track_quality_history = {}
    
    # Dictionary to store recognition results
    track_identities = {}
    
    # Dictionary to store track timing information
    track_timings = {}  # track_id -> {'first_seen': frame, 'last_seen': frame}
    
    # Processing parameters
    MIN_FRAMES_FOR_PROCESSING = 15
    QUALITY_CHECK_INTERVAL = 10
    RECOGNITION_INTERVAL = 25
    MAX_SILHOUETTES_PER_TRACK = 60
    
    frame_count = 0
    
    # Initialize progress bar
    cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(cap_frames, config.MAX_FRAMES) if cap_frames > 0 else config.MAX_FRAMES
    
    if config.SHOW_PROGRESS:
        pbar = tqdm(total=max_frames, desc="Processing video", unit="frames")
    
    print("Starting video processing...")
    
    # Process video frames
    while cap.isOpened() and frame_count < config.MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Update progress bar
        if config.SHOW_PROGRESS:
            pbar.update(1)
            pbar.set_postfix({
                'FPS': f'{fps:.1f}' if 'fps' in locals() else '0.0',
                'Tracks': len(tracks) if 'tracks' in locals() else 0
            })
        
        # Calculate FPS
        curr_time = time.time()
        if prev_time > 0:
            fps = 1/(curr_time - prev_time)
        prev_time = curr_time
        
        # Detect persons
        detections = detector.detect(frame)
        
        # Track persons
        tracks = tracker.update(detections)
        
        # Get active track IDs
        active_track_ids = {t.track_id for t in tracks}
        
        # Extract silhouettes for active tracks
        silhouette_sequences = silhouette_extractor.extract_silhouettes(frame, tracks)
        
        # Update track silhouettes and process quality
        for track_id, sequence in silhouette_sequences.items():
            if sequence:
                # Initialize track data if new
                if track_id not in track_silhouettes:
                    track_silhouettes[track_id] = []
                    track_quality_history[track_id] = []
                    track_timings[track_id] = {'first_seen': frame_count, 'last_seen': frame_count}
                else:
                    # Update last seen frame
                    track_timings[track_id]['last_seen'] = frame_count
                
                # Add new silhouettes
                track_silhouettes[track_id].extend(sequence)
                
                # Limit silhouette history to prevent memory issues
                if len(track_silhouettes[track_id]) > MAX_SILHOUETTES_PER_TRACK:
                    track_silhouettes[track_id] = track_silhouettes[track_id][-MAX_SILHOUETTES_PER_TRACK:]
                
                # Perform quality check periodically
                if (len(track_silhouettes[track_id]) >= MIN_FRAMES_FOR_PROCESSING and 
                    len(track_silhouettes[track_id]) % QUALITY_CHECK_INTERVAL == 0):
                    
                    quality_result = quality_assessor.assess_sequence_quality(track_silhouettes[track_id])
                    track_quality_history[track_id].append(quality_result)
                    
                    vprint(f"Track {track_id}: Quality={quality_result['overall_score']:.3f} "
                          f"({quality_result['quality_level']}) - {len(track_silhouettes[track_id])} frames")
                    
                    # Try recognition if quality is acceptable and enough frames
                    if (quality_result['is_acceptable'] and 
                        len(track_silhouettes[track_id]) >= RECOGNITION_INTERVAL):
                        
                        # Generate embedding for recognition
                        embedding = gait_recognizer.recognize(track_silhouettes[track_id])

                        # Person identification logic
                        if embedding is not None:
                            vprint(f"Generated embedding for track {track_id} - Shape: {embedding.shape}")
                            track_embeddings[track_id] = embedding
                            
                            # Get current quality assessment
                            current_quality = quality_result['overall_score']
                            
                            # First check if this track already has an identity
                            if track_id in track_identities:
                                # Track already identified, just update embedding if quality improved
                                current_person_id = track_identities[track_id]['person_id']
                                current_person_quality = track_identities[track_id]['quality']
                                
                                vprint(f"Track {track_id} already identified as {track_identities[track_id]['name']}")
                                if current_quality > current_person_quality:
                                    vprint(f"Updating existing identity {current_person_id} with higher quality embedding")
                                    person_db.update_person(current_person_id, embedding=embedding, quality=current_quality)
                                    track_identities[track_id]['quality'] = current_quality
                            else:
                                # Debug track identification state
                                vprint(f"Track {track_id} needs identity assignment")
                                
                                # New identification needed - search database using configured method
                                match_threshold = config.IDENTIFICATION_THRESHOLD
                                
                                if config.IDENTIFICATION_METHOD == "nucleus":
                                    matches = person_db.identify_person_adaptive(
                                        embedding, 
                                        method='nucleus',
                                        top_p=config.NUCLEUS_TOP_P,
                                        min_candidates=config.NUCLEUS_MIN_CANDIDATES,
                                        max_candidates=config.NUCLEUS_MAX_CANDIDATES,
                                        threshold=match_threshold
                                    )
                                    vprint(f"Track {track_id}: Using nucleus sampling (top_p={config.NUCLEUS_TOP_P})")
                                else:
                                    matches = person_db.identify_person(embedding, top_k=config.TOP_K_CANDIDATES, threshold=match_threshold)
                                    vprint(f"Track {track_id}: Using top-k sampling (k={config.TOP_K_CANDIDATES})")
                                
                                vprint(f"Track {track_id}: Found {len(matches)} potential matches")
                                for i, match in enumerate(matches):
                                    person_id, similarity, name, stored_quality = match
                                    vprint(f"  Match {i+1}: {name} (ID: {person_id}), similarity: {similarity:.3f}, quality: {stored_quality:.3f}")
                                vprint(f"Track {track_id} match threshold: {config.IDENTIFICATION_THRESHOLD}, best match similarity: {similarity if matches else 'N/A'}")
                                
                                # Get list of person IDs already assigned to OTHER active tracks
                                assigned_person_ids = set()
                                for other_id in active_track_ids:
                                    if other_id != track_id and other_id in track_identities:
                                        assigned_person_ids.add(track_identities[other_id]['person_id'])
                                
                                vprint(f"Person IDs already assigned to other tracks: {assigned_person_ids}")
                                
                                # Find matches that aren't already assigned
                                available_matches = []
                                for match in matches:
                                    person_id = match[0]
                                    if person_id not in assigned_person_ids:
                                        available_matches.append(match)
                                
                                # DEBUG: Print available matches
                                vprint(f"Available matches (not used by other tracks): {len(available_matches)}")
                                
                                # Check if we have available matches and use them
                                if len(available_matches) > 0:
                                    # Use best available match
                                    person_id, similarity, name, stored_quality = available_matches[0]
                                    
                                    vprint(f">>> ASSIGNING Track {track_id} to existing database person {name} with similarity {similarity:.3f}")
                                    
                                    # Assign this identity to the track
                                    track_identities[track_id] = {
                                        'person_id': person_id,
                                        'name': name,
                                        'confidence': similarity,
                                        'quality': current_quality,
                                        'is_new': False  # Not new, from database
                                    }
                                    
                                    vprint(f"SUCCESS: Track {track_id} identified as existing person {name}")
                                    
                                    # Update database if quality improved
                                    if current_quality > stored_quality:
                                        person_db.update_person(person_id, embedding=embedding, quality=current_quality)
                                else:
                                    # No available matches - create new person
                                    vprint(f"No available matches for Track {track_id} - creating new person")
                                    
                                    # Use sequential numbering for consistent IDs
                                    new_person_id = f"P{next_person_id:04d}"
                                    
                                    # Generate a simple numbered name
                                    new_person_name = f"Person-{next_person_id:04d}"
                                    
                                    # Increment the counter for next time
                                    next_person_id += 1
                                    
                                    # Add to database
                                    person_db.add_person(
                                        person_id=new_person_id,
                                        name=new_person_name,
                                        embedding=embedding,
                                        quality=current_quality,
                                        metadata={'first_seen': datetime.now().isoformat()}
                                    )
                                    
                                    # Assign to track
                                    track_identities[track_id] = {
                                        'person_id': new_person_id,
                                        'name': new_person_name,
                                        'confidence': 1.0,
                                        'quality': current_quality,
                                        'is_new': True  # New person
                                    }
                                    
                                    if matches:
                                        vprint(f"Track {track_id}: Created new person {new_person_name} because all matches were already assigned to other tracks")
                                    else:
                                        vprint(f"Track {track_id}: Created new person {new_person_name} because no matches found")
        
        # Clean up old tracks that are no longer active
        for track_id in list(track_silhouettes.keys()):
            if track_id not in active_track_ids:
                # Clean up silhouettes and quality history to save memory
                if track_id in track_silhouettes:
                    del track_silhouettes[track_id]
                if track_id in track_quality_history:
                    del track_quality_history[track_id]
                # Keep track timings and identities for a while for conflict detection
        
        # Visualize results
        vis_frame = visualizer.draw_tracks(frame, tracks)
        
        # Check for duplicate person IDs (for visualization)
        person_id_counts = {}
        for track_id in active_track_ids:
            if track_id in track_identities:
                person_id = track_identities[track_id]['person_id']
                if person_id in person_id_counts:
                    person_id_counts[person_id].append(track_id)
                else:
                    person_id_counts[person_id] = [track_id]
        
        # Find duplicate assignments
        duplicate_person_ids = {pid: tracks for pid, tracks in person_id_counts.items() if len(tracks) > 1}
        if duplicate_person_ids:
            print(f"WARNING: Duplicate person assignments detected: {duplicate_person_ids}")
        
        # Draw bounding box and identity information
        for track in tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = track.tlbr
            
            if track_id in track_identities:
                identity_info = track_identities[track_id]
                person_id = identity_info['person_id']
                confidence = identity_info['confidence']
                quality = identity_info['quality']
                name = identity_info['name']
                is_new = identity_info.get('is_new', False)
                
                # Check if this person ID is assigned to multiple tracks
                is_duplicate = person_id in duplicate_person_ids
                
                # Get person-specific color
                color = (0, 0, 255) if is_duplicate else get_color_for_id(person_id)
                
                # Draw bounding box with person-specific color
                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Add label with name and confidence
                label = f"{name} ({confidence:.2f})"
                if is_duplicate:
                    label += " ⚠️"  # Warning emoji for duplicate IDs
                if is_new:
                    label += " [NEW]"  # Indicate newly created person
                
                # Add background for text
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis_frame, 
                            (int(x1), int(y1) - text_size[1] - 10), 
                            (int(x1) + text_size[0], int(y1)), 
                            color, -1)
                
                # Draw text in white
                cv2.putText(vis_frame, label, (int(x1), int(y1) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add colored dot to indicate new vs. database match
                dot_color = (0, 0, 255) if is_new else (0, 255, 0)  # Red for new, Green for database match
                dot_radius = 5
                dot_position = (int(x2) - 10, int(y1) + 10)  # Top-right corner of box
                cv2.circle(vis_frame, dot_position, dot_radius, dot_color, -1)
                
                # Add quality indicator
                quality_color = (0, 255, 0) if quality > 0.7 else (0, 165, 255) if quality > 0.5 else (0, 0, 255)
                quality_text = f"Q:{quality:.2f}"
                cv2.putText(vis_frame, quality_text, (int(x1), int(y2) + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)
            else:
                # For tracks without identity yet, use a default color
                default_color = (0, 255, 255)  # Yellow
                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), default_color, 2)
                cv2.putText(vis_frame, f"Track {track_id}", (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, default_color, 2)

        # Add counter showing active tracks vs unique people
        num_active_tracks = len(tracks)
        unique_people_ids = set()
        for track_id in active_track_ids:
            if track_id in track_identities:
                unique_people_ids.add(track_identities[track_id]['person_id'])
        num_unique_people = len(unique_people_ids)
        
        counter_text = f"Tracks: {num_active_tracks}, Unique People: {num_unique_people}"
        cv2.putText(vis_frame, counter_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add database size indicator
        db_size = len(person_db.people) if hasattr(person_db, 'people') else 0
        db_text = f"Database: {db_size} persons"
        cv2.putText(vis_frame, db_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add FPS, frame info, and current model
        cv2.putText(vis_frame, f"FPS: {fps:.1f} | Frame: {frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show current gait recognition model and sampling method
        model_text = f"Model: {gait_recognizer.current_model_type}"
        cv2.putText(vis_frame, model_text, 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show sampling method
        if config.IDENTIFICATION_METHOD == "nucleus":
            sampling_text = f"Sampling: Nucleus (p={config.NUCLEUS_TOP_P})"
        else:
            sampling_text = f"Sampling: Top-K (k={config.TOP_K_CANDIDATES})"
        cv2.putText(vis_frame, sampling_text, 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display and/or save frame
        if config.SAVE_VIDEO and video_writer is not None:
            video_writer.write(vis_frame)
        
        if config.SAVE_FRAMES:
            frame_filename = f"{config.OUTPUT_FRAMES_DIR}/frame_{frame_count:04d}.jpg"
            cv2.imwrite(frame_filename, vis_frame)
        
        # Display frame if enabled
        if config.SHOW_DISPLAY:
            cv2.imshow('Gait Recognition with Quality Control', vis_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF if config.SHOW_DISPLAY else -1
        if key == ord('q'):
            break
        elif key == ord('s'):  # Save database on demand
            print("Manually saving person database...")
            person_db.save_to_disk(db_path)
            print(f"Database saved with {len(person_db.people)} persons")
        elif key == ord('i'):  # Show model info
            model_info = gait_recognizer.get_model_info()
            print(f"\nCurrent model info:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
        elif key == ord('h'):  # Show help
            print("\n===== Keyboard Controls =====")
            print("q: Quit")
            print("s: Save database")
            print("i: Show model information")
            print("v: Toggle verbose mode")
            print("h: Show this help")
            print("==============================")
        elif key == ord('v'):  # Toggle verbose mode
            config.VERBOSE = not config.VERBOSE
            print(f"Verbose mode: {'ON' if config.VERBOSE else 'OFF'}")

    # Final save of the database
    print("Saving person database...")
    person_db.save_to_disk(db_path)
    print(f"Database saved with {len(person_db.people)} persons")
    
    # Close progress bar
    if config.SHOW_PROGRESS:
        pbar.close()
    
    # Cleanup
    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"Output video saved to: {config.OUTPUT_VIDEO_PATH}")
    cv2.destroyAllWindows()
    
    # Print summary statistics
    print("\n===== Processing Summary =====")
    print(f"Total frames processed: {frame_count}")
    print(f"Total people identified: {len(person_db.people)}")
    print(f"Identification statistics:")
    
    # Count new vs returning people
    new_count = 0
    returning_count = 0
    for track_id, identity in track_identities.items():
        if identity.get('is_new', False):
            new_count += 1
        else:
            returning_count += 1
    
    print(f"- New people: {new_count}")
    print(f"- Returning people: {returning_count}")
    print("============================")
    
    print("Processing completed!")

if __name__ == "__main__":
    main()