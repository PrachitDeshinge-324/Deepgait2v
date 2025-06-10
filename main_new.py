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
from modules.detector import PersonDetector
from modules.tracker import PersonTracker
from modules.visualizer import Visualizer
from modules.silhouette_extractor import SilhouetteExtractor
from modules.gait_recognizer import GaitRecognizer
from modules.quality_assessor import GaitSequenceQualityAssessor
from modules.person_database import PersonEmbeddingDatabase

def main():
    """Main function to run the tracking and recognition application with quality control"""
    
    # Initialize components
    detector = PersonDetector(config.MODEL_PATH)
    tracker = PersonTracker(config.TRACKER_CONFIG)
    visualizer = Visualizer(config.VISUALIZATION_CONFIG)
    silhouette_extractor = SilhouetteExtractor(config.SILHOUETTE_CONFIG, config.SEG_MODEL_PATH)
    gait_recognizer = GaitRecognizer(config.GAIT_MODEL_PATH, config.GAIT_RECOGNIZER_CONFIG)
    
    # Initialize quality assessor and database
    quality_assessor = GaitSequenceQualityAssessor()
    person_db = PersonEmbeddingDatabase("person_embeddings.db")
    
    print("=== Gait Recognition System with Quality Control ===")
    db_stats = person_db.get_person_statistics()
    print(f"Database contains {db_stats.get('total_persons', 0)} persons with {db_stats.get('total_active_embeddings', 0)} embeddings")
    print(f"Average quality: {db_stats.get('average_quality', 0.0):.3f}")
    
    # Open video capture
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Variables for FPS calculation
    prev_time = 0
    curr_time = 0
    fps = 0
    
    # Dictionary to store silhouettes for each track
    track_silhouettes = {}
    
    # Dictionary to store quality assessments
    track_quality_history = {}
    
    # Dictionary to store recognition results
    track_identities = {}
    
    # Processing parameters
    MIN_FRAMES_FOR_PROCESSING = 15
    QUALITY_CHECK_INTERVAL = 10
    RECOGNITION_INTERVAL = 25
    MAX_SILHOUETTES_PER_TRACK = 60
    
    frame_count = 0
    
    print("Starting video processing...")
    
    # Process video frames
    while cap.isOpened() and frame_count < config.MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Calculate FPS
        curr_time = time.time()
        if prev_time > 0:
            fps = 1/(curr_time - prev_time)
        prev_time = curr_time
        
        # Detect persons
        detections = detector.detect(frame)
        
        # Track persons
        tracks = tracker.update(detections)
        
        # Extract silhouettes for active tracks
        silhouette_sequences = silhouette_extractor.extract_silhouettes(frame, tracks)
        
        # Update track silhouettes and process quality
        for track_id, sequence in silhouette_sequences.items():
            if sequence:
                # Initialize track data if new
                if track_id not in track_silhouettes:
                    track_silhouettes[track_id] = []
                    track_quality_history[track_id] = []
                
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
                    
                    print(f"Track {track_id}: Quality={quality_result['overall_score']:.3f} "
                          f"({quality_result['quality_level']}) - {len(track_silhouettes[track_id])} frames")
                    
                    # Try recognition if quality is acceptable and enough frames
                    if (quality_result['is_acceptable'] and 
                        len(track_silhouettes[track_id]) >= RECOGNITION_INTERVAL):
                        
                        # Generate embedding for recognition
                        embedding = gait_recognizer.recognize(track_silhouettes[track_id])
                        
                        if embedding is not None:
                            print(f"Generated embedding for track {track_id} - Shape: {embedding.shape}")
                            
                            # Try to find a match in the database
                            match_result = person_db.find_best_match(
                                embedding, 
                                quality_result['overall_score']
                            )
                            
                            if match_result:
                                person_id = match_result['person_id']
                                confidence = match_result['confidence']
                                similarity = match_result['similarity']
                                
                                track_identities[track_id] = {
                                    'person_id': person_id,
                                    'confidence': confidence,
                                    'similarity': similarity,
                                    'quality': quality_result['overall_score']
                                }
                                
                                print(f"✓ Track {track_id} matched to {person_id} "
                                      f"(confidence: {confidence:.3f}, similarity: {similarity:.3f})")
                                
                            else:
                                # No match found - create new person entry if quality is high
                                if quality_result['is_high_quality']:
                                    new_person_id = f"Person_{len(person_db.get_person_statistics().get('total_persons', 0)) + 1:03d}"
                                    
                                    success = person_db.add_person_embedding(
                                        person_id=new_person_id,
                                        embedding=embedding,
                                        quality_result=quality_result,
                                        sequence_length=len(track_silhouettes[track_id]),
                                        source_info=f"video_frame_{frame_count}_track_{track_id}"
                                    )
                                    
                                    if success:
                                        track_identities[track_id] = {
                                            'person_id': new_person_id,
                                            'confidence': 1.0,  # New person, high confidence
                                            'similarity': 1.0,
                                            'quality': quality_result['overall_score'],
                                            'is_new': True
                                        }
                                        
                                        print(f"✓ Created new person {new_person_id} for track {track_id} "
                                              f"(quality: {quality_result['overall_score']:.3f})")
                                    else:
                                        print(f"⚠ Failed to create new person for track {track_id}")
                                else:
                                    print(f"⚠ Track {track_id} quality too low for new person creation "
                                          f"({quality_result['overall_score']:.3f})")
                            
                            # If we have a good match and high quality, add to existing person's database
                            if (match_result and quality_result['is_high_quality'] and 
                                match_result['confidence'] > 0.8):
                                
                                person_db.add_person_embedding(
                                    person_id=match_result['person_id'],
                                    embedding=embedding,
                                    quality_result=quality_result,
                                    sequence_length=len(track_silhouettes[track_id]),
                                    source_info=f"video_frame_{frame_count}_track_{track_id}"
                                )
                                
                                print(f"✓ Added high-quality embedding to {match_result['person_id']}")
        
        # Clean up old tracks that are no longer active
        active_track_ids = {track.track_id for track in tracks}
        
        # Remove data for inactive tracks (but keep recognized identities for display)
        for track_id in list(track_silhouettes.keys()):
            if track_id not in active_track_ids:
                # Clean up silhouettes and quality history to save memory
                if track_id in track_silhouettes:
                    del track_silhouettes[track_id]
                if track_id in track_quality_history:
                    del track_quality_history[track_id]
        
        # Visualize results
        vis_frame = visualizer.draw_tracks(frame, tracks)
        
        # Add recognition results to visualization
        for track in tracks:
            if track.track_id in track_identities:
                identity_info = track_identities[track.track_id]
                person_id = identity_info['person_id']
                confidence = identity_info['confidence']
                quality = identity_info['quality']
                
                # Draw identity information
                x1, y1, x2, y2 = track.tlbr
                label = f"{person_id} ({confidence:.2f})"
                if identity_info.get('is_new', False):
                    label += " [NEW]"
                
                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(vis_frame, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add quality indicator
                quality_color = (0, 255, 0) if quality > 0.7 else (0, 165, 255) if quality > 0.5 else (0, 0, 255)
                cv2.putText(vis_frame, f"Q:{quality:.2f}", (int(x1), int(y2) + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)
        
        # Add FPS and frame info
        cv2.putText(vis_frame, f"FPS: {fps:.1f} | Frame: {frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add database stats
        if frame_count % 100 == 0:  # Update stats every 100 frames
            db_stats = person_db.get_person_statistics()
            stats_text = f"DB: {db_stats.get('total_persons', 0)} persons, {db_stats.get('total_active_embeddings', 0)} embeddings"
            cv2.putText(vis_frame, stats_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Gait Recognition with Quality Control', vis_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current state
            print("\n=== Saving Database State ===")
            cleanup_stats = person_db.cleanup_database()
            print(f"Cleanup completed: {cleanup_stats}")
            
            # Print final statistics
            final_stats = person_db.get_person_statistics()
            print(f"Final database statistics: {final_stats}")
            
        elif key == ord('c'):
            # Perform clustering analysis
            print("\n=== Performing Clustering Analysis ===")
            db_stats = person_db.get_person_statistics()
            if db_stats.get('total_persons', 0) > 0:
                # Get all person IDs and cluster their embeddings
                conn = sqlite3.connect("person_embeddings.db")
                cursor = conn.cursor()
                cursor.execute('SELECT DISTINCT person_id FROM embeddings WHERE is_active = 1')
                person_ids = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                for person_id in person_ids:
                    cluster_result = person_db.cluster_person_embeddings(person_id)
                    print(f"Person {person_id}: {cluster_result}")
            else:
                print("No persons in database to cluster")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics and cleanup
    print("\n=== Final Processing Summary ===")
    final_stats = person_db.get_person_statistics()
    print(f"Total persons registered: {final_stats.get('total_persons', 0)}")
    print(f"Total active embeddings: {final_stats.get('total_active_embeddings', 0)}")
    print(f"Average embedding quality: {final_stats.get('average_quality', 0.0):.3f}")
    print(f"Total recognition attempts: {final_stats.get('total_recognitions', 0)}")
    
    # Perform final cleanup
    cleanup_stats = person_db.cleanup_database()
    print(f"Final cleanup: {cleanup_stats}")
    
    print("Processing completed!")

if __name__ == "__main__":
    main()
