#!/usr/bin/env python3
"""
Check saved database for face embeddings
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.database import PersonEmbeddingDatabase

def check_saved_database():
    """Check if face embeddings are saved in the database"""
    print("=== Checking Saved Database for Face Embeddings ===")
    
    # Try to load existing database
    db_paths = [
        "data_DeepGaitV2/person_database",
        "person_database",
        "data/person_database"
    ]
    
    loaded = False
    db = PersonEmbeddingDatabase()
    
    for db_path in db_paths:
        if os.path.exists(f"{db_path}.meta") and os.path.exists(f"{db_path}.index"):
            print(f"Found database at: {db_path}")
            try:
                db.load_from_disk(db_path)
                loaded = True
                break
            except Exception as e:
                print(f"Failed to load database from {db_path}: {e}")
    
    if not loaded:
        print("No database found or failed to load.")
        print("Available files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.meta') or file.endswith('.index'):
                print(f"  {file}")
        return
    
    print(f"\nDatabase loaded successfully!")
    print(f"Total people in database: {len(db.people)}")
    
    # Check face embedding statistics
    people_with_faces = 0
    people_with_gait = 0
    total_face_embeddings = 0
    total_gait_embeddings = 0
    
    print("\nPerson details:")
    for person_id, data in db.people.items():
        has_face = 'face_embeddings' in data and len(data['face_embeddings']) > 0
        has_gait = 'gait_embeddings' in data and len(data['gait_embeddings']) > 0
        
        face_count = len(data.get('face_embeddings', []))
        gait_count = len(data.get('gait_embeddings', []))
        
        print(f"  {data['name']} ({person_id}):")
        print(f"    Gait embeddings: {gait_count}")
        print(f"    Face embeddings: {face_count}")
        print(f"    Quality: {data.get('quality', 'N/A')}")
        print(f"    Data keys: {list(data.keys())}")
        
        if has_face:
            people_with_faces += 1
            total_face_embeddings += face_count
        if has_gait:
            people_with_gait += 1
            total_gait_embeddings += gait_count
    
    print(f"\nSummary:")
    print(f"  People with gait embeddings: {people_with_gait}")
    print(f"  People with face embeddings: {people_with_faces}")
    print(f"  Total gait embeddings: {total_gait_embeddings}")
    print(f"  Total face embeddings: {total_face_embeddings}")
    
    # Test face identification if we have face embeddings
    if total_face_embeddings > 0:
        print(f"\nTesting face identification...")
        for person_id, data in db.people.items():
            if 'face_embeddings' in data and len(data['face_embeddings']) > 0:
                test_embedding = data['face_embeddings'][0]
                matches = db.identify_person_face(test_embedding, threshold=0.5)
                print(f"  Face ID test for {data['name']}: {len(matches)} matches")
                if matches:
                    best_match = matches[0]
                    print(f"    Best match: {best_match[2]} (similarity: {best_match[1]:.3f})")
                break
    
    print("\n=== Check Complete ===")

if __name__ == "__main__":
    check_saved_database()
