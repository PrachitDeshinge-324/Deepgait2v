"""
Quality Assessment Test and Analysis Tool
Test the robustness of the quality assessment system
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from modules.quality_assessor import GaitSequenceQualityAssessor
from modules.person_database import PersonEmbeddingDatabase
import config

def create_test_silhouettes(quality_type="good"):
    """Create test silhouettes for different quality scenarios"""
    
    # Base parameters
    height, width = 128, 64
    sequence_length = 30
    
    silhouettes = []
    
    if quality_type == "good":
        # High quality sequence with consistent silhouettes
        for i in range(sequence_length):
            sil = np.zeros((height, width), dtype=np.uint8)
            
            # Create a walking person silhouette that changes over time
            center_x = width // 2
            center_y = height // 2
            
            # Body (ellipse)
            body_width = 20 + 3 * np.sin(i * 0.3)  # Slight width variation
            body_height = 60
            cv2.ellipse(sil, (center_x, center_y), (int(body_width), body_height), 0, 0, 360, 255, -1)
            
            # Head
            cv2.circle(sil, (center_x, center_y - 40), 12, 255, -1)
            
            # Legs with gait cycle motion
            leg_separation = 15 + 10 * np.sin(i * 0.4)
            left_leg_x = center_x - int(leg_separation)
            right_leg_x = center_x + int(leg_separation)
            
            # Left leg
            cv2.rectangle(sil, (left_leg_x - 5, center_y + 30), (left_leg_x + 5, center_y + 70), 255, -1)
            # Right leg
            cv2.rectangle(sil, (right_leg_x - 5, center_y + 30), (right_leg_x + 5, center_y + 70), 255, -1)
            
            # Arms
            arm_swing = 8 * np.sin(i * 0.4)
            left_arm_x = center_x - 25 + int(arm_swing)
            right_arm_x = center_x + 25 - int(arm_swing)
            
            cv2.rectangle(sil, (left_arm_x - 3, center_y - 20), (left_arm_x + 3, center_y + 10), 255, -1)
            cv2.rectangle(sil, (right_arm_x - 3, center_y - 20), (right_arm_x + 3, center_y + 10), 255, -1)
            
            silhouettes.append(sil)
    
    elif quality_type == "poor":
        # Poor quality sequence with issues
        for i in range(sequence_length):
            sil = np.zeros((height, width), dtype=np.uint8)
            
            # Inconsistent and incomplete silhouettes
            if i % 5 == 0:  # Missing frames
                silhouettes.append(sil)
                continue
            
            # Very small or fragmented silhouette
            center_x = width // 2 + np.random.randint(-10, 10)  # Random position drift
            center_y = height // 2 + np.random.randint(-5, 5)
            
            # Small, inconsistent body
            body_size = np.random.randint(8, 15)
            cv2.circle(sil, (center_x, center_y), body_size, 255, -1)
            
            # Add noise
            noise = np.random.randint(0, 2, (height, width)) * 255
            sil = cv2.bitwise_or(sil, noise.astype(np.uint8))
            
            silhouettes.append(sil)
    
    elif quality_type == "medium":
        # Medium quality with some issues
        for i in range(sequence_length):
            sil = np.zeros((height, width), dtype=np.uint8)
            
            center_x = width // 2
            center_y = height // 2
            
            # Body with some inconsistency
            body_width = 18 + 5 * np.sin(i * 0.3) + np.random.randint(-2, 2)
            body_height = 55 + np.random.randint(-5, 5)
            cv2.ellipse(sil, (center_x, center_y), (int(body_width), body_height), 0, 0, 360, 255, -1)
            
            # Head (sometimes missing)
            if i % 7 != 0:
                cv2.circle(sil, (center_x, center_y - 35), 10, 255, -1)
            
            # Legs with moderate motion
            leg_separation = 12 + 8 * np.sin(i * 0.35)
            left_leg_x = center_x - int(leg_separation)
            right_leg_x = center_x + int(leg_separation)
            
            cv2.rectangle(sil, (left_leg_x - 4, center_y + 25), (left_leg_x + 4, center_y + 65), 255, -1)
            cv2.rectangle(sil, (right_leg_x - 4, center_y + 25), (right_leg_x + 4, center_y + 65), 255, -1)
            
            # Blur some frames
            if i % 4 == 0:
                sil = cv2.GaussianBlur(sil, (5, 5), 0)
                sil = (sil > 127).astype(np.uint8) * 255
            
            silhouettes.append(sil)
    
    elif quality_type == "short":
        # Too short sequence
        sequence_length = 8
        for i in range(sequence_length):
            sil = np.zeros((height, width), dtype=np.uint8)
            center_x = width // 2
            center_y = height // 2
            cv2.ellipse(sil, (center_x, center_y), (20, 60), 0, 0, 360, 255, -1)
            cv2.circle(sil, (center_x, center_y - 40), 12, 255, -1)
            silhouettes.append(sil)
    
    return silhouettes

def test_quality_assessment():
    """Test quality assessment on different types of sequences"""
    
    print("=== Quality Assessment Robustness Test ===")
    
    # Initialize quality assessor
    quality_assessor = GaitSequenceQualityAssessor(config.QUALITY_ASSESSOR_CONFIG)
    
    # Test different quality scenarios
    test_cases = ["good", "poor", "medium", "short"]
    results = {}
    
    for case in test_cases:
        print(f"\nTesting {case} quality sequence...")
        
        silhouettes = create_test_silhouettes(case)
        quality_result = quality_assessor.assess_sequence_quality(silhouettes)
        
        results[case] = quality_result
        
        print(f"Overall Score: {quality_result['overall_score']:.3f}")
        print(f"Quality Level: {quality_result['quality_level']}")
        print(f"Is Acceptable: {quality_result['is_acceptable']}")
        print(f"Is High Quality: {quality_result['is_high_quality']}")
        
        # Print component scores
        component_scores = quality_result['metrics'].get('component_scores', {})
        print("Component Scores:")
        for component, score in component_scores.items():
            print(f"  {component}: {score:.3f}")
        
        # Print recommendations
        recommendations = quality_result.get('recommendations', [])
        if recommendations:
            print("Recommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
    
    return results

def test_database_operations():
    """Test database operations with quality control"""
    
    print("\n=== Database Operations Test ===")
    
    # Initialize database
    db = PersonEmbeddingDatabase("test_person_embeddings.db", config.PERSON_DATABASE_CONFIG)
    quality_assessor = GaitSequenceQualityAssessor(config.QUALITY_ASSESSOR_CONFIG)
    
    # Create test embeddings with different qualities
    test_persons = [
        {"id": "TestPerson_001", "name": "Alice", "quality_type": "good"},
        {"id": "TestPerson_002", "name": "Bob", "quality_type": "medium"},
        {"id": "TestPerson_003", "name": "Charlie", "quality_type": "poor"},
    ]
    
    for person in test_persons:
        # Create test silhouettes
        silhouettes = create_test_silhouettes(person["quality_type"])
        
        # Assess quality
        quality_result = quality_assessor.assess_sequence_quality(silhouettes)
        
        # Create dummy embedding
        embedding = np.random.randn(256)  # Random embedding for testing
        
        # Try to add to database
        success = db.add_person_embedding(
            person_id=person["id"],
            embedding=embedding,
            quality_result=quality_result,
            sequence_length=len(silhouettes),
            source_info=f"test_{person['quality_type']}",
            person_name=person["name"]
        )
        
        print(f"Person {person['name']} ({person['quality_type']} quality): "
              f"Added to DB: {success}, Quality: {quality_result['overall_score']:.3f}")
    
    # Test database queries
    print("\nDatabase Statistics:")
    stats = db.get_person_statistics()
    print(f"Total persons: {stats.get('total_persons', 0)}")
    print(f"Total embeddings: {stats.get('total_active_embeddings', 0)}")
    print(f"Average quality: {stats.get('average_quality', 0.0):.3f}")
    
    # Test person-specific statistics
    for person in test_persons:
        person_stats = db.get_person_statistics(person["id"])
        if 'error' not in person_stats:
            print(f"{person['name']}: {person_stats['active_embeddings']} embeddings, "
                  f"avg quality: {person_stats['average_quality']:.3f}")
    
    # Test clustering
    print("\nTesting clustering:")
    for person in test_persons:
        if db.get_person_statistics(person["id"]).get('active_embeddings', 0) > 0:
            cluster_result = db.cluster_person_embeddings(person["id"])
            print(f"{person['name']}: {cluster_result}")
    
    # Cleanup test database
    import os
    try:
        os.remove("test_person_embeddings.db")
        print("\nTest database cleaned up")
    except:
        pass

def analyze_quality_thresholds():
    """Analyze the effectiveness of quality thresholds"""
    
    print("\n=== Quality Threshold Analysis ===")
    
    quality_assessor = GaitSequenceQualityAssessor(config.QUALITY_ASSESSOR_CONFIG)
    
    # Test various scenarios
    scenarios = {
        "Excellent": "good",
        "Degraded": "medium", 
        "Poor": "poor",
        "Too Short": "short"
    }
    
    print("Quality Assessment Results:")
    print("-" * 60)
    print(f"{'Scenario':<12} {'Score':<8} {'Level':<12} {'Acceptable':<10} {'High Quality'}")
    print("-" * 60)
    
    for name, quality_type in scenarios.items():
        silhouettes = create_test_silhouettes(quality_type)
        result = quality_assessor.assess_sequence_quality(silhouettes)
        
        score = result['overall_score']
        level = result['quality_level']
        acceptable = "Yes" if result['is_acceptable'] else "No"
        high_quality = "Yes" if result['is_high_quality'] else "No"
        
        print(f"{name:<12} {score:<8.3f} {level:<12} {acceptable:<10} {high_quality}")
    
    print("-" * 60)

def visualize_sample_sequence():
    """Create a visual sample of different quality sequences"""
    
    print("\n=== Creating Visual Samples ===")
    
    quality_types = ["good", "medium", "poor"]
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle('Sample Silhouette Sequences by Quality Level')
    
    for i, quality_type in enumerate(quality_types):
        silhouettes = create_test_silhouettes(quality_type)
        
        # Show first 5 frames
        for j in range(5):
            if j < len(silhouettes):
                axes[i, j].imshow(silhouettes[j], cmap='gray')
            axes[i, j].set_title(f'{quality_type.title()} - Frame {j+1}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('quality_samples.png', dpi=150, bbox_inches='tight')
    print("Visual samples saved as 'quality_samples.png'")
    plt.close()

def main():
    """Run comprehensive quality assessment tests"""
    
    print("Starting Quality Assessment and Database Robustness Tests")
    print("=" * 60)
    
    # Test quality assessment
    quality_results = test_quality_assessment()
    
    # Test database operations
    test_database_operations()
    
    # Analyze thresholds
    analyze_quality_thresholds()
    
    # Create visual samples
    try:
        visualize_sample_sequence()
    except Exception as e:
        print(f"Could not create visual samples: {e}")
    
    print("\n" + "=" * 60)
    print("Quality Assessment and Database Tests Completed!")
    
    # Summary of results
    print("\nSummary:")
    for quality_type, result in quality_results.items():
        score = result['overall_score']
        acceptable = result['is_acceptable']
        print(f"- {quality_type.title()} quality: Score {score:.3f}, "
              f"Acceptable: {acceptable}")

if __name__ == "__main__":
    main()
