"""
Comprehensive Demo of the Robust Gait Recognition System
Demonstrates quality assessment, database management, and person identification
"""

import numpy as np
import cv2
import os
import sys
import time
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add modules to path
sys.path.append('.')

from modules.quality_assessor import GaitSequenceQualityAssessor
from modules.person_database import PersonEmbeddingDatabase
from modules.gait_recognizer import GaitRecognizer
import config

class GaitRecognitionDemo:
    """Comprehensive demonstration of the robust gait recognition system"""
    
    def __init__(self):
        # Initialize components
        self.quality_assessor = GaitSequenceQualityAssessor(config.QUALITY_ASSESSOR_CONFIG)
        self.person_db = PersonEmbeddingDatabase("demo_person_embeddings.db", config.PERSON_DATABASE_CONFIG)
        
        # Initialize gait recognizer if model is available
        self.gait_recognizer = None
        try:
            if os.path.exists(config.GAIT_MODEL_PATH):
                self.gait_recognizer = GaitRecognizer(config.GAIT_MODEL_PATH, config.GAIT_RECOGNIZER_CONFIG)
                print("✓ Gait recognizer loaded successfully")
            else:
                print("⚠ Gait model not found - using simulated embeddings")
        except Exception as e:
            print(f"⚠ Could not load gait recognizer: {e}")
            print("  Using simulated embeddings for demo")
        
        # Demo data
        self.demo_persons = [
            {"name": "Alice", "id": "PERSON_001", "sequences": []},
            {"name": "Bob", "id": "PERSON_002", "sequences": []},
            {"name": "Charlie", "id": "PERSON_003", "sequences": []},
        ]
        
        print("🚀 Gait Recognition Demo System Initialized")
        print("=" * 60)
    
    def create_realistic_silhouette_sequence(self, person_config: Dict) -> List[np.ndarray]:
        """Create realistic silhouette sequences with different quality characteristics"""
        
        height, width = 128, 64
        sequence_length = person_config.get('length', 30)
        quality_type = person_config.get('quality', 'good')
        
        silhouettes = []
        
        # Person-specific characteristics
        base_height = person_config.get('height_ratio', 0.7)  # Relative height
        base_width = person_config.get('width_ratio', 0.3)    # Relative width
        walking_speed = person_config.get('walking_speed', 1.0)
        
        for i in range(sequence_length):
            sil = np.zeros((height, width), dtype=np.uint8)
            
            # Calculate frame-specific parameters
            frame_factor = i / sequence_length
            
            # Simulate walking cycle
            gait_phase = (i * walking_speed * 0.4) % (2 * np.pi)
            
            # Body center with slight movement
            center_x = width // 2 + int(3 * np.sin(gait_phase * 0.5))
            center_y = int(height * (0.3 + base_height * 0.4))
            
            # Body dimensions with gait-based variation
            body_width = int(base_width * width * (1 + 0.2 * np.sin(gait_phase)))
            body_height = int(base_height * height * 0.6)
            
            if quality_type == 'excellent':
                # High-quality silhouette
                self._draw_detailed_silhouette(sil, center_x, center_y, body_width, body_height, gait_phase)
                
            elif quality_type == 'good':
                # Good quality with minor variations
                self._draw_good_silhouette(sil, center_x, center_y, body_width, body_height, gait_phase, i)
                
            elif quality_type == 'medium':
                # Medium quality with some issues
                self._draw_medium_silhouette(sil, center_x, center_y, body_width, body_height, gait_phase, i)
                
            elif quality_type == 'poor':
                # Poor quality with significant issues
                self._draw_poor_silhouette(sil, center_x, center_y, body_width, body_height, gait_phase, i)
                
            elif quality_type == 'very_poor':
                # Very poor quality
                self._draw_very_poor_silhouette(sil, center_x, center_y, body_width, body_height, i)
            
            silhouettes.append(sil)
        
        return silhouettes
    
    def _draw_detailed_silhouette(self, sil, center_x, center_y, body_width, body_height, gait_phase):
        """Draw high-quality detailed silhouette"""
        # Head
        head_radius = 12
        cv2.circle(sil, (center_x, center_y - body_height//2 - head_radius), head_radius, 255, -1)
        
        # Torso
        cv2.ellipse(sil, (center_x, center_y), (body_width//2, body_height//2), 0, 0, 360, 255, -1)
        
        # Arms with swing
        arm_swing = 15 * np.sin(gait_phase)
        left_arm_x = center_x - body_width//2 - 5 + int(arm_swing)
        right_arm_x = center_x + body_width//2 + 5 - int(arm_swing)
        
        cv2.rectangle(sil, (left_arm_x - 3, center_y - body_height//3), 
                     (left_arm_x + 3, center_y + body_height//4), 255, -1)
        cv2.rectangle(sil, (right_arm_x - 3, center_y - body_height//3), 
                     (right_arm_x + 3, center_y + body_height//4), 255, -1)
        
        # Legs with detailed gait cycle
        leg_separation = 15 + 10 * np.sin(gait_phase)
        left_leg_x = center_x - int(leg_separation)
        right_leg_x = center_x + int(leg_separation)
        
        # Thighs
        cv2.rectangle(sil, (left_leg_x - 6, center_y + body_height//4), 
                     (left_leg_x + 6, center_y + body_height//2 + 20), 255, -1)
        cv2.rectangle(sil, (right_leg_x - 6, center_y + body_height//4), 
                     (right_leg_x + 6, center_y + body_height//2 + 20), 255, -1)
        
        # Lower legs
        left_lower = left_leg_x + int(5 * np.sin(gait_phase + np.pi/4))
        right_lower = right_leg_x + int(5 * np.sin(gait_phase - np.pi/4))
        
        cv2.rectangle(sil, (left_lower - 4, center_y + body_height//2 + 15), 
                     (left_lower + 4, center_y + body_height//2 + 45), 255, -1)
        cv2.rectangle(sil, (right_lower - 4, center_y + body_height//2 + 15), 
                     (right_lower + 4, center_y + body_height//2 + 45), 255, -1)
    
    def _draw_good_silhouette(self, sil, center_x, center_y, body_width, body_height, gait_phase, frame_idx):
        """Draw good quality silhouette with minor imperfections"""
        # Similar to detailed but with slight variations
        self._draw_detailed_silhouette(sil, center_x, center_y, body_width, body_height, gait_phase)
        
        # Add minor noise occasionally
        if frame_idx % 8 == 0:
            noise = np.random.randint(0, 2, sil.shape) * 50
            sil = cv2.add(sil, noise.astype(np.uint8))
            sil = np.clip(sil, 0, 255)
    
    def _draw_medium_silhouette(self, sil, center_x, center_y, body_width, body_height, gait_phase, frame_idx):
        """Draw medium quality silhouette with some issues"""
        # Simplified body
        cv2.ellipse(sil, (center_x, center_y), (body_width//2, body_height//2), 0, 0, 360, 255, -1)
        
        # Head (sometimes missing)
        if frame_idx % 6 != 0:
            cv2.circle(sil, (center_x, center_y - body_height//2 - 10), 10, 255, -1)
        
        # Basic legs
        leg_sep = 12 + 8 * np.sin(gait_phase)
        cv2.rectangle(sil, (center_x - int(leg_sep) - 4, center_y + body_height//4), 
                     (center_x - int(leg_sep) + 4, center_y + body_height//2 + 35), 255, -1)
        cv2.rectangle(sil, (center_x + int(leg_sep) - 4, center_y + body_height//4), 
                     (center_x + int(leg_sep) + 4, center_y + body_height//2 + 35), 255, -1)
        
        # Add blur occasionally
        if frame_idx % 5 == 0:
            sil = cv2.GaussianBlur(sil, (3, 3), 0)
            sil = (sil > 127).astype(np.uint8) * 255
    
    def _draw_poor_silhouette(self, sil, center_x, center_y, body_width, body_height, gait_phase, frame_idx):
        """Draw poor quality silhouette with significant issues"""
        # Very basic shape with inconsistencies
        if frame_idx % 4 == 0:  # Missing frames
            return
        
        # Random size variation
        size_var = np.random.uniform(0.7, 1.3)
        body_width = int(body_width * size_var)
        body_height = int(body_height * size_var)
        
        # Basic blob
        cv2.ellipse(sil, (center_x, center_y), (body_width//2, body_height//2), 0, 0, 360, 200, -1)
        
        # Add significant noise
        noise = np.random.randint(0, 3, sil.shape) * 100
        sil = cv2.add(sil, noise.astype(np.uint8))
        sil = np.clip(sil, 0, 255)
    
    def _draw_very_poor_silhouette(self, sil, center_x, center_y, body_width, body_height, frame_idx):
        """Draw very poor quality silhouette"""
        # Highly fragmented and inconsistent
        if frame_idx % 3 == 0:  # Many missing frames
            return
        
        # Random scattered blobs
        num_blobs = np.random.randint(1, 4)
        for _ in range(num_blobs):
            blob_x = center_x + np.random.randint(-20, 20)
            blob_y = center_y + np.random.randint(-30, 30)
            blob_size = np.random.randint(5, 15)
            cv2.circle(sil, (blob_x, blob_y), blob_size, 150, -1)
    
    def generate_demo_data(self):
        """Generate demonstration data for different persons and quality levels"""
        
        print("📊 Generating demonstration data...")
        
        # Define different scenarios for each person
        scenarios = [
            # Alice - High quality sequences
            {
                "person": 0, "quality": "excellent", "length": 35, 
                "height_ratio": 0.8, "width_ratio": 0.3, "walking_speed": 1.0
            },
            {
                "person": 0, "quality": "good", "length": 40, 
                "height_ratio": 0.8, "width_ratio": 0.3, "walking_speed": 1.1
            },
            {
                "person": 0, "quality": "good", "length": 30, 
                "height_ratio": 0.8, "width_ratio": 0.3, "walking_speed": 0.9
            },
            
            # Bob - Mixed quality sequences
            {
                "person": 1, "quality": "good", "length": 32, 
                "height_ratio": 0.9, "width_ratio": 0.35, "walking_speed": 1.2
            },
            {
                "person": 1, "quality": "medium", "length": 28, 
                "height_ratio": 0.9, "width_ratio": 0.35, "walking_speed": 1.1
            },
            {
                "person": 1, "quality": "poor", "length": 25, 
                "height_ratio": 0.9, "width_ratio": 0.35, "walking_speed": 1.0
            },
            
            # Charlie - Lower quality sequences
            {
                "person": 2, "quality": "medium", "length": 22, 
                "height_ratio": 0.7, "width_ratio": 0.28, "walking_speed": 0.8
            },
            {
                "person": 2, "quality": "poor", "length": 20, 
                "height_ratio": 0.7, "width_ratio": 0.28, "walking_speed": 0.9
            },
            {
                "person": 2, "quality": "very_poor", "length": 15, 
                "height_ratio": 0.7, "width_ratio": 0.28, "walking_speed": 1.0
            },
        ]
        
        for i, scenario in enumerate(scenarios):
            person_idx = scenario["person"]
            
            print(f"  Generating sequence {i+1}/9 for {self.demo_persons[person_idx]['name']} "
                  f"({scenario['quality']} quality)...")
            
            # Generate silhouette sequence
            silhouettes = self.create_realistic_silhouette_sequence(scenario)
            
            # Assess quality
            quality_result = self.quality_assessor.assess_sequence_quality(silhouettes)
            
            # Generate or simulate embedding
            if self.gait_recognizer:
                try:
                    embedding = self.gait_recognizer.recognize(silhouettes)
                except:
                    embedding = None
            else:
                embedding = None
            
            if embedding is None:
                # Create person-specific simulated embedding
                base_embedding = np.random.RandomState(person_idx * 100 + i).randn(256)
                # Add small variations for different sequences of same person
                variation = np.random.RandomState(i * 50).randn(256) * 0.1
                embedding = base_embedding + variation
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
            
            # Store in demo data
            sequence_data = {
                'silhouettes': silhouettes,
                'quality_result': quality_result,
                'embedding': embedding,
                'scenario': scenario,
                'sequence_id': i + 1
            }
            
            self.demo_persons[person_idx]['sequences'].append(sequence_data)
            
            print(f"    Quality: {quality_result['overall_score']:.3f} "
                  f"({quality_result['quality_level']})")
        
        print("✓ Demo data generation completed\n")
    
    def demonstrate_quality_assessment(self):
        """Demonstrate the quality assessment capabilities"""
        
        print("🔍 Quality Assessment Demonstration")
        print("-" * 50)
        
        # Collect all quality results
        all_results = []
        
        for person in self.demo_persons:
            for seq_data in person['sequences']:
                quality_result = seq_data['quality_result']
                scenario = seq_data['scenario']
                
                result_summary = {
                    'person': person['name'],
                    'sequence_id': seq_data['sequence_id'],
                    'quality_type': scenario['quality'],
                    'overall_score': quality_result['overall_score'],
                    'quality_level': quality_result['quality_level'],
                    'is_acceptable': quality_result['is_acceptable'],
                    'is_high_quality': quality_result['is_high_quality'],
                    'length': len(seq_data['silhouettes']),
                    'components': quality_result['metrics']['component_scores']
                }
                
                all_results.append(result_summary)
                
                print(f"{person['name']} Seq#{seq_data['sequence_id']:2d} "
                      f"({scenario['quality']:10s}): "
                      f"Score={quality_result['overall_score']:.3f} "
                      f"({quality_result['quality_level']:10s}) "
                      f"Accept={quality_result['is_acceptable']} "
                      f"High={quality_result['is_high_quality']}")
        
        # Analysis
        print(f"\n📈 Quality Analysis Summary:")
        print(f"  Total sequences: {len(all_results)}")
        acceptable_count = sum(1 for r in all_results if r['is_acceptable'])
        high_quality_count = sum(1 for r in all_results if r['is_high_quality'])
        
        print(f"  Acceptable quality: {acceptable_count}/{len(all_results)} "
              f"({acceptable_count/len(all_results)*100:.1f}%)")
        print(f"  High quality: {high_quality_count}/{len(all_results)} "
              f"({high_quality_count/len(all_results)*100:.1f}%)")
        
        # Quality distribution by intended quality
        quality_types = ['excellent', 'good', 'medium', 'poor', 'very_poor']
        for q_type in quality_types:
            type_results = [r for r in all_results if r['quality_type'] == q_type]
            if type_results:
                avg_score = np.mean([r['overall_score'] for r in type_results])
                accept_rate = np.mean([r['is_acceptable'] for r in type_results])
                print(f"  {q_type:10s}: {len(type_results)} seqs, "
                      f"avg={avg_score:.3f}, accept_rate={accept_rate:.1%}")
        
        return all_results
    
    def demonstrate_database_operations(self):
        """Demonstrate database operations with quality control"""
        
        print("\n💾 Database Operations Demonstration")
        print("-" * 50)
        
        added_count = 0
        rejected_count = 0
        
        # Add embeddings to database
        for person in self.demo_persons:
            print(f"\nProcessing {person['name']} ({person['id']}):")
            
            for seq_data in person['sequences']:
                success = self.person_db.add_person_embedding(
                    person_id=person['id'],
                    embedding=seq_data['embedding'],
                    quality_result=seq_data['quality_result'],
                    sequence_length=len(seq_data['silhouettes']),
                    source_info=f"demo_seq_{seq_data['sequence_id']}_{seq_data['scenario']['quality']}",
                    person_name=person['name']
                )
                
                if success:
                    added_count += 1
                    print(f"  ✓ Seq#{seq_data['sequence_id']:2d} added "
                          f"(quality: {seq_data['quality_result']['overall_score']:.3f})")
                else:
                    rejected_count += 1
                    print(f"  ✗ Seq#{seq_data['sequence_id']:2d} rejected "
                          f"(quality: {seq_data['quality_result']['overall_score']:.3f})")
        
        print(f"\n📊 Database Population Summary:")
        print(f"  Added: {added_count}, Rejected: {rejected_count}")
        
        # Database statistics
        db_stats = self.person_db.get_person_statistics()
        print(f"  Total persons: {db_stats.get('total_persons', 0)}")
        print(f"  Total embeddings: {db_stats.get('total_active_embeddings', 0)}")
        print(f"  Average quality: {db_stats.get('average_quality', 0.0):.3f}")
        
        # Per-person statistics
        print(f"\n👥 Per-Person Statistics:")
        for person in self.demo_persons:
            person_stats = self.person_db.get_person_statistics(person['id'])
            if 'error' not in person_stats:
                print(f"  {person['name']:8s}: {person_stats['active_embeddings']} embeddings, "
                      f"avg_quality={person_stats['average_quality']:.3f}, "
                      f"max_quality={person_stats['max_quality']:.3f}")
    
    def demonstrate_recognition_accuracy(self):
        """Demonstrate recognition accuracy with quality-based matching"""
        
        print("\n🎯 Recognition Accuracy Demonstration")
        print("-" * 50)
        
        # Create test scenarios
        test_embeddings = []
        
        # Use some sequences for testing (simulate new observations)
        for person in self.demo_persons:
            if person['sequences']:
                # Use the first sequence as a "new observation"
                test_seq = person['sequences'][0]
                test_embeddings.append({
                    'person_id': person['id'],
                    'person_name': person['name'],
                    'embedding': test_seq['embedding'],
                    'quality': test_seq['quality_result']['overall_score'],
                    'sequence_id': test_seq['sequence_id']
                })
        
        print("Testing recognition with 'new' observations:")
        
        correct_matches = 0
        total_tests = 0
        
        for test_data in test_embeddings:
            # Try to match against database
            match_result = self.person_db.find_best_match(
                test_data['embedding'], 
                test_data['quality']
            )
            
            total_tests += 1
            
            if match_result:
                matched_id = match_result['person_id']
                confidence = match_result['confidence']
                similarity = match_result['similarity']
                
                is_correct = (matched_id == test_data['person_id'])
                if is_correct:
                    correct_matches += 1
                
                status = "✓" if is_correct else "✗"
                print(f"  {status} {test_data['person_name']:8s} -> {matched_id} "
                      f"(conf: {confidence:.3f}, sim: {similarity:.3f}) "
                      f"{'CORRECT' if is_correct else 'INCORRECT'}")
            else:
                print(f"  ? {test_data['person_name']:8s} -> No match found")
        
        accuracy = correct_matches / total_tests if total_tests > 0 else 0
        print(f"\n📈 Recognition Accuracy: {correct_matches}/{total_tests} "
              f"({accuracy:.1%})")
    
    def demonstrate_clustering_analysis(self):
        """Demonstrate clustering analysis for each person"""
        
        print("\n🔗 Clustering Analysis Demonstration")
        print("-" * 50)
        
        for person in self.demo_persons:
            person_stats = self.person_db.get_person_statistics(person['id'])
            
            if person_stats.get('active_embeddings', 0) > 1:
                print(f"\nClustering analysis for {person['name']}:")
                
                cluster_result = self.person_db.cluster_person_embeddings(person['id'])
                
                if 'error' not in cluster_result:
                    clusters = cluster_result['clusters']
                    noise_points = cluster_result['noise_points']
                    total_embeddings = cluster_result['total_embeddings']
                    
                    print(f"  Total embeddings: {total_embeddings}")
                    print(f"  Identified clusters: {clusters}")
                    print(f"  Noise points: {noise_points}")
                    
                    if clusters > 1:
                        print(f"  ⚠ Multiple gait patterns detected for {person['name']}")
                    elif clusters == 1:
                        print(f"  ✓ Consistent gait pattern for {person['name']}")
                    else:
                        print(f"  ? No clear gait patterns identified")
                else:
                    print(f"  Error in clustering: {cluster_result['error']}")
            else:
                print(f"\n{person['name']}: Insufficient embeddings for clustering")
    
    def create_quality_visualization(self, quality_results: List[Dict]):
        """Create visualization of quality assessment results"""
        
        print("\n📊 Creating Quality Visualization...")
        
        try:
            # Prepare data for visualization
            persons = [r['person'] for r in quality_results]
            scores = [r['overall_score'] for r in quality_results]
            quality_types = [r['quality_type'] for r in quality_results]
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Gait Sequence Quality Assessment Results', fontsize=16)
            
            # 1. Quality scores by person
            persons_unique = list(set(persons))
            person_scores = {person: [r['overall_score'] for r in quality_results if r['person'] == person] 
                           for person in persons_unique}
            
            ax1.boxplot([person_scores[person] for person in persons_unique], 
                       labels=persons_unique)
            ax1.set_title('Quality Score Distribution by Person')
            ax1.set_ylabel('Quality Score')
            ax1.grid(True, alpha=0.3)
            
            # 2. Quality scores by intended quality type
            quality_types_unique = list(set(quality_types))
            quality_type_scores = {qt: [r['overall_score'] for r in quality_results if r['quality_type'] == qt] 
                                 for qt in quality_types_unique}
            
            ax2.boxplot([quality_type_scores[qt] for qt in quality_types_unique], 
                       labels=quality_types_unique)
            ax2.set_title('Quality Score by Intended Quality Level')
            ax2.set_ylabel('Quality Score')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # 3. Component scores heatmap
            component_names = ['length', 'silhouette', 'motion', 'temporal', 'completeness', 'pose', 'sharpness']
            component_matrix = []
            
            for result in quality_results:
                row = [result['components'].get(comp, 0) for comp in component_names]
                component_matrix.append(row)
            
            im = ax3.imshow(component_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax3.set_title('Component Scores Heatmap')
            ax3.set_xlabel('Quality Components')
            ax3.set_ylabel('Sequence Index')
            ax3.set_xticks(range(len(component_names)))
            ax3.set_xticklabels(component_names, rotation=45)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax3)
            cbar.set_label('Score')
            
            # 4. Acceptance rate by quality type
            acceptance_rates = {}
            for qt in quality_types_unique:
                qt_results = [r for r in quality_results if r['quality_type'] == qt]
                acceptance_rate = np.mean([r['is_acceptable'] for r in qt_results])
                acceptance_rates[qt] = acceptance_rate
            
            bars = ax4.bar(acceptance_rates.keys(), acceptance_rates.values(), 
                          color=['green' if rate > 0.7 else 'orange' if rate > 0.3 else 'red' 
                                for rate in acceptance_rates.values()])
            ax4.set_title('Acceptance Rate by Quality Type')
            ax4.set_ylabel('Acceptance Rate')
            ax4.set_ylim(0, 1)
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, rate in zip(bars, acceptance_rates.values()):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{rate:.1%}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('quality_assessment_demo.png', dpi=150, bbox_inches='tight')
            print("  ✓ Visualization saved as 'quality_assessment_demo.png'")
            plt.close()
            
        except Exception as e:
            print(f"  ⚠ Could not create visualization: {e}")
    
    def run_comprehensive_demo(self):
        """Run the complete demonstration"""
        
        print("🌟 Starting Comprehensive Gait Recognition Demo")
        print("=" * 60)
        
        # Generate demonstration data
        self.generate_demo_data()
        
        # Demonstrate quality assessment
        quality_results = self.demonstrate_quality_assessment()
        
        # Demonstrate database operations
        self.demonstrate_database_operations()
        
        # Demonstrate recognition accuracy
        self.demonstrate_recognition_accuracy()
        
        # Demonstrate clustering analysis
        self.demonstrate_clustering_analysis()
        
        # Create visualizations
        self.create_quality_visualization(quality_results)
        
        # Final database cleanup and statistics
        print("\n🧹 Final Database Cleanup")
        print("-" * 50)
        cleanup_stats = self.person_db.cleanup_database()
        print(f"Cleanup results: {cleanup_stats}")
        
        final_stats = self.person_db.get_person_statistics()
        print(f"\nFinal database state:")
        print(f"  Persons: {final_stats.get('total_persons', 0)}")
        print(f"  Active embeddings: {final_stats.get('total_active_embeddings', 0)}")
        print(f"  Average quality: {final_stats.get('average_quality', 0.0):.3f}")
        
        print("\n" + "=" * 60)
        print("🎉 Comprehensive Demo Completed!")
        print("=" * 60)
        
        # Cleanup demo database
        try:
            os.remove("demo_person_embeddings.db")
            print("Demo database cleaned up")
        except:
            pass

def main():
    """Run the comprehensive demo"""
    demo = GaitRecognitionDemo()
    demo.run_comprehensive_demo()

if __name__ == "__main__":
    main()
