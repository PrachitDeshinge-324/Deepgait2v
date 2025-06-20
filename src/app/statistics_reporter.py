"""
Statistics reporting for the gait recognition system
"""

from datetime import datetime
import numpy as np
import os
import time

class StatisticsReporter:
    """Generates and reports statistics"""
    
    def __init__(self):
        """Initialize statistics reporter"""
        self.start_time = datetime.now()
        self.total_frames = 0
        self.total_tracks = 0
        self.total_identities = 0
        self.track_durations = []
        self.quality_scores = []
        self.confidence_scores = []
    
    def report_statistics(self, frame_count, track_identities, person_db):
        """
        Generate and report statistics
        
        Args:
            frame_count: Total frames processed
            track_identities: Dictionary of track identities
            person_db: Person database
        """
        self.total_frames = frame_count
        self.total_tracks = len(track_identities)
        self.total_identities = len(person_db)
        
        # Calculate duration
        processing_time = (datetime.now() - self.start_time).total_seconds()
        fps = frame_count / processing_time if processing_time > 0 else 0
        
        # Print processing summary
        print("\n" + "="*50)
        print("PROCESSING STATISTICS")
        print("="*50)
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Frames processed: {frame_count}")
        print(f"Average FPS: {fps:.2f}")
        print(f"Total tracks: {len(track_identities)}")
        print(f"Total identities in database: {len(person_db)}")
        
        # Print identity statistics
        if track_identities:
            # Calculate confidence stats
            confidences = [info['confidence'] for info in track_identities.values() if 'confidence' in info]
            if confidences:
                print(f"Average identification confidence: {np.mean(confidences):.3f}")
                print(f"Min confidence: {np.min(confidences):.3f}, Max confidence: {np.max(confidences):.3f}")
            
            # Calculate quality stats
            qualities = [info.get('quality', 0.0) for info in track_identities.values()]
            if qualities:
                print(f"Average sequence quality: {np.mean(qualities):.3f}")
                print(f"Min quality: {np.min(qualities):.3f}, Max quality: {np.max(qualities):.3f}")
            
            # Count new vs. existing identities
            new_identities = sum(1 for info in track_identities.values() if info.get('is_new', False))
            existing_identities = sum(1 for info in track_identities.values() if not info.get('is_new', True))
            print(f"New identities: {new_identities}, Recognized identities: {existing_identities}")
        
        # Print database statistics
        db_stats = person_db.get_stats() if hasattr(person_db, 'get_stats') else {}
        if db_stats:
            print("\nDATABASE STATISTICS")
            print("-"*50)
            print(f"Total persons: {db_stats.get('count', 0)}")
            print(f"Persons with face data: {db_stats.get('multimodal_count', 0)}")
            print(f"Average quality: {db_stats.get('avg_quality', 0.0):.3f}")
            print(f"Last update: {db_stats.get('last_update', 'N/A')}")
        
        print("="*50 + "\n")
        
        # Generate visual statistics if matplotlib is available
        try:
            self._generate_visual_statistics(track_identities, person_db)
        except ImportError:
            print("Matplotlib not available, skipping visual statistics")
    
    def _generate_visual_statistics(self, track_identities, person_db):
        """Generate visual statistics using matplotlib"""
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import time
        
        plt.figure(figsize=(12, 8))
        
        # Get persons data safely
        persons = []
        if hasattr(person_db, 'get_persons'):
            persons = person_db.get_persons()
        elif hasattr(person_db, 'persons'):
            # person_db.persons is a dictionary, so get the values
            persons = list(person_db.persons.values())
        
        # Process each person correctly by checking data type
        valid_persons = []
        for person in persons:
            if isinstance(person, dict):
                valid_persons.append(person)
            elif isinstance(person, str):
                # Handle string entries - might need to parse them as JSON if they're serialized dictionaries
                try:
                    import json
                    parsed = json.loads(person)
                    if isinstance(parsed, dict):
                        valid_persons.append(parsed)
                except:
                    print(f"Could not parse person data: {str(person)[:30]}...")
            else:
                # Handle other types appropriately
                try:
                    # Try to convert to dictionary if it has __dict__ attribute
                    if hasattr(person, '__dict__'):
                        valid_persons.append(person.__dict__)
                    elif hasattr(person, 'to_dict'):
                        valid_persons.append(person.to_dict())
                except:
                    pass
    
        # 1. Identification Modalities Pie Chart
        plt.subplot(2, 2, 1)
        face_count = sum(1 for person in valid_persons if person.get('has_face', False))
        gait_only = len(valid_persons) - face_count
        
        # Check if we have any data before creating the pie
        if face_count > 0 or gait_only > 0:
            plt.pie([face_count, gait_only],
                    labels=['Face + Gait', 'Gait Only'],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['#66b3ff', '#99ff99'])
            plt.axis('equal')
        else:
            plt.text(0.5, 0.5, 'No data available', 
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=plt.gca().transAxes)
        
        plt.title('Identification Modalities')
        
        # 2. Quality Distribution Histogram
        plt.subplot(2, 2, 2)
        qualities = [person.get('quality', 0) for person in valid_persons]
        
        if qualities:
            plt.hist(qualities, bins=10, range=(0, 1), edgecolor='black', alpha=0.7)
            plt.xlabel('Quality Score')
            plt.ylabel('Number of Identities')
            plt.title('Identity Quality Distribution')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No quality data available', 
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=plt.gca().transAxes)
        
        # 3. Confidence Bar Chart
        plt.subplot(2, 2, 3)
        confidences = []
        labels = []
        
        for track_id, identity in track_identities.items():
            if isinstance(identity, dict) and 'confidence' in identity and 'name' in identity:
                confidences.append(identity['confidence'])
                labels.append(f"{identity['name']} (ID:{track_id})")
        
        if confidences:
            # Sort by confidence
            sorted_indices = np.argsort(confidences)
            sorted_confidences = [confidences[i] for i in sorted_indices]
            sorted_labels = [labels[i] for i in sorted_indices]
            
            # Limit to top 10 if there are many
            if len(sorted_confidences) > 10:
                sorted_confidences = sorted_confidences[-10:]
                sorted_labels = sorted_labels[-10:]
            
            # Create bar chart
            y_pos = range(len(sorted_labels))
            plt.barh(y_pos, sorted_confidences, align='center', alpha=0.7)
            plt.yticks(y_pos, sorted_labels)
            plt.xlabel('Confidence Score')
            plt.title('Identity Confidence Scores')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No confidence data available', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes)
        
        # 4. Time/Frame Analysis or Additional Metrics
        plt.subplot(2, 2, 4)
        
        # Get times from database if available
        creation_times = [person.get('created_at', 0) for person in valid_persons 
                         if 'created_at' in person]
        
        if creation_times:
            # Convert to relative time (hours passed)
            now = time.time()
            hours_ago = [(now - t) / 3600 for t in creation_times]
            
            plt.hist(hours_ago, bins=24, edgecolor='black', alpha=0.7)
            plt.xlabel('Hours Ago')
            plt.ylabel('Number of New Identities')
            plt.title('Identity Creation Timeline')
            plt.grid(True, alpha=0.3)
        else:
            # If no time data, show system stats
            sys_stats = [
                f"Database Size: {len(valid_persons)} persons",
                f"Active Tracks: {len(track_identities)}",
                f"Face Recognition: {'Enabled' if face_count > 0 else 'Disabled'}",
                f"Data Collection: Active"
            ]
            
            plt.axis('off')  # Turn off axes
            for i, stat in enumerate(sys_stats):
                plt.text(0.1, 0.8 - (i * 0.2), stat, fontsize=12)
            
            plt.title('System Statistics')
    
        # Adjust layout and add overall title
        plt.tight_layout()
        plt.suptitle('Gait Recognition System Analytics', fontsize=16, y=0.98)
        plt.subplots_adjust(top=0.9)
        
        # Make sure output directory exists
        output_dir = 'output/statistics'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the plot
        stats_path = os.path.join(output_dir, f"statistics_{int(time.time())}.png")
        plt.savefig(stats_path)
        
        if hasattr(self, 'logger'):
            self.logger.info(f"Visual statistics saved to {stats_path}")
        else:
            print(f"Visual statistics saved to {stats_path}")
        
        # Close the plot to avoid memory leaks
        plt.close()
        
        return stats_path