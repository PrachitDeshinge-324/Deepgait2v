"""
Configuration settings for the tracking application
"""

# Paths
MODEL_PATH = '../Person_New/weights/yolo11m.pt'
SEG_MODEL_PATH = 'weights/yolo11m-seg.pt'
VIDEO_PATH = '../Person_New/input/3c1.mp4'
SIMILARITY_THRESHOLD = 0.3
MAX_FRAMES = 1000

# Tracker settings
TRACKER_CONFIG = {
    'frame_rate': 30,
    'track_thresh': 0.6, 
    'high_thresh': 0.6, 
    'match_thresh': 0.7
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'box_color': (0, 255, 0),  # Green
    'text_color': (0, 255, 0),  # Green
    'fps_color': (0, 0, 255),   # Red
    'line_thickness': 2,
    'font_scale': 0.5,
    'fps_font_scale': 0.8
}

# Silhouette extraction settings
SILHOUETTE_CONFIG = {
    'min_track_frames': 15,        # Reduced from 20
    'confidence_threshold': 0.7,   # Reduced from 0.8
    'min_quality_score': 0.6,      # Reduced from 0.7
    'resolution': (64, 44),
    'window_size': 25,             # Reduced from 30 for easier sequence finding
    'max_cache_size': 50
}

# Update the GAIT_RECOGNIZER_CONFIG to use 2D mode
GAIT_RECOGNIZER_CONFIG = {
    'Backbone': {
        'in_channels': 1,
        'mode': '2d',  # Changed from 'p3d' to '2d' for MPS compatibility
        'layers': [1, 4, 4, 1],
        'channels': [64, 128, 256, 512]
    },
    'SeparateBNNecks': {
        'class_num': 3000  # Adjust based on your trained model
    },
    'use_emb2': False
}

GAIT_MODEL_PATH = "../Person_Temp/DeepGaitV2_30_DA-50000.pt"  # Update this path

# Quality Assessment Configuration
QUALITY_ASSESSOR_CONFIG = {
    'min_sequence_length': 15,      # Minimum frames for reliable gait analysis
    'max_sequence_length': 120,     # Maximum frames to avoid redundancy
    'min_silhouette_area': 800,     # Minimum silhouette area in pixels
    'max_silhouette_area': 40000,   # Maximum silhouette area in pixels
    'min_aspect_ratio': 0.4,        # Minimum height/width ratio
    'max_aspect_ratio': 3.5,        # Maximum height/width ratio
    'min_motion_threshold': 15,     # Minimum motion between frames
    'max_motion_threshold': 150,    # Maximum motion (too fast movement)
    'consistency_threshold': 0.75,  # Silhouette consistency across frames
    'completeness_threshold': 0.8,  # Percentage of complete silhouettes
    'sharpness_threshold': 60,      # Minimum sharpness (Laplacian variance)
    'temporal_consistency_window': 5, # Window for temporal consistency check
    'gait_cycle_min_frames': 12,    # Minimum frames for one gait cycle
    'pose_variation_threshold': 0.35, # Minimum pose variation for gait analysis
}

# Person Database Configuration
PERSON_DATABASE_CONFIG = {
    'max_embeddings_per_person': 8,     # Maximum embeddings to store per person
    'min_quality_threshold': 0.55,      # Minimum quality score to store
    'similarity_threshold': 0.90,       # Threshold for considering embeddings similar (increased for better discrimination)
    'clustering_eps': 0.18,             # DBSCAN epsilon for clustering
    'clustering_min_samples': 2,        # DBSCAN minimum samples
    'auto_cleanup_threshold': 150,      # Auto cleanup when database exceeds this size
    'duplicate_detection': True,        # Enable duplicate detection
    'quality_decay_factor': 0.95,      # Quality decay for old embeddings
    'max_age_days': 30,                # Maximum age for embeddings (days)
}

# Enhanced recognition thresholds
RECOGNITION_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for positive identification
HIGH_QUALITY_THRESHOLD = 0.60           # Threshold for high-quality sequences (adjusted for real video)
NEW_PERSON_QUALITY_THRESHOLD = 0.60     # Minimum quality to create new person (adjusted for real video)