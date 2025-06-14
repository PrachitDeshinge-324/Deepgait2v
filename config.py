"""
Configuration settings for the tracking application
"""

# Paths
MODEL_PATH = 'weights/yolo11m.pt'
SEG_MODEL_PATH = 'weights/yolo11m-seg.pt'
DEEPGAITV2_MODEL_PATH = "weights/DeepGaitV2_30_DA-50000.pt"
GAITBASE_MODEL_PATH = "weights/GaitBase_DA-180000.pt"
VIDEO_PATH = '../Person_new/input/My Movie.mp4'

# Data storage paths
DATA_DIR = "data"  # Directory for storing data files like databases
OUTPUT_VIDEO_PATH = "output/processed_video_MM.mp4"  # Path for saving output video
OUTPUT_FRAMES_DIR = "output/frames"  # Directory for saving individual frames

# Person identification settings
SIMILARITY_THRESHOLD = 0.3
IDENTIFICATION_THRESHOLD = 0.15  # Similarity threshold for positive identification
SPATIAL_CONFLICT_THRESHOLD = 150  # Pixel distance threshold for spatial conflict detection

# Processing limits
MAX_FRAMES = 6000
SAVE_VIDEO = True  # Whether to save processed video
SAVE_FRAMES = False  # Whether to save individual frames
SHOW_DISPLAY = True  # Whether to show display window
VERBOSE = False  # Whether to show detailed processing logs
SHOW_PROGRESS = True  # Whether to show progress bar

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

# Model Selection and Configuration
GAIT_MODEL_TYPE = "GaitBase"  # Options: "DeepGaitV2" or "GaitBase" - Change this to switch models

# DeepGaitV2 Configuration
# This is a CNN-based model with excellent performance on various datasets
DEEPGAITV2_CONFIG = {
    'Backbone': {
        'in_channels': 1,
        'mode': '2d',  # Changed from 'p3d' to '2d' for MPS compatibility and better performance
        'layers': [1, 4, 4, 1],  # Layer configuration: [layer1, layer2, layer3, layer4]
        'channels': [64, 128, 256, 512]  # Channel progression through layers
    },
    'SeparateBNNecks': {
        'class_num': 3000  # Number of identities in training set - adjust based on your trained model
    },
    'use_emb2': False  # Whether to use second embedding layer output for inference
}

# GaitBase Configuration 
# This is the baseline model providing strong performance with simpler architecture
GAITBASE_CONFIG = {
    'backbone_cfg': {
        'type': 'ResNet9',           # ResNet-9 backbone architecture
        'block': 'BasicBlock',       # Use BasicBlock (no bottleneck)
        'channels': [64, 128, 256, 512],  # Channel progression
        'layers': [1, 1, 1, 1],      # Number of blocks per layer
        'strides': [1, 2, 2, 1],     # Stride for each layer
        'maxpool': False             # Disable max pooling in the first layer
    },
    'SeparateFCs': {
        'in_channels': 512,          # Input channels from backbone
        'out_channels': 256,         # Output feature dimension
        'parts_num': 16              # Number of horizontal parts for HPP
    },
    'SeparateBNNecks': {
        'class_num': 3000,           # Number of identities - adjust based on your trained model
        'in_channels': 256,          # Input channels from SeparateFCs
        'parts_num': 16              # Number of parts (should match SeparateFCs)
    },
    'bin_num': [16]                  # Horizontal pooling pyramid bins
}

# Get the appropriate config and model path based on selected model type
def get_current_model_config():
    """Get the configuration for the currently selected model"""
    if GAIT_MODEL_TYPE == "DeepGaitV2":
        return DEEPGAITV2_CONFIG
    elif GAIT_MODEL_TYPE == "GaitBase":
        return GAITBASE_CONFIG
    else:
        raise ValueError(f"Unknown model type: {GAIT_MODEL_TYPE}. Use 'DeepGaitV2' or 'GaitBase'")

def get_current_model_path():
    """Get the model path for the currently selected model"""
    if GAIT_MODEL_TYPE == "DeepGaitV2":
        return DEEPGAITV2_MODEL_PATH
    elif GAIT_MODEL_TYPE == "GaitBase":
        return GAITBASE_MODEL_PATH
    else:
        raise ValueError(f"Unknown model type: {GAIT_MODEL_TYPE}. Use 'DeepGaitV2' or 'GaitBase'")

# Dynamic config assignment based on selected model
GAIT_RECOGNIZER_CONFIG = get_current_model_config()
GAIT_MODEL_PATH = get_current_model_path()

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