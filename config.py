"""
Configuration settings for the tracking application
"""

# Paths
MODEL_PATH = 'weights/yolo11m.pt'
SEG_MODEL_PATH = 'weights/yolo11l-seg.pt'
# Model Selection and Configuration
GAIT_MODEL_TYPE = "DeepGaitV2"  # Start with best-performing model for your CCTV setup
DEEPGAITV2_MODEL_PATH = "weights/DeepGaitV2_30_DA-50000.pt"
GAITBASE_MODEL_PATH = "weights/GaitBase_DA-180000.pt"
SKELETONGAITPP_MODEL_PATH = "weights/SkeletonGaitPP_30_DA-50000.pt"
VIDEO_DIR = '../Person_new/input'
VIDEO_NAME = '3c.mp4'
VIDEO_PATH = f"{VIDEO_DIR}/{VIDEO_NAME}"  # Path to the input video file

# Data storage paths
DATA_DIR = f"data_{GAIT_MODEL_TYPE}"  # Directory for storing data files like databases
OUTPUT_VIDEO_PATH = f"output/processed_video_{VIDEO_NAME}_{GAIT_MODEL_TYPE}.mp4"  # Path for saving output video
OUTPUT_FRAMES_DIR = "output/frames"  # Directory for saving individual frames

# Person identification settings - Basic optimization for CCTV
# Start with more permissive settings that work well for your DeepGaitV2
SIMILARITY_THRESHOLD = 0.20  # Slightly higher for better discrimination
IDENTIFICATION_THRESHOLD = 0.15  # Balanced threshold for reliability
SPATIAL_CONFLICT_THRESHOLD = 150  # Pixel distance threshold for spatial conflict detection

# CCTV-specific quality settings - Basic level
MIN_SEQUENCE_LENGTH = 15  # Reasonable minimum for CCTV
MIN_QUALITY_THRESHOLD = 0.40  # More permissive for CCTV quality
HIGH_QUALITY_THRESHOLD = 0.60  # Realistic for CCTV expectations

# Sampling method configuration
IDENTIFICATION_METHOD = "nucleus"  # Options: "top_k" or "nucleus"
NUCLEUS_TOP_P = 0.85  # For nucleus sampling: cumulative probability mass (0.8-0.95 recommended)
NUCLEUS_MIN_CANDIDATES = 1  # Minimum candidates to return
NUCLEUS_MAX_CANDIDATES = 5  # Maximum candidates to return
TOP_K_CANDIDATES = 5  # For top-k sampling: fixed number of candidates

# Advanced nucleus sampling parameters for close similarities
NUCLEUS_CLOSE_SIM_THRESHOLD = 0.08  # Increased for CCTV scenarios with similar poses
NUCLEUS_AMPLIFICATION_FACTOR = 35.0  # Higher amplification for better discrimination
NUCLEUS_QUALITY_WEIGHT = 0.8  # Increased quality weighting for CCTV scenarios
NUCLEUS_ENHANCED_RANKING = True  # Enable multi-factor ranking for close similarities

# CCTV-specific enhancement parameters
TEMPORAL_CONSISTENCY_WEIGHT = 0.7  # Higher weight for temporal consistency
ENSEMBLE_IDENTIFICATION = True  # Enable ensemble methods for better accuracy
MULTI_SEQUENCE_MATCHING = True  # Use multiple sequences per person for matching

# Processing limits
MAX_FRAMES = 600  # Increased for better gallery building
SAVE_VIDEO = True  # Whether to save processed video
SAVE_FRAMES = False  # Whether to save individual frames
SHOW_DISPLAY = False  # Whether to show display window
VERBOSE = False  # Enable detailed logs for debugging
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

# Silhouette extraction settings - Enhanced for better accuracy
SILHOUETTE_CONFIG = {
    'min_track_frames': 12,        # Reduced from 15 for more permissive tracking
    'confidence_threshold': 0.6,   # Reduced for CCTV footage
    'min_quality_score': 0.3,      # More permissive quality threshold  
    'resolution': (64, 44),
    'window_size': 25,             # Reduced from 30 for easier sequence finding
    'max_cache_size': 150,         # Increased for better temporal consistency
    
    # Enhanced morphological operation settings
    'small_kernel_size': 3,        # Reduced from 5 to preserve details
    'use_bilateral_filter': True,  # Enable edge preservation
    'bilateral_d': 5,              # Bilateral filter neighborhood
    'bilateral_sigma_color': 50,   # Bilateral filter color sigma
    'bilateral_sigma_space': 50,   # Bilateral filter space sigma
    
    # Temporal consistency settings
    'temporal_consistency_enabled': True,
    'temporal_iou_threshold': 0.7, # IoU threshold for temporal blending
    'temporal_blend_alpha': 0.8,   # Weight for current frame vs previous
}

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

# SkeletonGait++ Configuration
# This model uses both pose heatmaps and silhouettes for enhanced gait recognition
SKELETONGAITPP_CONFIG = {
    'Backbone': {
        'in_channels': 3,            # 2 channels for pose heatmaps + 1 for silhouette
        'blocks': [1, 4, 4, 1],      # Layer configuration: [layer1, layer2, layer3, layer4]
        'C': 2                       # Channel multiplier
    },
    'SeparateBNNecks': {
        'class_num': 3000            # Number of identities in training set - adjust based on your trained model
    },
    'use_emb2': False                # Whether to use second embedding layer output for inference
}

# SkeletonGaitPP Multimodal Preprocessing Configuration
SKELETONGAITPP_MULTIMODAL_PREPROCESSING = {
    'enabled': True,
    'pose_intensity': 0.8,           # Increased intensity for pose heatmaps
    'pose_sigma': 2.0,               # Gaussian sigma for pose keypoints
    'quality_threshold': 0.35,       # Lower threshold for synthetic pose data
    'temporal_consistency': True,    # Enable temporal consistency for poses
    'pose_amplification': 1.5,       # Amplify pose signals for better detection
}

# Get the appropriate config and model path based on selected model type
def get_current_model_config():
    """Get the configuration for the currently selected model"""
    if GAIT_MODEL_TYPE == "DeepGaitV2":
        return DEEPGAITV2_CONFIG
    elif GAIT_MODEL_TYPE == "GaitBase":
        return GAITBASE_CONFIG
    elif GAIT_MODEL_TYPE == "SkeletonGaitPP":
        return SKELETONGAITPP_CONFIG
    else:
        raise ValueError(f"Unknown model type: {GAIT_MODEL_TYPE}. Use 'DeepGaitV2', 'GaitBase', or 'SkeletonGaitPP'")

def get_current_model_path():
    """Get the model path for the currently selected model"""
    if GAIT_MODEL_TYPE == "DeepGaitV2":
        return DEEPGAITV2_MODEL_PATH
    elif GAIT_MODEL_TYPE == "GaitBase":
        return GAITBASE_MODEL_PATH
    elif GAIT_MODEL_TYPE == "SkeletonGaitPP":
        return SKELETONGAITPP_MODEL_PATH
    else:
        raise ValueError(f"Unknown model type: {GAIT_MODEL_TYPE}. Use 'DeepGaitV2', 'GaitBase', or 'SkeletonGaitPP'")

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