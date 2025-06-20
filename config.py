"""
Configuration settings for the multimodal gait and face recognition system.

This module contains all configuration parameters organized by functionality:
- Model paths and selection
- Processing parameters  
- Recognition thresholds
- Multimodal fusion settings
- System performance settings
"""

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

# Model Selection
GAIT_MODEL_TYPE = "DeepGaitV2"  # Options: "DeepGaitV2", "GaitBase", "SkeletonGaitPP"
GAIT_MODEL_BACKBONE_MODE = "p3d"  # Backbone mode (if supported by loader)

# Model Paths
MODEL_PATH = 'weights/yolo11m.pt'  # YOLO detection model
SEG_MODEL_PATH = 'weights/yolo11l-seg.pt'  # YOLO segmentation model
DEEPGAITV2_MODEL_PATH = "weights/DeepGaitV2_30_DA-50000.pt"
GAITBASE_MODEL_PATH = "weights/GaitBase_DA-180000.pt"
SKELETONGAITPP_MODEL_PATH = "weights/SkeletonGaitPP_30_DA-50000.pt"

# ==============================================================================
# INPUT/OUTPUT PATHS
# ==============================================================================

# Video Input
VIDEO_DIR = '../Person_new/input'
VIDEO_NAME = '3c.mp4'
VIDEO_PATH = f"{VIDEO_DIR}/{VIDEO_NAME}"

# Data Storage
DATA_DIR = f"data_{GAIT_MODEL_TYPE}"  # Database storage directory
OUTPUT_VIDEO_PATH = f"output/processed_video_{VIDEO_NAME}_{GAIT_MODEL_TYPE}.mp4"
OUTPUT_FRAMES_DIR = "output/frames"
OUTPUT_DIR = "output/statistics"

# ==============================================================================
# RECOGNITION THRESHOLDS & PARAMETERS
# ==============================================================================

# Basic Recognition Thresholds
SIMILARITY_THRESHOLD = 0.15  # Gait similarity threshold
IDENTIFICATION_THRESHOLD = 0.15  # General identification threshold
RECOGNITION_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for positive ID
SPATIAL_CONFLICT_THRESHOLD = 150  # Pixel distance for spatial conflict detection

# Quality Thresholds
MIN_SEQUENCE_LENGTH = 15  # Minimum frames for gait analysis
MIN_QUALITY_THRESHOLD = 0.40  # Minimum quality for processing
HIGH_QUALITY_THRESHOLD = 0.60  # Threshold for high-quality sequences
NEW_PERSON_QUALITY_THRESHOLD = 0.60  # Minimum quality to create new person

# ==============================================================================
# IDENTIFICATION METHODS & SAMPLING
# ==============================================================================

# Sampling Configuration
IDENTIFICATION_METHOD = "nucleus"  # Options: "top_k" or "nucleus"
NUCLEUS_TOP_P = 0.90  # Cumulative probability mass for nucleus sampling
NUCLEUS_MIN_CANDIDATES = 1  # Minimum candidates to return
NUCLEUS_MAX_CANDIDATES = 5  # Maximum candidates to return
TOP_K_CANDIDATES = 5  # Fixed number of candidates for top-k sampling

# Advanced Nucleus Sampling Parameters
NUCLEUS_CLOSE_SIM_THRESHOLD = 0.08  # Threshold for close similarities
NUCLEUS_AMPLIFICATION_FACTOR = 35.0  # Amplification for discrimination
NUCLEUS_QUALITY_WEIGHT = 0.8  # Quality weighting factor
NUCLEUS_ENHANCED_RANKING = True  # Enable multi-factor ranking

# ==============================================================================
# FACE RECOGNITION CONFIGURATION
# ==============================================================================

# Face Recognition Settings
ENABLE_FACE_RECOGNITION = True  # Enable/disable face recognition
FACE_DETECTION_MODEL = 'buffalo_l'  # InsightFace model
FACE_RECOGNITION_MODEL = 'buffalo_l'  # Face recognition model
FACE_DETECTION_SIZE = (320, 320)  # Detection resolution
FACE_THRESHOLD = 0.3  # Face recognition similarity threshold (updated)
FACE_QUALITY_THRESHOLD = 0.15  # Minimum face quality threshold

# Face Processing Parameters
FACE_CACHE_SIZE = 30  # Number of faces to cache per track
FACE_QUALITY_WINDOW = 10  # Frames to wait before selecting best face
FACE_CROP_UPPER_BODY_RATIO = 0.6  # Focus on upper body for detection
FACE_MIN_SIZE = 50  # Minimum face size in pixels
FACE_MAX_EDGE_DISTANCE = 10  # Min distance from crop edge for valid face
FACE_BLUR_THRESHOLD = 500  # Minimum Laplacian variance for sharpness

# Face Recognition Enhancement
FACE_EMBEDDING_NORMALIZATION = True  # Normalize face embeddings
FACE_TEMPORAL_CONSISTENCY = True  # Use temporal consistency

# ==============================================================================
# MULTIMODAL FUSION CONFIGURATION
# ==============================================================================

# Fusion Weights (updated based on audit)
FACE_WEIGHT = 0.7  # Face recognition weight in fusion (increased)
GAIT_WEIGHT = 0.3  # Gait recognition weight in fusion (decreased)
REQUIRE_BOTH_MODALITIES = False  # Require both modalities for ID

# Proximity-Aware Recognition
PROXIMITY_AWARENESS = True  # Enable proximity-aware identification
PROXIMITY_THRESHOLD = 100  # Pixel height for close-to-camera detection
DISTANT_IDENTITY_CHANGE_THRESHOLD = 0.3  # Extra confidence needed when distant
PROXIMITY_FACE_PRIORITY = True  # Prioritize face when close to camera

# ==============================================================================
# CROSS-CAMERA DOMAIN ADAPTATION (ENHANCED)
# ==============================================================================

# Current Camera Configuration
CURRENT_CAMERA_ID = "camera_1"  # Set this to different values for different cameras

# Domain Adaptation Settings
ENABLE_CROSS_CAMERA_ADAPTATION = True  # Enable cross-camera domain adaptation
ENABLE_PCA_WHITENING = True  # Enable PCA whitening for noise reduction
ENABLE_CROSS_CAMERA_NORM = True  # Enable cross-camera normalization
MIN_ADAPTATION_SAMPLES = 5  # Reduced for faster adaptation

# Camera-Invariant Processing
ENABLE_CAMERA_INVARIANT_PREPROCESSING = True  # Enable preprocessing
CAMERA_ADAPTIVE_SIMILARITY = True  # Use camera-adaptive similarity metrics
DOMAIN_ADAPTATION_STRENGTH = 0.8  # Increased strength for better adaptation (0-1)

# Enhanced Cross-Camera Similarity Thresholds
CROSS_CAMERA_GAIT_THRESHOLD = 0.12  # More lenient for gait across cameras
CROSS_CAMERA_FACE_THRESHOLD = 0.30  # More lenient for face across cameras
SAME_CAMERA_BONUS = 0.15  # Increased bonus for same-camera matches

# Advanced Domain Adaptation Parameters
CROSS_CAMERA_EMBEDDING_SMOOTHING = 0.7  # Smoothing factor for embeddings
CAMERA_BIAS_CORRECTION = True  # Enable camera bias correction
TEMPORAL_DOMAIN_ADAPTATION = True  # Enable temporal adaptation
ADAPTIVE_THRESHOLD_SCALING = True  # Scale thresholds based on camera distance

# Multi-Camera Fusion Weights
CROSS_CAMERA_FACE_WEIGHT = 0.8  # Higher weight for face in cross-camera
CROSS_CAMERA_GAIT_WEIGHT = 0.2  # Lower weight for gait in cross-camera
SAME_CAMERA_FACE_WEIGHT = 0.7   # Standard weight for same camera
SAME_CAMERA_GAIT_WEIGHT = 0.3   # Standard weight for same camera

# ==============================================================================
# SYSTEM PERFORMANCE & PROCESSING
# ==============================================================================

# Processing Limits
MAX_FRAMES = 600  # Maximum frames to process
TEMPORAL_CONSISTENCY_WEIGHT = 0.7  # Weight for temporal consistency
ENSEMBLE_IDENTIFICATION = True  # Enable ensemble methods
MULTI_SEQUENCE_MATCHING = True  # Use multiple sequences per person

# Output Control
SAVE_VIDEO = True  # Save processed video
SAVE_FRAMES = False  # Save individual frames
SHOW_DISPLAY = True  # Show display window
VERBOSE = False  # Enable detailed logging
SHOW_PROGRESS = True  # Show progress bar
SHOW_STATISTICS_PLOTS = False  # Show statistics plots

# ==============================================================================
# GAIT MODEL CONFIGURATIONS
# ==============================================================================

# DeepGaitV2 Configuration
DEEPGAITV2_CONFIG = {
    'Backbone': {
        'in_channels': 1,
        'mode': '2d',  # 2D mode for MPS compatibility
        'layers': [1, 4, 4, 1],  # Layer configuration
        'channels': [64, 128, 256, 512]  # Channel progression
    },
    'SeparateBNNecks': {
        'class_num': 3000  # Number of identities in training set
    },
    'use_emb2': False  # Use second embedding layer output
}

# GaitBase Configuration
GAITBASE_CONFIG = {
    'backbone_cfg': {
        'type': 'ResNet9',
        'block': 'BasicBlock',
        'channels': [64, 128, 256, 512],
        'layers': [1, 1, 1, 1],
        'strides': [1, 2, 2, 1],
        'maxpool': False
    },
    'SeparateFCs': {
        'in_channels': 512,
        'out_channels': 256,
        'parts_num': 16
    },
    'SeparateBNNecks': {
        'class_num': 3000,
        'in_channels': 256,
        'parts_num': 16
    },
    'bin_num': [16]
}

# SkeletonGait++ Configuration
SKELETONGAITPP_CONFIG = {
    'Backbone': {
        'in_channels': 3,  # 2 pose channels + 1 silhouette
        'blocks': [1, 4, 4, 1],
        'C': 2
    },
    'SeparateBNNecks': {
        'class_num': 3000
    },
    'use_emb2': False
}

# SkeletonGaitPP Multimodal Preprocessing
SKELETONGAITPP_MULTIMODAL_PREPROCESSING = {
    'enabled': True,
    'pose_intensity': 0.8,
    'pose_sigma': 2.0,
    'quality_threshold': 0.35,
    'temporal_consistency': True,
    'pose_amplification': 1.5,
}

# ==============================================================================
# TRACKER & VISUALIZATION SETTINGS
# ==============================================================================

# Tracker Configuration
TRACKER_CONFIG = {
    'frame_rate': 30,
    'track_thresh': 0.6,
    'high_thresh': 0.6,
    'match_thresh': 0.7
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'box_color': (0, 255, 0),  # Green
    'text_color': (0, 255, 0),  # Green
    'fps_color': (0, 0, 255),  # Red
    'line_thickness': 2,
    'font_scale': 0.5,
    'fps_font_scale': 0.8
}

# ==============================================================================
# SILHOUETTE EXTRACTION SETTINGS
# ==============================================================================

SILHOUETTE_CONFIG = {
    # Basic Parameters
    'min_track_frames': 12,
    'confidence_threshold': 0.6,
    'min_quality_score': 0.3,
    'resolution': (64, 44),
    'window_size': 25,
    'max_cache_size': 150,
    
    # Morphological Operations
    'small_kernel_size': 3,
    'use_bilateral_filter': True,
    'bilateral_d': 5,
    'bilateral_sigma_color': 50,
    'bilateral_sigma_space': 50,
    
    # Temporal Consistency
    'temporal_consistency_enabled': True,
    'temporal_iou_threshold': 0.7,
    'temporal_blend_alpha': 0.8,
}

# ==============================================================================
# QUALITY ASSESSMENT CONFIGURATION
# ==============================================================================

QUALITY_ASSESSOR_CONFIG = {
    # Sequence Parameters
    'min_sequence_length': 15,
    'max_sequence_length': 120,
    'gait_cycle_min_frames': 12,
    
    # Silhouette Quality
    'min_silhouette_area': 800,
    'max_silhouette_area': 40000,
    'min_aspect_ratio': 0.4,
    'max_aspect_ratio': 3.5,
    
    # Motion Analysis
    'min_motion_threshold': 15,
    'max_motion_threshold': 150,
    'pose_variation_threshold': 0.35,
    
    # Consistency Checks
    'consistency_threshold': 0.75,
    'completeness_threshold': 0.8,
    'sharpness_threshold': 60,
    'temporal_consistency_window': 5,
}

# ==============================================================================
# DATABASE CONFIGURATION
# ==============================================================================

PERSON_DATABASE_CONFIG = {
    # Storage Limits
    'max_embeddings_per_person': 8,
    'auto_cleanup_threshold': 150,
    'max_age_days': 30,
    
    # Quality Control
    'min_quality_threshold': 0.55,
    'quality_decay_factor': 0.95,
    
    # Similarity & Clustering
    'similarity_threshold': 0.90,
    'clustering_eps': 0.18,
    'clustering_min_samples': 2,
    
    # Features
    'duplicate_detection': True,
}

# ==============================================================================
# METRIC LEARNING CONFIGURATION
# ==============================================================================

METRIC_LEARNING_CONFIG = {
    'enabled': True,
    'model_path': 'weights/metric_learning/best_metric_model.pth',
    'input_dim': 512,
    'embedding_dim': 256,
    'similarity_threshold': 0.18,
    'amplification_factor': 3.0
}

# Update nucleus sampling for metric embeddings
NUCLEUS_AMPLIFICATION_FACTOR = 3.0  # Reduced for metric space
NUCLEUS_CLOSE_SIM_THRESHOLD = 0.10  # Adjusted for metric space

# ==============================================================================
# WARNING SUPPRESSION CONFIGURATION
# ==============================================================================

# Suppress common warnings from dependencies
SUPPRESS_WARNINGS = True  # Enable/disable warning suppression
SUPPRESSED_WARNING_CATEGORIES = [
    'FutureWarning',  # InsightFace rcond warnings
    'UserWarning',    # ONNX provider warnings
    'DeprecationWarning'  # Deprecated function warnings
]

# Specific warning patterns to suppress
SUPPRESSED_WARNING_PATTERNS = [
    ".*rcond.*",  # NumPy rcond parameter warnings
    ".*provider.*not in available.*",  # ONNX runtime provider warnings
    ".*xFormers.*",  # xFormers availability warnings
]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_current_model_config():
    """Get the configuration for the currently selected model."""
    if GAIT_MODEL_TYPE == "DeepGaitV2":
        return DEEPGAITV2_CONFIG
    elif GAIT_MODEL_TYPE == "GaitBase":
        return GAITBASE_CONFIG
    elif GAIT_MODEL_TYPE == "SkeletonGaitPP":
        return SKELETONGAITPP_CONFIG
    else:
        raise ValueError(f"Unknown model type: {GAIT_MODEL_TYPE}")

def get_current_model_path():
    """Get the model path for the currently selected model."""
    if GAIT_MODEL_TYPE == "DeepGaitV2":
        return DEEPGAITV2_MODEL_PATH
    elif GAIT_MODEL_TYPE == "GaitBase":
        return GAITBASE_MODEL_PATH
    elif GAIT_MODEL_TYPE == "SkeletonGaitPP":
        return SKELETONGAITPP_MODEL_PATH
    else:
        raise ValueError(f"Unknown model type: {GAIT_MODEL_TYPE}")

# ==============================================================================
# DYNAMIC CONFIGURATION ASSIGNMENT
# ==============================================================================

# Assign configurations based on selected model
GAIT_RECOGNIZER_CONFIG = get_current_model_config()
GAIT_MODEL_PATH = get_current_model_path()