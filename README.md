# Multimodal Gait and Face Recognition System

## 🎯 Overview

A production-ready multimodal gait and face recognition system designed for cross-camera scenarios. The system combines gait biometrics and face recognition using advanced domain adaptation techniques to maintain high accuracy across different camera setups.

## ✨ Key Features

- **Multimodal Recognition**: Combines gait and face biometrics for robust identification
- **Cross-Camera Support**: Domain adaptation for consistent performance across cameras
- **Real-time Processing**: Optimized for live video stream analysis
- **Quality Assessment**: Intelligent filtering of low-quality detections
- **Modular Architecture**: Clean, organized codebase for easy maintenance and extension

## 🏗️ Architecture

### Project Structure:
```
main.py                          # Entry point
config.py                        # Configuration settings
├── src/
│   ├── core/                    # Core detection and tracking
│   │   ├── detector.py          # Object detection
│   │   ├── tracker.py           # Multi-object tracking
│   │   └── visualizer.py        # Visualization utilities
│   ├── models/                  # Recognition models
│   │   ├── gait_recognizer.py   # Gait recognition
│   │   └── skeletongait_loader.py # Skeleton-based gait
│   ├── processing/              # Feature extraction
│   │   ├── silhouette_extractor.py # Gait silhouettes
│   │   ├── face_embedding_extractor.py # Face features
│   │   ├── pose_generator.py    # Pose estimation
│   │   └── quality_assessor.py  # Quality metrics
│   ├── identification/          # Identity matching
│   │   └── multimodal_identifier.py # Multimodal fusion
│   ├── adapters/               # Domain adaptation
│   │   └── cross_camera_adapter.py # Cross-camera adaptation
│   ├── app/                    # Application logic
│   │   ├── gait_recognition_app.py # Main application
│   │   ├── identification_manager.py # Identity management
│   │   ├── database_handler.py  # Database operations
│   │   └── ...                 # Other app components
│   └── utils/                  # Utilities
│       ├── database.py         # Database utilities
│       ├── device.py           # Device management
│       └── ...
```

### Core Components:
- **Detection & Tracking**: Real-time person detection and tracking
- **Feature Extraction**: Gait silhouettes, poses, and face embeddings  
- **Multimodal Fusion**: Intelligent combination of biometric modalities
- **Domain Adaptation**: Cross-camera performance optimization
- **Database Management**: Efficient storage and retrieval of biometric templates

## � Quick Start

### Prerequisites:
- Python 3.8+
- CUDA-capable GPU (recommended)
- Required model weights in `weights/` directory

### Installation:
```bash
# Clone the repository
git clone <repository-url>
cd only_gait

# Install Python dependencies
pip install -r requirements.txt

# Note: For GPU support with FAISS, replace faiss-cpu with faiss-gpu in requirements.txt
# pip install faiss-gpu

# Ensure model weights are in the weights/ directory
# Download required model weights:
# - YOLO models: yolo11m.pt, yolo11l-seg.pt
# - Gait models: DeepGaitV2_30_DA-50000.pt, GaitBase_DA-180000.pt, etc.
# - Face models: face_detection_yunet.onnx, face_recognition_sface.onnx
```

### Basic Usage:
```bash
# Run the main application
python main.py

# For testing with specific configurations
python main.py --config custom_config.py
```

### Configuration:
Key settings in `config.py`:
```python
# Model Settings
GAIT_MODEL_PATH = "weights/DeepGaitV2_30_DA-50000.pt"
FACE_MODEL_PATH = "weights/face_recognition_sface.onnx"

# Processing Parameters
MIN_SEQUENCE_LENGTH = 16
QUALITY_THRESHOLD = 0.5

# Cross-Camera Adaptation
ENABLE_CROSS_CAMERA_ADAPTATION = True
CURRENT_CAMERA_ID = "camera_1"

# Fusion Weights
FACE_WEIGHT = 0.7
GAIT_WEIGHT = 0.3
```

## 📁 Directory Structure

```
only_gait/
├── main.py                    # Main entry point
├── config.py                  # Configuration settings
├── __init__.py               # Package initialization
├── src/                      # Source code
│   ├── core/                 # Core detection and tracking
│   ├── models/               # Recognition models
│   ├── processing/           # Feature extraction
│   ├── identification/       # Identity matching
│   ├── adapters/            # Domain adaptation
│   ├── app/                 # Application logic
│   └── utils/               # Utility functions
├── weights/                  # Model weights
├── data_DeepGaitV2_v1_face/ # Database (if present)
├── output_v1_face/          # Processing results
└── OpenGait/                # OpenGait framework
```

## �️ Development

### Key Modules:
- **`src.identification.multimodal_identifier`**: Core multimodal fusion logic
- **`src.adapters.cross_camera_adapter`**: Domain adaptation for cross-camera scenarios
- **`src.app.gait_recognition_app`**: Main application framework
- **`src.core.detector`**: Person detection using YOLO
- **`src.models.gait_recognizer`**: Gait feature extraction

### Adding New Features:
1. Create new modules in appropriate `src/` subdirectories
2. Update `__init__.py` files for clean imports
3. Add configuration options in `config.py`
4. Test with existing database or create new test cases

## � Production Readiness

The codebase has been optimized for production use:
- ✅ **Modular Architecture**: Clean separation of concerns
- ✅ **Organized Structure**: Logical file organization under `src/`
- ✅ **Clean Imports**: Proper package structure with `__init__.py` files
- ✅ **No Duplicates**: Removed redundant and unused modules
- ✅ **Documentation**: Comprehensive README and inline comments
- ✅ **Configurable**: Centralized configuration management
