"""
Organized Gait Recognition System

A modular multimodal gait and face recognition system with cross-camera adaptation.
"""

__version__ = "2.0.0"
__author__ = "Gait Recognition Team"

# Make core components easily importable
from .src.app.gait_recognition_app import GaitRecognitionApp
from .src.identification.multimodal_identifier import MultiModalIdentifier
from .src.adapters.cross_camera_adapter import CrossCameraDomainAdapter

__all__ = [
    'GaitRecognitionApp',
    'MultiModalIdentifier', 
    'CrossCameraDomainAdapter'
]
