"""
Gait recognition models and loaders
"""

from .gait_recognizer import GaitRecognizer
from .skeletongait_loader import SkeletonGaitPP

__all__ = [
    'GaitRecognizer',
    'SkeletonGaitPP'
]
