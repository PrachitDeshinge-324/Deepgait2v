"""
Processing components for silhouettes, quality assessment and features
"""

from .silhouette_extractor import SilhouetteExtractor
from .quality_assessor import GaitSequenceQualityAssessor
from .face_embedding_extractor import FaceEmbeddingExtractor
from .pose_generator import PoseHeatmapGenerator, SimplifiedPoseGenerator

__all__ = [
    'SilhouetteExtractor',
    'GaitSequenceQualityAssessor', 
    'FaceEmbeddingExtractor',
    'PoseHeatmapGenerator',
    'SimplifiedPoseGenerator'
]
