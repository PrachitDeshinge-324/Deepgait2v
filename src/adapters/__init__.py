"""
Cross-camera domain adaptation components
"""

from .cross_camera_adapter import (
    CrossCameraDomainAdapter,
    CameraInvariantPreprocessor,
    apply_cross_camera_adaptation,
    compute_cross_camera_similarity,
    domain_adapter
)

__all__ = [
    'CrossCameraDomainAdapter',
    'CameraInvariantPreprocessor',
    'apply_cross_camera_adaptation',
    'compute_cross_camera_similarity',
    'domain_adapter'
]
