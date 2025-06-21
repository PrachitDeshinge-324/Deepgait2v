"""
Improved Multimodal Fusion Strategy for XGait + Face Recognition
Addresses issues with feature combination that may cause low accuracy
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ImprovedMultimodalFusion:
    """Enhanced multimodal fusion strategy for better accuracy"""
    
    def __init__(self, config):
        self.config = config
        
        # Adaptive fusion parameters
        self.base_face_weight = getattr(config, 'FACE_WEIGHT', 0.4)  # Reduced from 0.5 for gait-focused systems
        self.base_gait_weight = getattr(config, 'GAIT_WEIGHT', 0.6)  # Increased from 0.5 for better gait utilization
        
        # Quality-based adaptation
        self.quality_adaptation = True
        self.confidence_based_weighting = True
        
        # Normalization strategy
        self.normalize_features = True
        self.feature_scaling = True
        
        # Cross-modal consistency
        self.consistency_checking = True
        self.temporal_smoothing = True
        
        # Thresholds
        self.min_gait_quality = 0.3
        self.min_face_quality = 0.4
        self.consistency_threshold = 0.3
        
    def fuse_features_adaptive(self, gait_embedding: Optional[np.ndarray], 
                             face_embedding: Optional[np.ndarray],
                             gait_quality: float = 0.5,
                             face_quality: float = 0.5,
                             sequence_consistency: float = 0.5) -> Dict[str, Any]:
        """
        Adaptive fusion of gait and face features with quality awareness
        
        Args:
            gait_embedding: Gait feature embedding
            face_embedding: Face feature embedding  
            gait_quality: Quality score of gait sequence
            face_quality: Quality score of face detection
            sequence_consistency: Temporal consistency score
            
        Returns:
            Dictionary with fusion results and metadata
        """
        fusion_result = {
            'fused_embedding': None,
            'fusion_weights': {'gait': 0.0, 'face': 0.0},
            'confidence': 0.0,
            'modalities_used': [],
            'quality_scores': {'gait': gait_quality, 'face': face_quality},
            'fusion_strategy': 'none'
        }
        
        # Validate inputs
        valid_gait = gait_embedding is not None and gait_quality >= self.min_gait_quality
        valid_face = face_embedding is not None and face_quality >= self.min_face_quality
        
        if not valid_gait and not valid_face:
            logger.warning("No valid modalities for fusion")
            return fusion_result
        
        # Normalize embeddings if enabled
        if self.normalize_features:
            if valid_gait:
                gait_embedding = self._normalize_embedding(gait_embedding)
            if valid_face:
                face_embedding = self._normalize_embedding(face_embedding)
        
        # Choose fusion strategy based on available modalities
        if valid_gait and valid_face:
            return self._fuse_both_modalities(
                gait_embedding, face_embedding, gait_quality, face_quality,
                sequence_consistency, fusion_result
            )
        elif valid_gait:
            return self._use_gait_only(gait_embedding, gait_quality, fusion_result)
        else:  # valid_face only
            return self._use_face_only(face_embedding, face_quality, fusion_result)
    
    def _fuse_both_modalities(self, gait_embedding: np.ndarray, face_embedding: np.ndarray,
                            gait_quality: float, face_quality: float, 
                            sequence_consistency: float, fusion_result: Dict) -> Dict:
        """Fuse both gait and face modalities with adaptive weighting"""
        
        # Calculate adaptive weights based on quality and confidence
        gait_weight, face_weight = self._calculate_adaptive_weights(
            gait_quality, face_quality, sequence_consistency
        )
        
        # Handle dimension mismatch
        gait_dim = gait_embedding.shape[-1] if gait_embedding.ndim > 1 else len(gait_embedding)
        face_dim = face_embedding.shape[-1] if face_embedding.ndim > 1 else len(face_embedding)
        
        if gait_dim != face_dim:
            # Project to common dimension
            target_dim = min(gait_dim, face_dim)
            gait_embedding = self._project_embedding(gait_embedding, target_dim)
            face_embedding = self._project_embedding(face_embedding, target_dim)
        
        # Scale features to similar ranges
        if self.feature_scaling:
            gait_embedding = self._scale_features(gait_embedding)
            face_embedding = self._scale_features(face_embedding)
        
        # Perform weighted fusion
        fused_embedding = (gait_weight * gait_embedding + face_weight * face_embedding)
        
        # Apply consistency penalty if embeddings are too different
        if self.consistency_checking:
            consistency_penalty = self._calculate_consistency_penalty(
                gait_embedding, face_embedding
            )
            confidence = (gait_weight * gait_quality + face_weight * face_quality) * consistency_penalty
        else:
            confidence = gait_weight * gait_quality + face_weight * face_quality
        
        fusion_result.update({
            'fused_embedding': fused_embedding,
            'fusion_weights': {'gait': gait_weight, 'face': face_weight},
            'confidence': confidence,
            'modalities_used': ['gait', 'face'],
            'fusion_strategy': 'adaptive_weighted'
        })
        
        return fusion_result
    
    def _use_gait_only(self, gait_embedding: np.ndarray, gait_quality: float, 
                      fusion_result: Dict) -> Dict:
        """Use gait modality only"""
        fusion_result.update({
            'fused_embedding': gait_embedding,
            'fusion_weights': {'gait': 1.0, 'face': 0.0},
            'confidence': gait_quality,
            'modalities_used': ['gait'],
            'fusion_strategy': 'gait_only'
        })
        return fusion_result
    
    def _use_face_only(self, face_embedding: np.ndarray, face_quality: float,
                      fusion_result: Dict) -> Dict:
        """Use face modality only"""
        fusion_result.update({
            'fused_embedding': face_embedding,
            'fusion_weights': {'gait': 0.0, 'face': 1.0},
            'confidence': face_quality * 0.8,  # Slightly penalize face-only identification
            'modalities_used': ['face'],
            'fusion_strategy': 'face_only'
        })
        return fusion_result
    
    def _calculate_adaptive_weights(self, gait_quality: float, face_quality: float,
                                  sequence_consistency: float) -> Tuple[float, float]:
        """Calculate adaptive weights based on quality and consistency"""
        
        # Base weights
        gait_weight = self.base_gait_weight
        face_weight = self.base_face_weight
        
        if self.quality_adaptation:
            # Adjust weights based on relative quality
            total_quality = gait_quality + face_quality
            if total_quality > 0:
                quality_factor = 0.6  # How much quality affects the weights
                
                # Quality-based adjustment
                quality_gait_weight = (gait_quality / total_quality) * quality_factor
                quality_face_weight = (face_quality / total_quality) * quality_factor
                
                # Combine with base weights
                gait_weight = (1 - quality_factor) * gait_weight + quality_gait_weight
                face_weight = (1 - quality_factor) * face_weight + quality_face_weight
        
        # Apply consistency boost to gait (more reliable for person re-identification)
        if sequence_consistency > 0.7:
            consistency_boost = 0.2
            gait_weight = min(1.0, gait_weight + consistency_boost)
            face_weight = max(0.0, face_weight - consistency_boost)
        
        # Normalize weights
        total_weight = gait_weight + face_weight
        if total_weight > 0:
            gait_weight /= total_weight
            face_weight /= total_weight
        
        return gait_weight, face_weight
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit length"""
        if embedding.ndim == 1:
            norm = np.linalg.norm(embedding)
            return embedding / max(norm, 1e-8)
        else:
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            return embedding / np.maximum(norms, 1e-8)
    
    def _project_embedding(self, embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """Project embedding to target dimension"""
        if embedding.ndim == 1:
            current_dim = len(embedding)
        else:
            current_dim = embedding.shape[-1]
        
        if current_dim == target_dim:
            return embedding
        elif current_dim > target_dim:
            # Truncate to target dimension
            if embedding.ndim == 1:
                return embedding[:target_dim]
            else:
                return embedding[..., :target_dim]
        else:
            # Pad with zeros to reach target dimension
            if embedding.ndim == 1:
                padded = np.zeros(target_dim, dtype=embedding.dtype)
                padded[:current_dim] = embedding
                return padded
            else:
                pad_width = [(0, 0)] * (embedding.ndim - 1) + [(0, target_dim - current_dim)]
                return np.pad(embedding, pad_width, mode='constant', constant_values=0)
    
    def _scale_features(self, embedding: np.ndarray) -> np.ndarray:
        """Scale features to have similar magnitude"""
        # Simple min-max scaling to [0, 1] range
        if embedding.ndim == 1:
            min_val, max_val = embedding.min(), embedding.max()
            if max_val > min_val:
                return (embedding - min_val) / (max_val - min_val)
            return embedding
        else:
            min_vals = embedding.min(axis=1, keepdims=True)
            max_vals = embedding.max(axis=1, keepdims=True)
            ranges = max_vals - min_vals
            ranges = np.maximum(ranges, 1e-8)
            return (embedding - min_vals) / ranges
    
    def _calculate_consistency_penalty(self, gait_embedding: np.ndarray, 
                                     face_embedding: np.ndarray) -> float:
        """Calculate consistency penalty based on embedding similarity"""
        # Calculate cosine similarity between embeddings
        if gait_embedding.ndim > 1:
            gait_flat = gait_embedding.flatten()
        else:
            gait_flat = gait_embedding
        
        if face_embedding.ndim > 1:
            face_flat = face_embedding.flatten()
        else:
            face_flat = face_embedding
        
        # Ensure same length
        min_len = min(len(gait_flat), len(face_flat))
        gait_flat = gait_flat[:min_len]
        face_flat = face_flat[:min_len]
        
        # Calculate cosine similarity
        dot_product = np.dot(gait_flat, face_flat)
        norm_gait = np.linalg.norm(gait_flat)
        norm_face = np.linalg.norm(face_flat)
        
        if norm_gait > 0 and norm_face > 0:
            similarity = dot_product / (norm_gait * norm_face)
        else:
            similarity = 0.0
        
        # Convert similarity to consistency factor
        # High similarity = low penalty (factor close to 1.0)
        # Low similarity = high penalty (factor closer to 0.5)
        consistency_factor = 0.5 + 0.5 * max(0, similarity)
        return consistency_factor
    
    def get_fusion_recommendations(self) -> List[str]:
        """Get recommendations for improving multimodal fusion"""
        recommendations = [
            "🔧 Balance fusion weights: Try gait=0.6, face=0.4 for gait-focused systems",
            "🔧 Enable quality-based adaptive weighting",
            "🔧 Use feature normalization to ensure fair comparison",
            "🔧 Apply consistency checking to detect conflicting modalities",
            "🔧 Consider temporal smoothing for video sequences",
            "🔧 Project embeddings to common dimension space",
            "🔧 Scale features to similar ranges before fusion",
            "🔧 Penalize face-only identification in gait-focused scenarios"
        ]
        return recommendations
    
    def analyze_fusion_performance(self, fusion_results: List[Dict]) -> Dict[str, Any]:
        """Analyze fusion performance across multiple samples"""
        if not fusion_results:
            return {}
        
        analysis = {
            'total_samples': len(fusion_results),
            'strategies_used': {},
            'avg_confidence': 0.0,
            'modality_usage': {'gait_only': 0, 'face_only': 0, 'both': 0},
            'avg_weights': {'gait': 0.0, 'face': 0.0}
        }
        
        confidences = []
        gait_weights = []
        face_weights = []
        
        for result in fusion_results:
            strategy = result.get('fusion_strategy', 'unknown')
            analysis['strategies_used'][strategy] = analysis['strategies_used'].get(strategy, 0) + 1
            
            modalities = result.get('modalities_used', [])
            if len(modalities) == 1:
                if 'gait' in modalities:
                    analysis['modality_usage']['gait_only'] += 1
                elif 'face' in modalities:
                    analysis['modality_usage']['face_only'] += 1
            elif len(modalities) == 2:
                analysis['modality_usage']['both'] += 1
            
            confidences.append(result.get('confidence', 0.0))
            weights = result.get('fusion_weights', {'gait': 0.0, 'face': 0.0})
            gait_weights.append(weights.get('gait', 0.0))
            face_weights.append(weights.get('face', 0.0))
        
        analysis['avg_confidence'] = np.mean(confidences) if confidences else 0.0
        analysis['avg_weights']['gait'] = np.mean(gait_weights) if gait_weights else 0.0
        analysis['avg_weights']['face'] = np.mean(face_weights) if face_weights else 0.0
        
        return analysis
