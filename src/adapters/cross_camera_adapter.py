"""
Cross-Camera Domain Adaptation Module for Gait and Face Recognition

This module implements domain adaptation techniques to improve cross-camera performance:
1. Feature normalization and standardization
2. Camera-invariant preprocessing
3. Adaptive embedding adjustment
4. Domain-aware similarity metrics
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import config

class CrossCameraDomainAdapter:
    """
    Handles domain adaptation for cross-camera scenarios
    """
    
    def __init__(self):
        """Initialize domain adaptation components"""
        
        # Feature normalization components
        self.gait_scaler = RobustScaler()  # Robust to outliers
        self.face_scaler = StandardScaler()  # Standard normalization for faces
        
        # PCA for dimension reduction and noise removal
        # Disabled PCA to avoid dimensionality mismatches during identification
        self.gait_pca = None  # Disable PCA for gait embeddings
        self.face_pca = None  # Disable PCA for faces to preserve 512-dim embeddings
        
        # Camera-specific statistics
        self.camera_stats = {}  # camera_id -> statistics
        self.domain_statistics_collected = False
        
        # Embedding banks for domain adaptation
        self.reference_embeddings = {'gait': [], 'face': []}
        self.camera_embeddings = {}  # camera_id -> {'gait': [], 'face': []}
        
        # Configuration
        self.enable_pca_whitening = getattr(config, 'ENABLE_PCA_WHITENING', True)
        self.enable_cross_camera_normalization = getattr(config, 'ENABLE_CROSS_CAMERA_NORM', True)
        self.min_samples_for_adaptation = getattr(config, 'MIN_ADAPTATION_SAMPLES', 10)
        
    def collect_camera_statistics(self, embeddings: Dict[str, np.ndarray], 
                                camera_id: str = "default"):
        """
        Collect embeddings from a specific camera for domain adaptation
        
        Args:
            embeddings: Dictionary with 'gait' and/or 'face' embeddings
            camera_id: Camera identifier
        """
        if camera_id not in self.camera_embeddings:
            self.camera_embeddings[camera_id] = {'gait': [], 'face': []}
        
        for modality, embedding in embeddings.items():
            if embedding is not None and modality in ['gait', 'face']:
                self.camera_embeddings[camera_id][modality].append(embedding.copy())
                
        # Update reference embeddings pool
        if 'gait' in embeddings and embeddings['gait'] is not None:
            self.reference_embeddings['gait'].append(embeddings['gait'].copy())
        if 'face' in embeddings and embeddings['face'] is not None:
            self.reference_embeddings['face'].append(embeddings['face'].copy())
    
    def fit_domain_adaptation(self):
        """
        Fit domain adaptation models using collected embeddings
        """
        print("🔄 Fitting cross-camera domain adaptation...")
        
        # Fit normalization for gait embeddings
        if len(self.reference_embeddings['gait']) >= self.min_samples_for_adaptation:
            gait_stack = np.vstack(self.reference_embeddings['gait'])
            
            # Handle different gait shapes (flatten if needed)
            if len(gait_stack.shape) > 2:
                original_shape = gait_stack.shape[1:]
                gait_stack = gait_stack.reshape(gait_stack.shape[0], -1)
                self.gait_original_shape = original_shape
            else:
                self.gait_original_shape = None
                
            self.gait_scaler.fit(gait_stack)
            
            # Skip PCA for gait embeddings to avoid dimension mismatches
            print(f"✅ Gait domain adaptation fitted with {len(self.reference_embeddings['gait'])} samples (PCA disabled to avoid dimension mismatches)")
        
        # Fit normalization for face embeddings
        if len(self.reference_embeddings['face']) >= self.min_samples_for_adaptation:
            face_stack = np.vstack(self.reference_embeddings['face'])
            self.face_scaler.fit(face_stack)
            
            # Skip PCA for face embeddings to preserve 512 dimensions
            print(f"✅ Face domain adaptation fitted with {len(self.reference_embeddings['face'])} samples (PCA disabled to preserve dimensions)")
        
        # Compute camera-specific adaptation statistics
        self._compute_camera_adaptation_stats()
        self.domain_statistics_collected = True
        
    def _compute_camera_adaptation_stats(self):
        """Compute camera-specific adaptation statistics"""
        for camera_id, embeddings in self.camera_embeddings.items():
            self.camera_stats[camera_id] = {}
            
            # Gait statistics
            if len(embeddings['gait']) > 0:
                gait_stack = np.vstack(embeddings['gait'])
                if len(gait_stack.shape) > 2:
                    gait_stack = gait_stack.reshape(gait_stack.shape[0], -1)
                
                self.camera_stats[camera_id]['gait'] = {
                    'mean': np.mean(gait_stack, axis=0),
                    'std': np.std(gait_stack, axis=0) + 1e-8,
                    'median': np.median(gait_stack, axis=0)
                }
                
            # Face statistics  
            if len(embeddings['face']) > 0:
                face_stack = np.vstack(embeddings['face'])
                self.camera_stats[camera_id]['face'] = {
                    'mean': np.mean(face_stack, axis=0),
                    'std': np.std(face_stack, axis=0) + 1e-8,
                    'median': np.median(face_stack, axis=0)
                }
    
    def adapt_embedding(self, embedding: np.ndarray, modality: str, 
                       camera_id: str = "default") -> np.ndarray:
        """
        Apply domain adaptation to an embedding
        
        Args:
            embedding: Input embedding
            modality: 'gait' or 'face'
            camera_id: Source camera identifier
            
        Returns:
            Domain-adapted embedding
        """
        if embedding is None:
            return None
            
        adapted = embedding.copy()
        
        # Reshape gait embeddings if needed
        original_shape = None
        if modality == 'gait' and len(adapted.shape) > 1:
            original_shape = adapted.shape
            adapted = adapted.reshape(-1)
        
        # Apply robust normalization
        if modality == 'gait' and hasattr(self.gait_scaler, 'scale_'):
            adapted = adapted.reshape(1, -1)
            adapted = self.gait_scaler.transform(adapted)[0]
            
            # Skip PCA for gait embeddings to avoid dimension mismatches
                
        elif modality == 'face' and hasattr(self.face_scaler, 'scale_'):
            adapted = adapted.reshape(1, -1)
            adapted = self.face_scaler.transform(adapted)[0]
            
            # Skip PCA for face embeddings to preserve 512 dimensions
        
        # Camera-specific adaptation
        if (self.domain_statistics_collected and 
            camera_id in self.camera_stats and 
            modality in self.camera_stats[camera_id]):
            
            camera_mean = self.camera_stats[camera_id][modality]['mean']
            camera_std = self.camera_stats[camera_id][modality]['std']
            
            # Remove camera-specific bias
            if len(adapted) == len(camera_mean):
                adapted = (adapted - camera_mean) / camera_std
                adapted = np.tanh(adapted)  # Bounded normalization
        
        # Restore original shape for gait
        if original_shape is not None and modality == 'gait':
            try:
                adapted = adapted.reshape(original_shape)
            except:
                pass  # Keep flattened if reshape fails
        
        return adapted
    
    def compute_camera_invariant_similarity(self, emb1: np.ndarray, emb2: np.ndarray,
                                          modality: str) -> float:
        """
        Compute camera-invariant similarity between embeddings
        
        Args:
            emb1, emb2: Embeddings to compare
            modality: 'gait' or 'face'
            
        Returns:
            Similarity score
        """
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Flatten if needed
        if len(emb1.shape) > 1:
            emb1 = emb1.reshape(-1)
        if len(emb2.shape) > 1:
            emb2 = emb2.reshape(-1)
        
        # Ensure same length
        min_len = min(len(emb1), len(emb2))
        emb1 = emb1[:min_len]
        emb2 = emb2[:min_len]
        
        # L2 normalize
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        # Cosine similarity (more robust for cross-camera)
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Apply modality-specific adjustments
        if modality == 'gait':
            # Gait can be more variable across cameras
            similarity = np.tanh(similarity * 1.2)  # Slightly boost good matches
        elif modality == 'face':
            # Face features are more stable across cameras
            similarity = similarity ** 0.8  # Slightly reduce penalty for moderate matches
        
        return float(similarity)

class CameraInvariantPreprocessor:
    """
    Preprocessing pipeline for camera-invariant features
    """
    
    def __init__(self):
        """Initialize preprocessing components"""
        self.histogram_equalizer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
    def preprocess_silhouette(self, silhouette: np.ndarray, 
                            camera_params: Optional[Dict] = None) -> np.ndarray:
        """
        Apply camera-invariant preprocessing to silhouette
        
        Args:
            silhouette: Input silhouette
            camera_params: Camera-specific parameters
            
        Returns:
            Preprocessed silhouette
        """
        if silhouette is None:
            return None
            
        processed = silhouette.copy()
        
        # Histogram equalization for lighting invariance
        if len(processed.shape) == 2:
            processed = self.histogram_equalizer.apply(processed)
        
        # Morphological operations for noise reduction
        kernel = np.ones((3,3), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        # Gaussian blur to reduce camera-specific noise
        processed = cv2.GaussianBlur(processed, (3,3), 0.5)
        
        # Normalize intensity
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
        
        return processed
    
    def preprocess_face_crop(self, face_crop: np.ndarray,
                           camera_params: Optional[Dict] = None) -> np.ndarray:
        """
        Apply camera-invariant preprocessing to face crop
        
        Args:
            face_crop: Input face crop
            camera_params: Camera-specific parameters
            
        Returns:
            Preprocessed face crop
        """
        if face_crop is None:
            return None
            
        processed = face_crop.copy()
        
        # Color space conversion for illumination invariance
        if len(processed.shape) == 3:
            # Convert to LAB for better lighting invariance
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            l = self.histogram_equalizer.apply(l)
            
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Gaussian blur to reduce camera-specific artifacts
        processed = cv2.GaussianBlur(processed, (3,3), 0.7)
        
        return processed

# Global instances for cross-camera adaptation
domain_adapter = CrossCameraDomainAdapter()
camera_preprocessor = CameraInvariantPreprocessor()

def apply_cross_camera_adaptation(gait_embedding: Optional[np.ndarray],
                                face_embedding: Optional[np.ndarray],
                                camera_id: str = "default") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Apply cross-camera domain adaptation to embeddings
    
    Args:
        gait_embedding: Gait embedding
        face_embedding: Face embedding  
        camera_id: Camera identifier
        
    Returns:
        Tuple of adapted (gait_embedding, face_embedding)
    """
    adapted_gait = None
    adapted_face = None
    
    if gait_embedding is not None:
        adapted_gait = domain_adapter.adapt_embedding(gait_embedding, 'gait', camera_id)
    
    if face_embedding is not None:
        adapted_face = domain_adapter.adapt_embedding(face_embedding, 'face', camera_id)
    
    return adapted_gait, adapted_face

def compute_cross_camera_similarity(emb1: np.ndarray, emb2: np.ndarray, 
                                  modality: str) -> float:
    """
    Compute similarity using camera-invariant metrics
    
    Args:
        emb1, emb2: Embeddings to compare
        modality: 'gait' or 'face'
        
    Returns:
        Similarity score
    """
    return domain_adapter.compute_camera_invariant_similarity(emb1, emb2, modality)
