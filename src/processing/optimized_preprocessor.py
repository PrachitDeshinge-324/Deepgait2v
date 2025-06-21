"""
Optimized Preprocessing Pipeline for Gait Recognition
Handles efficient silhouette and parsing preprocessing with device optimization
"""

import cv2
import numpy as np
import torch
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import time

from ..utils.optimized_device_manager import get_device_manager
from ..utils.optimized_batch_processor import optimize_sequences_for_batching

logger = logging.getLogger(__name__)

class OptimizedPreprocessor:
    """
    Optimized preprocessing pipeline for silhouettes and parsing maps
    with intelligent batching and device-aware operations
    """
    
    def __init__(self, target_size: Tuple[int, int] = (64, 44)):
        """
        Initialize optimized preprocessor
        
        Args:
            target_size: Target size for silhouettes (height, width)
        """
        self.target_size = target_size
        self.device_manager = get_device_manager()
        
        # Preprocessing statistics
        self.stats = {
            "total_sequences": 0,
            "total_frames": 0,
            "avg_preprocessing_time": 0.0,
            "preprocessing_times": []
        }
        
        logger.info(f"Optimized preprocessor initialized with target size {target_size}")
    
    def preprocess_sequence(self, 
                          silhouettes: List[np.ndarray],
                          parsings: Optional[List[np.ndarray]] = None,
                          quality_threshold: float = 0.3,
                          normalize: bool = True) -> Dict[str, Any]:
        """
        Preprocess a single sequence with quality filtering and optimization
        
        Args:
            silhouettes: List of silhouette frames
            parsings: Optional parsing frames
            quality_threshold: Quality threshold for frame filtering
            normalize: Whether to normalize pixel values
            
        Returns:
            Dictionary with preprocessed data
        """
        start_time = time.perf_counter()
        
        if not silhouettes:
            return {"silhouettes": [], "parsings": [], "sequence_length": 0}
        
        # Apply quality filtering and optimization
        filtered_silhouettes = self._filter_quality_frames(silhouettes, quality_threshold)
        filtered_parsings = None
        
        if parsings is not None:
            # Filter parsings to match silhouettes
            if len(parsings) == len(silhouettes):
                # Get indices of kept silhouettes
                kept_indices = self._get_kept_frame_indices(silhouettes, filtered_silhouettes)
                filtered_parsings = [parsings[i] for i in kept_indices]
            else:
                logger.warning("Silhouette and parsing counts don't match")
                filtered_parsings = parsings[:len(filtered_silhouettes)]
        
        # Resize and normalize frames
        processed_silhouettes = self._process_frames(filtered_silhouettes, normalize)
        processed_parsings = None
        
        if filtered_parsings is not None:
            processed_parsings = self._process_parsing_frames(filtered_parsings, normalize)
        
        # Update statistics
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        self._update_stats(processing_time, 1, len(processed_silhouettes))
        
        return {
            "silhouettes": processed_silhouettes,
            "parsings": processed_parsings,
            "sequence_length": len(processed_silhouettes),
            "processing_time": processing_time,
            "original_length": len(silhouettes)
        }
    
    def preprocess_batch(self, 
                        batch_sequences: List[List[np.ndarray]],
                        batch_parsings: Optional[List[List[np.ndarray]]] = None,
                        quality_threshold: float = 0.3,
                        optimize_lengths: bool = True) -> List[Dict[str, Any]]:
        """
        Preprocess multiple sequences with batch optimization
        
        Args:
            batch_sequences: List of silhouette sequences
            batch_parsings: Optional list of parsing sequences
            quality_threshold: Quality threshold for frame filtering
            optimize_lengths: Whether to optimize sequence lengths for batching
            
        Returns:
            List of preprocessed sequence dictionaries
        """
        start_time = time.perf_counter()
        
        if not batch_sequences:
            return []
        
        # Optimize sequences for batching if requested
        if optimize_lengths:
            optimized_sequences = optimize_sequences_for_batching(
                batch_sequences, 
                quality_threshold=quality_threshold
            )
        else:
            optimized_sequences = batch_sequences
        
        # Process each sequence
        results = []
        for i, silhouettes in enumerate(optimized_sequences):
            parsings = batch_parsings[i] if batch_parsings and i < len(batch_parsings) else None
            
            result = self.preprocess_sequence(
                silhouettes=silhouettes,
                parsings=parsings,
                quality_threshold=quality_threshold,
                normalize=True
            )
            
            results.append(result)
        
        # Update batch statistics
        end_time = time.perf_counter()
        batch_time = end_time - start_time
        total_frames = sum(len(seq) for seq in optimized_sequences)
        self._update_stats(batch_time, len(optimized_sequences), total_frames)
        
        logger.debug(f"Preprocessed batch of {len(optimized_sequences)} sequences in {batch_time:.3f}s")
        
        return results
    
    def _filter_quality_frames(self, 
                              silhouettes: List[np.ndarray], 
                              quality_threshold: float) -> List[np.ndarray]:
        """Filter frames based on quality metrics"""
        if quality_threshold <= 0:
            return silhouettes
        
        quality_scores = []
        for sil in silhouettes:
            score = self._calculate_frame_quality(sil)
            quality_scores.append(score)
        
        # Filter frames above threshold
        filtered_frames = []
        for sil, score in zip(silhouettes, quality_scores):
            if score >= quality_threshold:
                filtered_frames.append(sil)
        
        # Ensure minimum sequence length
        if len(filtered_frames) < 3 and silhouettes:
            # Keep best frames if too few remain
            best_indices = np.argsort(quality_scores)[-max(3, len(silhouettes)//2):]
            filtered_frames = [silhouettes[i] for i in sorted(best_indices)]
        
        return filtered_frames or silhouettes  # Return original if all filtered
    
    def _calculate_frame_quality(self, frame: np.ndarray) -> float:
        """Calculate quality score for a frame"""
        if frame.size == 0:
            return 0.0
        
        # Silhouette area ratio
        area_ratio = np.sum(frame > 0) / frame.size
        
        # Aspect ratio quality (prefer human-like proportions)
        coords = np.column_stack(np.where(frame > 0))
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            height = y_max - y_min + 1
            width = x_max - x_min + 1
            aspect_ratio = height / max(1, width)
            aspect_quality = 1.0 - abs(aspect_ratio - 1.5) / 1.5
        else:
            aspect_quality = 0.0
        
        # Edge continuity (using simple gradient)
        if frame.dtype != np.uint8:
            frame_uint8 = (frame * 255).astype(np.uint8)
        else:
            frame_uint8 = frame
        
        # Simple edge detection
        edges = cv2.Canny(frame_uint8, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Combine metrics
        quality = (area_ratio * 0.4 + aspect_quality * 0.4 + edge_ratio * 0.2)
        return min(1.0, max(0.0, quality))
    
    def _get_kept_frame_indices(self, 
                               original_frames: List[np.ndarray], 
                               filtered_frames: List[np.ndarray]) -> List[int]:
        """Get indices of frames that were kept after filtering"""
        # Simple approach: find first occurrence of each filtered frame
        kept_indices = []
        used_indices = set()
        
        for filtered_frame in filtered_frames:
            for i, original_frame in enumerate(original_frames):
                if i not in used_indices and np.array_equal(filtered_frame, original_frame):
                    kept_indices.append(i)
                    used_indices.add(i)
                    break
        
        return kept_indices
    
    def _process_frames(self, 
                       frames: List[np.ndarray], 
                       normalize: bool = True) -> List[np.ndarray]:
        """Process silhouette frames"""
        processed_frames = []
        
        for frame in frames:
            # Resize to target size
            if frame.shape[:2] != self.target_size:
                resized = cv2.resize(frame, (self.target_size[1], self.target_size[0]), 
                                   interpolation=cv2.INTER_LINEAR)
            else:
                resized = frame.copy()
            
            # Normalize if requested
            if normalize:
                if resized.dtype != np.float32:
                    resized = resized.astype(np.float32)
                if resized.max() > 1.0:
                    resized = resized / 255.0
            
            # Ensure binary silhouette
            if normalize:
                resized = (resized > 0.5).astype(np.float32)
            else:
                resized = (resized > 127).astype(np.uint8) * 255
            
            processed_frames.append(resized)
        
        return processed_frames
    
    def _process_parsing_frames(self, 
                               frames: List[np.ndarray], 
                               normalize: bool = True) -> List[np.ndarray]:
        """Process parsing frames"""
        processed_frames = []
        
        for frame in frames:
            # Resize to target size using nearest neighbor for labels
            if frame.shape[:2] != self.target_size:
                resized = cv2.resize(frame, (self.target_size[1], self.target_size[0]), 
                                   interpolation=cv2.INTER_NEAREST)
            else:
                resized = frame.copy()
            
            # Normalize if requested
            if normalize:
                if resized.dtype != np.float32:
                    resized = resized.astype(np.float32)
                # Normalize parsing labels
                max_label = resized.max()
                if max_label > 0:
                    resized = resized / max_label
            
            processed_frames.append(resized)
        
        return processed_frames
    
    def create_model_tensors(self, 
                           preprocessed_data: Dict[str, Any],
                           model_type: str) -> Tuple[torch.Tensor, ...]:
        """
        Create model-ready tensors from preprocessed data
        
        Args:
            preprocessed_data: Output from preprocess_sequence
            model_type: Type of model ("XGait", "DeepGaitV2", etc.)
            
        Returns:
            Tuple of tensors ready for model input
        """
        silhouettes = preprocessed_data["silhouettes"]
        parsings = preprocessed_data["parsings"]
        seq_len = preprocessed_data["sequence_length"]
        
        if not silhouettes:
            raise ValueError("No silhouettes in preprocessed data")
        
        height, width = self.target_size
        
        if model_type == "XGait":
            # XGait requires both silhouettes and parsing
            sils_tensor = torch.zeros((1, seq_len, height, width), dtype=torch.float32)
            pars_tensor = torch.zeros((1, seq_len, height, width), dtype=torch.float32)
            
            for i, sil in enumerate(silhouettes):
                sils_tensor[0, i] = torch.from_numpy(sil)
            
            if parsings is not None:
                for i, par in enumerate(parsings):
                    pars_tensor[0, i] = torch.from_numpy(par)
            
            # Optimize tensors for device
            sils_tensor = self.device_manager.optimize_tensor_operations(sils_tensor)
            pars_tensor = self.device_manager.optimize_tensor_operations(pars_tensor)
            
            return pars_tensor, sils_tensor
        
        elif model_type == "SkeletonGaitPP":
            # SkeletonGaitPP requires 3-channel input
            tensor_3ch = torch.zeros((1, seq_len, 3, height, width), dtype=torch.float32)
            
            for i, sil in enumerate(silhouettes):
                sil_tensor = torch.from_numpy(sil)
                # Use silhouette for all 3 channels (placeholder for pose)
                tensor_3ch[0, i, 0] = sil_tensor  # pose_x
                tensor_3ch[0, i, 1] = sil_tensor  # pose_y
                tensor_3ch[0, i, 2] = sil_tensor  # silhouette
            
            tensor_3ch = self.device_manager.optimize_tensor_operations(tensor_3ch)
            return (tensor_3ch,)
        
        else:  # DeepGaitV2, GaitBase
            # Single silhouette input
            sils_tensor = torch.zeros((1, seq_len, height, width), dtype=torch.float32)
            
            for i, sil in enumerate(silhouettes):
                sils_tensor[0, i] = torch.from_numpy(sil)
            
            sils_tensor = self.device_manager.optimize_tensor_operations(sils_tensor)
            return (sils_tensor,)
    
    def _update_stats(self, processing_time: float, num_sequences: int, num_frames: int):
        """Update preprocessing statistics"""
        self.stats["total_sequences"] += num_sequences
        self.stats["total_frames"] += num_frames
        self.stats["preprocessing_times"].append(processing_time)
        
        # Update average
        if self.stats["preprocessing_times"]:
            self.stats["avg_preprocessing_time"] = np.mean(self.stats["preprocessing_times"])
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        stats = self.stats.copy()
        
        if stats["preprocessing_times"]:
            stats["total_preprocessing_time"] = sum(stats["preprocessing_times"])
            stats["avg_time_per_sequence"] = stats["avg_preprocessing_time"]
            stats["avg_time_per_frame"] = stats["total_preprocessing_time"] / max(1, stats["total_frames"])
        
        return stats
    
    def reset_stats(self):
        """Reset preprocessing statistics"""
        self.stats = {
            "total_sequences": 0,
            "total_frames": 0,
            "avg_preprocessing_time": 0.0,
            "preprocessing_times": []
        }


# Convenience functions
def create_optimized_preprocessor(target_size: Tuple[int, int] = (64, 44)) -> OptimizedPreprocessor:
    """Create an optimized preprocessor instance"""
    return OptimizedPreprocessor(target_size)

def preprocess_for_inference(silhouettes: List[np.ndarray],
                            parsings: Optional[List[np.ndarray]] = None,
                            model_type: str = "DeepGaitV2",
                            quality_threshold: float = 0.3) -> Tuple[torch.Tensor, ...]:
    """
    Convenience function for preprocessing sequences for inference
    
    Args:
        silhouettes: List of silhouette frames
        parsings: Optional parsing frames
        model_type: Type of model
        quality_threshold: Quality threshold for filtering
        
    Returns:
        Model-ready tensors
    """
    preprocessor = create_optimized_preprocessor()
    
    # Preprocess sequence
    preprocessed = preprocessor.preprocess_sequence(
        silhouettes=silhouettes,
        parsings=parsings,
        quality_threshold=quality_threshold
    )
    
    # Create model tensors
    return preprocessor.create_model_tensors(preprocessed, model_type)
