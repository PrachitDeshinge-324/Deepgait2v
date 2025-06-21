"""
Optimized Batch Processing for Maximum Inference Performance
Handles efficient batching, memory management, and tensor operations
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Union, Any, Iterator
import logging
from dataclasses import dataclass
from collections import defaultdict
import math
import time

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    max_batch_size: int = 8
    min_batch_size: int = 1
    sequence_length_tolerance: int = 5  # Allow sequences within this range to be batched
    memory_threshold_gb: float = 1.0  # Memory threshold for batch size adjustment
    enable_dynamic_batching: bool = True
    enable_sequence_padding: bool = True
    padding_strategy: str = "replicate"  # "replicate", "zero", "interpolate"

class OptimizedBatchProcessor:
    """
    Optimized batch processor for inference with dynamic batching,
    memory-aware processing, and efficient tensor operations
    """
    
    def __init__(self, device_manager, config: Optional[BatchConfig] = None):
        """
        Initialize batch processor
        
        Args:
            device_manager: Optimized device manager instance
            config: Batch processing configuration
        """
        self.device_manager = device_manager
        self.config = config or BatchConfig()
        self.device = device_manager.selected_device
        
        # Batch statistics
        self.batch_stats = {
            "total_batches": 0,
            "total_items": 0,
            "avg_batch_size": 0,
            "memory_usage": [],
            "processing_times": []
        }
        
        logger.info(f"Initialized batch processor for {self.device}")
    
    def create_dynamic_batches(self, 
                             silhouettes_list: List[List[np.ndarray]], 
                             parsings_list: Optional[List[List[np.ndarray]]] = None,
                             metadata_list: Optional[List[Dict[str, Any]]] = None) -> Iterator[Dict[str, Any]]:
        """
        Create dynamic batches based on sequence similarity and memory constraints
        
        Args:
            silhouettes_list: List of silhouette sequences
            parsings_list: Optional list of parsing sequences
            metadata_list: Optional metadata for each sequence
            
        Yields:
            Batch dictionaries with tensors and metadata
        """
        if not silhouettes_list:
            return
        
        # Group sequences by similarity for efficient batching
        sequence_groups = self._group_sequences_by_similarity(silhouettes_list, parsings_list, metadata_list)
        
        for group in sequence_groups:
            # Create batches within each group
            yield from self._create_batches_from_group(group)
    
    def _group_sequences_by_similarity(self, 
                                     silhouettes_list: List[List[np.ndarray]], 
                                     parsings_list: Optional[List[List[np.ndarray]]], 
                                     metadata_list: Optional[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Group sequences by length and characteristics for efficient batching"""
        
        # Create sequence items with metadata
        sequence_items = []
        for i, silhouettes in enumerate(silhouettes_list):
            item = {
                "index": i,
                "silhouettes": silhouettes,
                "parsings": parsings_list[i] if parsings_list else None,
                "metadata": metadata_list[i] if metadata_list else {},
                "sequence_length": len(silhouettes),
                "avg_area": self._calculate_avg_silhouette_area(silhouettes),
                "shape": silhouettes[0].shape if silhouettes else (64, 44)
            }
            sequence_items.append(item)
        
        # Group by sequence length with tolerance
        length_groups = defaultdict(list)
        for item in sequence_items:
            # Find the best group based on sequence length
            best_group_key = None
            best_diff = float('inf')
            
            for group_key in length_groups.keys():
                diff = abs(group_key - item["sequence_length"])
                if diff <= self.config.sequence_length_tolerance and diff < best_diff:
                    best_diff = diff
                    best_group_key = group_key
            
            if best_group_key is None:
                # Create new group
                length_groups[item["sequence_length"]].append(item)
            else:
                # Add to existing group
                length_groups[best_group_key].append(item)
        
        # Convert to list of groups
        return list(length_groups.values())
    
    def _create_batches_from_group(self, group: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Create batches from a sequence group"""
        
        # Determine optimal batch size for this group
        optimal_batch_size = self._calculate_optimal_batch_size(group)
        
        # Create batches
        for i in range(0, len(group), optimal_batch_size):
            batch_items = group[i:i + optimal_batch_size]
            batch = self._create_tensor_batch(batch_items)
            yield batch
    
    def _calculate_optimal_batch_size(self, group: List[Dict[str, Any]]) -> int:
        """Calculate optimal batch size based on memory and device constraints"""
        
        if not group:
            return 1
        
        # Start with configured max batch size
        batch_size = self.config.max_batch_size
        
        # Apply device manager optimization
        batch_size = self.device_manager.optimize_batch_processing(batch_size)
        
        # Estimate memory usage per item
        sample_item = group[0]
        estimated_memory_per_item = self._estimate_memory_usage(sample_item)
        
        # Adjust based on memory constraints
        available_memory_gb = self.device_manager.memory_monitor.get_available_memory_gb()
        max_items_by_memory = int((available_memory_gb * 0.8) / estimated_memory_per_item)  # Use 80% of available memory
        
        batch_size = min(batch_size, max_items_by_memory, len(group))
        batch_size = max(batch_size, self.config.min_batch_size)
        
        return batch_size
    
    def _create_tensor_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create tensor batch from batch items"""
        
        if not batch_items:
            raise ValueError("Cannot create batch from empty items")
        
        batch_size = len(batch_items)
        
        # Determine target sequence length (use max length in batch)
        target_seq_len = max(item["sequence_length"] for item in batch_items)
        
        # Get sample shape
        sample_shape = batch_items[0]["shape"]
        height, width = sample_shape[:2]
        
        # Create silhouette batch tensor
        sils_batch = torch.zeros((batch_size, target_seq_len, height, width), dtype=torch.float32)
        
        # Create parsing batch tensor if needed
        pars_batch = None
        if any(item["parsings"] is not None for item in batch_items):
            pars_batch = torch.zeros((batch_size, target_seq_len, height, width), dtype=torch.float32)
        
        # Sequence lengths for each item in batch
        seq_lengths = []
        
        # Fill batch tensors
        for batch_idx, item in enumerate(batch_items):
            silhouettes = item["silhouettes"]
            parsings = item["parsings"]
            seq_len = len(silhouettes)
            seq_lengths.append(seq_len)
            
            # Process silhouettes
            padded_silhouettes = self._pad_sequence(silhouettes, target_seq_len)
            for seq_idx, sil in enumerate(padded_silhouettes):
                # Normalize and resize if needed
                if sil.shape != (height, width):
                    import cv2
                    sil = cv2.resize(sil, (width, height), interpolation=cv2.INTER_LINEAR)
                
                sil_normalized = sil.astype(np.float32) / 255.0
                sils_batch[batch_idx, seq_idx] = torch.from_numpy(sil_normalized)
            
            # Process parsings if available
            if pars_batch is not None and parsings is not None:
                padded_parsings = self._pad_sequence(parsings, target_seq_len)
                for seq_idx, par in enumerate(padded_parsings):
                    if par.shape != (height, width):
                        import cv2
                        par = cv2.resize(par, (width, height), interpolation=cv2.INTER_NEAREST)
                    
                    par_normalized = par.astype(np.float32) / max(1, par.max()) if par.max() > 0 else par.astype(np.float32)
                    pars_batch[batch_idx, seq_idx] = torch.from_numpy(par_normalized)
        
        # Move tensors to device and optimize
        sils_batch = self.device_manager.optimize_tensor_operations(sils_batch)
        if pars_batch is not None:
            pars_batch = self.device_manager.optimize_tensor_operations(pars_batch)
        
        # Create sequence length tensor
        seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long, device=self.device)
        
        # Create batch dictionary
        batch = {
            "silhouettes": sils_batch,
            "parsings": pars_batch,
            "sequence_lengths": seq_lengths_tensor,
            "batch_size": batch_size,
            "target_seq_len": target_seq_len,
            "metadata": [item["metadata"] for item in batch_items],
            "indices": [item["index"] for item in batch_items]
        }
        
        return batch
    
    def _pad_sequence(self, sequence: List[np.ndarray], target_length: int) -> List[np.ndarray]:
        """Pad sequence to target length using configured strategy"""
        
        if len(sequence) >= target_length:
            return sequence[:target_length]
        
        if not self.config.enable_sequence_padding:
            return sequence
        
        padded_sequence = sequence.copy()
        padding_needed = target_length - len(sequence)
        
        if self.config.padding_strategy == "replicate":
            # Replicate last frame
            if sequence:
                last_frame = sequence[-1]
                padded_sequence.extend([last_frame.copy() for _ in range(padding_needed)])
        
        elif self.config.padding_strategy == "zero":
            # Add zero frames
            if sequence:
                zero_frame = np.zeros_like(sequence[0])
                padded_sequence.extend([zero_frame for _ in range(padding_needed)])
        
        elif self.config.padding_strategy == "interpolate":
            # Simple interpolation between last few frames
            if len(sequence) >= 2:
                frame1, frame2 = sequence[-2], sequence[-1]
                for i in range(padding_needed):
                    # Simple linear interpolation
                    alpha = (i + 1) / (padding_needed + 1)
                    interpolated = (1 - alpha) * frame1 + alpha * frame2
                    padded_sequence.append(interpolated.astype(sequence[0].dtype))
            elif sequence:
                # Fallback to replication
                last_frame = sequence[-1]
                padded_sequence.extend([last_frame.copy() for _ in range(padding_needed)])
        
        return padded_sequence
    
    def process_batch_with_model(self, 
                               batch: Dict[str, Any], 
                               model: torch.nn.Module,
                               model_type: str) -> Dict[str, torch.Tensor]:
        """
        Process a batch through the model with optimizations
        
        Args:
            batch: Batch dictionary from create_tensor_batch
            model: Model to run inference with
            model_type: Type of model ("XGait", "DeepGaitV2", etc.)
            
        Returns:
            Model outputs
        """
        start_time = time.perf_counter()
        start_memory = self.device_manager.memory_monitor.get_memory_usage()
        
        # Extract batch components
        sils_batch = batch["silhouettes"]
        pars_batch = batch["parsings"]
        seq_lengths = batch["sequence_lengths"]
        batch_size = batch["batch_size"]
        
        # Create model inputs based on model type
        with self.device_manager.optimized_inference_context(model) as optimized_model:
            # Create dummy labels for inference
            labs = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            typs = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            vies = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            seqL = [seq_lengths]
            
            # Prepare inputs based on model type
            if model_type == "XGait":
                if pars_batch is None:
                    # Generate dummy parsing if not available
                    pars_batch = torch.zeros_like(sils_batch)
                inputs = ([pars_batch, sils_batch], labs, typs, vies, seqL)
            
            elif model_type == "SkeletonGaitPP":
                # SkeletonGaitPP expects 3-channel input (pose + silhouette)
                if sils_batch.shape[2] != 3:  # Not already 3-channel
                    # Create 3-channel input: [pose_x, pose_y, silhouette]
                    pose_x = sils_batch  # Use silhouette as placeholder
                    pose_y = sils_batch  # Use silhouette as placeholder
                    multimodal_batch = torch.stack([pose_x, pose_y, sils_batch], dim=2)
                else:
                    multimodal_batch = sils_batch
                inputs = ([multimodal_batch], labs, typs, vies, seqL)
            
            else:  # DeepGaitV2, GaitBase
                inputs = ([sils_batch], labs, typs, vies, seqL)
            
            # Run inference
            with torch.no_grad():
                outputs = optimized_model(inputs)
        
        # Record statistics
        end_time = time.perf_counter()
        end_memory = self.device_manager.memory_monitor.get_memory_usage()
        
        processing_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        self.batch_stats["total_batches"] += 1
        self.batch_stats["total_items"] += batch_size
        self.batch_stats["processing_times"].append(processing_time)
        self.batch_stats["memory_usage"].append(memory_delta)
        
        # Update average batch size
        self.batch_stats["avg_batch_size"] = (
            self.batch_stats["total_items"] / self.batch_stats["total_batches"]
        )
        
        logger.debug(f"Processed batch of {batch_size} items in {processing_time:.3f}s")
        
        return outputs
    
    def _estimate_memory_usage(self, item: Dict[str, Any]) -> float:
        """Estimate memory usage for a single item in GB"""
        
        # Basic calculation based on tensor size
        seq_len = item["sequence_length"]
        height, width = item["shape"][:2]
        
        # Silhouette tensor: seq_len * height * width * 4 bytes (float32)
        sil_memory = seq_len * height * width * 4
        
        # Parsing tensor (if applicable): same size
        par_memory = sil_memory if item["parsings"] is not None else 0
        
        # Model computation memory (rough estimate): 2x input size
        computation_memory = (sil_memory + par_memory) * 2
        
        total_memory_bytes = sil_memory + par_memory + computation_memory
        total_memory_gb = total_memory_bytes / (1024**3)
        
        return total_memory_gb
    
    def _calculate_avg_silhouette_area(self, silhouettes: List[np.ndarray]) -> float:
        """Calculate average silhouette area (for grouping similar sequences)"""
        if not silhouettes:
            return 0.0
        
        total_area = 0
        for sil in silhouettes:
            area = np.sum(sil > 0) / sil.size  # Normalized area
            total_area += area
        
        return total_area / len(silhouettes)
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        stats = self.batch_stats.copy()
        
        if stats["processing_times"]:
            stats["avg_processing_time"] = np.mean(stats["processing_times"])
            stats["total_processing_time"] = np.sum(stats["processing_times"])
        
        if stats["memory_usage"]:
            stats["avg_memory_delta"] = np.mean(stats["memory_usage"])
            stats["max_memory_delta"] = np.max(stats["memory_usage"])
        
        return stats
    
    def reset_statistics(self):
        """Reset batch processing statistics"""
        self.batch_stats = {
            "total_batches": 0,
            "total_items": 0,
            "avg_batch_size": 0,
            "memory_usage": [],
            "processing_times": []
        }


class SequenceOptimizer:
    """Optimize sequences for better batch processing and inference accuracy"""
    
    @staticmethod
    def optimize_sequence_length(sequences: List[List[np.ndarray]], 
                               target_length: Optional[int] = None) -> List[List[np.ndarray]]:
        """
        Optimize sequence lengths for better batching
        
        Args:
            sequences: List of sequences to optimize
            target_length: Target sequence length (auto-determined if None)
            
        Returns:
            Optimized sequences
        """
        if not sequences:
            return sequences
        
        # Determine target length if not provided
        if target_length is None:
            lengths = [len(seq) for seq in sequences]
            target_length = int(np.median(lengths))
            target_length = max(8, min(32, target_length))  # Reasonable bounds
        
        optimized_sequences = []
        for sequence in sequences:
            optimized_seq = SequenceOptimizer._resize_sequence(sequence, target_length)
            optimized_sequences.append(optimized_seq)
        
        return optimized_sequences
    
    @staticmethod
    def _resize_sequence(sequence: List[np.ndarray], target_length: int) -> List[np.ndarray]:
        """Resize a single sequence to target length"""
        current_length = len(sequence)
        
        if current_length == target_length:
            return sequence
        
        elif current_length > target_length:
            # Downsample: select evenly spaced frames
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            return [sequence[i] for i in indices]
        
        else:
            # Upsample: interpolate or repeat frames
            result = sequence.copy()
            while len(result) < target_length:
                # Simple frame duplication
                if len(result) < current_length * 2:
                    # Duplicate frames evenly
                    for i in range(min(current_length, target_length - len(result))):
                        result.append(sequence[i].copy())
                else:
                    # Add last frame
                    result.append(sequence[-1].copy())
            
            return result[:target_length]
    
    @staticmethod
    def filter_low_quality_frames(sequence: List[np.ndarray], 
                                quality_threshold: float = 0.3) -> List[np.ndarray]:
        """
        Filter out low-quality frames from a sequence
        
        Args:
            sequence: Input sequence
            quality_threshold: Minimum quality threshold (0-1)
            
        Returns:
            Filtered sequence
        """
        if not sequence:
            return sequence
        
        filtered_sequence = []
        for frame in sequence:
            quality = SequenceOptimizer._calculate_frame_quality(frame)
            if quality >= quality_threshold:
                filtered_sequence.append(frame)
        
        # Ensure minimum sequence length
        if len(filtered_sequence) < 3 and sequence:
            # Keep best frames if too few remain
            qualities = [SequenceOptimizer._calculate_frame_quality(frame) for frame in sequence]
            best_indices = np.argsort(qualities)[-3:]  # Keep top 3
            filtered_sequence = [sequence[i] for i in sorted(best_indices)]
        
        return filtered_sequence or sequence  # Return original if all filtered out
    
    @staticmethod
    def _calculate_frame_quality(frame: np.ndarray) -> float:
        """Calculate quality score for a single frame"""
        if frame.size == 0:
            return 0.0
        
        # Basic quality metrics
        area_ratio = np.sum(frame > 0) / frame.size
        
        # Edge quality (contour smoothness)
        edges = cv2.Canny((frame * 255).astype(np.uint8), 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Aspect ratio quality (prefer human-like proportions)
        coords = np.column_stack(np.where(frame > 0))
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            height = y_max - y_min + 1
            width = x_max - x_min + 1
            aspect_ratio = height / max(1, width)
            aspect_quality = 1.0 - abs(aspect_ratio - 1.5) / 1.5  # Prefer ~1.5 aspect ratio
        else:
            aspect_quality = 0.0
        
        # Combine metrics
        quality = (area_ratio * 0.4 + edge_ratio * 0.3 + aspect_quality * 0.3)
        return min(1.0, max(0.0, quality))


# Convenience functions
def create_optimized_batch_processor(device_manager, config: Optional[BatchConfig] = None) -> OptimizedBatchProcessor:
    """Create an optimized batch processor instance"""
    return OptimizedBatchProcessor(device_manager, config)

def optimize_sequences_for_batching(sequences: List[List[np.ndarray]], 
                                  target_length: Optional[int] = None,
                                  quality_threshold: float = 0.3) -> List[List[np.ndarray]]:
    """Optimize sequences for better batching performance"""
    # Filter low-quality frames
    filtered_sequences = [
        SequenceOptimizer.filter_low_quality_frames(seq, quality_threshold) 
        for seq in sequences
    ]
    
    # Optimize sequence lengths
    optimized_sequences = SequenceOptimizer.optimize_sequence_length(filtered_sequences, target_length)
    
    return optimized_sequences
