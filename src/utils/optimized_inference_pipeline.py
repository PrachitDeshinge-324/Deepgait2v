"""
Optimized Model Inference Pipeline
Integrates device management, batch processing, and memory optimization
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any, Union
import logging
import time
from pathlib import Path
import warnings

# Apply warning suppression for cleaner output
from .warning_suppressor import WarningManager
_warning_manager = WarningManager()
_warning_manager.suppress_common_warnings()

from .optimized_device_manager import get_device_manager, OptimizedDeviceManager
from .optimized_batch_processor import (
    OptimizedBatchProcessor, 
    BatchConfig, 
    optimize_sequences_for_batching
)

logger = logging.getLogger(__name__)

class OptimizedInferencePipeline:
    """
    Complete optimized inference pipeline that integrates:
    - Intelligent device selection and management
    - Efficient batch processing
    - Memory optimization
    - Preprocessing optimization
    - Postprocessing and feature normalization
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 model_type: str,
                 enable_profiling: bool = False,
                 batch_config: Optional[BatchConfig] = None):
        """
        Initialize optimized inference pipeline
        
        Args:
            model: PyTorch model for inference
            model_type: Type of model ("XGait", "DeepGaitV2", "SkeletonGaitPP", "GaitBase")
            enable_profiling: Enable performance profiling
            batch_config: Batch processing configuration
        """
        self.model = model
        self.model_type = model_type
        self.enable_profiling = enable_profiling
        
        # Initialize optimized components
        self.device_manager = get_device_manager(enable_profiling)
        self.batch_processor = OptimizedBatchProcessor(self.device_manager, batch_config)
        
        # Model-specific configurations
        self.model_config = self._get_model_config(model_type)
        
        # Performance tracking
        self.inference_stats = {
            "total_inferences": 0,
            "total_time": 0.0,
            "avg_time_per_inference": 0.0,
            "memory_stats": []
        }
        
        logger.info(f"Optimized inference pipeline initialized for {model_type}")
        logger.info(f"Device: {self.device_manager.selected_device}")
    
    def _get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get model-specific configuration"""
        configs = {
            "XGait": {
                "requires_parsing": True,
                "input_channels": 1,
                "expected_shape": (64, 44),
                "supports_batch": True,
                "preprocessing_fn": self._preprocess_xgait
            },
            "DeepGaitV2": {
                "requires_parsing": False,
                "input_channels": 1,
                "expected_shape": (64, 44),
                "supports_batch": True,
                "preprocessing_fn": self._preprocess_deepgaitv2
            },
            "SkeletonGaitPP": {
                "requires_parsing": False,
                "input_channels": 3,  # pose_x + pose_y + silhouette
                "expected_shape": (64, 44),
                "supports_batch": True,
                "preprocessing_fn": self._preprocess_skeletongaitpp
            },
            "GaitBase": {
                "requires_parsing": False,
                "input_channels": 1,
                "expected_shape": (64, 44),
                "supports_batch": True,
                "preprocessing_fn": self._preprocess_gaitbase
            }
        }
        
        return configs.get(model_type, configs["DeepGaitV2"])
    
    def infer_single(self, 
                    silhouettes: List[np.ndarray],
                    parsings: Optional[List[np.ndarray]] = None,
                    quality_threshold: float = 0.3) -> Optional[np.ndarray]:
        """
        Perform inference on a single sequence with full optimization
        
        Args:
            silhouettes: List of silhouette frames
            parsings: Optional parsing frames (for XGait)
            quality_threshold: Quality threshold for frame filtering
            
        Returns:
            Feature embeddings or None if failed
        """
        if not silhouettes:
            logger.warning("Empty silhouette sequence provided")
            return None
        
        start_time = time.perf_counter()
        
        try:
            # Optimize sequence
            optimized_sequences = optimize_sequences_for_batching(
                [silhouettes], 
                quality_threshold=quality_threshold
            )
            
            if not optimized_sequences or not optimized_sequences[0]:
                logger.warning("Sequence optimization resulted in empty sequence")
                return None
            
            optimized_silhouettes = optimized_sequences[0]
            
            # Create batch for single sequence
            batch_data = [{
                "silhouettes": optimized_silhouettes,
                "parsings": parsings,
                "metadata": {"original_length": len(silhouettes)}
            }]
            
            # Process through batch pipeline
            results = self.infer_batch(batch_data)
            
            if results and len(results) > 0:
                return results[0]
            else:
                return None
        
        except Exception as e:
            logger.error(f"Single inference failed: {e}")
            return None
        
        finally:
            # Update statistics
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            self._update_stats(inference_time, 1)
    
    def infer_batch(self, 
                   batch_data: List[Dict[str, Any]],
                   optimize_sequences: bool = True) -> List[Optional[np.ndarray]]:
        """
        Perform batch inference with full optimization
        
        Args:
            batch_data: List of dictionaries with 'silhouettes', 'parsings', 'metadata'
            optimize_sequences: Whether to optimize sequences before processing
            
        Returns:
            List of feature embeddings (one per input sequence)
        """
        if not batch_data:
            return []
        
        start_time = time.perf_counter()
        total_sequences = len(batch_data)
        
        try:
            # Extract sequences
            silhouettes_list = [item["silhouettes"] for item in batch_data]
            parsings_list = [item.get("parsings") for item in batch_data]
            metadata_list = [item.get("metadata", {}) for item in batch_data]
            
            # Optimize sequences if enabled
            if optimize_sequences:
                silhouettes_list = optimize_sequences_for_batching(silhouettes_list)
            
            # Process through optimized batch pipeline
            all_results = []
            
            # Create dynamic batches and process
            batch_generator = self.batch_processor.create_dynamic_batches(
                silhouettes_list, parsings_list, metadata_list
            )
            
            for batch in batch_generator:
                # Process batch through model
                batch_outputs = self.batch_processor.process_batch_with_model(
                    batch, self.model, self.model_type
                )
                
                # Extract embeddings
                embeddings = batch_outputs['inference_feat']['embeddings']
                embeddings_np = embeddings.cpu().numpy()
                
                # Apply postprocessing
                processed_embeddings = self._postprocess_embeddings(embeddings_np)
                
                # Add to results
                for i in range(batch["batch_size"]):
                    all_results.append(processed_embeddings[i])
            
            return all_results
        
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            # Return None for each sequence
            return [None] * total_sequences
        
        finally:
            # Update statistics
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            self._update_stats(inference_time, total_sequences)
    
    def _preprocess_xgait(self, silhouettes: List[np.ndarray], 
                         parsings: Optional[List[np.ndarray]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess inputs for XGait model"""
        # XGait requires both silhouettes and parsing maps
        if parsings is None:
            # Generate dummy parsing maps
            parsings = [np.zeros_like(sil) for sil in silhouettes]
        
        return self._create_tensor_pair(silhouettes, parsings)
    
    def _preprocess_deepgaitv2(self, silhouettes: List[np.ndarray], 
                              parsings: Optional[List[np.ndarray]] = None) -> torch.Tensor:
        """Preprocess inputs for DeepGaitV2 model"""
        return self._create_silhouette_tensor(silhouettes)
    
    def _preprocess_skeletongaitpp(self, silhouettes: List[np.ndarray], 
                                  parsings: Optional[List[np.ndarray]] = None) -> torch.Tensor:
        """Preprocess inputs for SkeletonGaitPP model"""
        # Create 3-channel input: [pose_x, pose_y, silhouette]
        # For now, use silhouette as placeholder for pose channels
        seq_len = len(silhouettes)
        height, width = self.model_config["expected_shape"]
        
        # Create 3-channel tensor
        tensor_3ch = torch.zeros((1, seq_len, 3, height, width), dtype=torch.float32)
        
        for i, sil in enumerate(silhouettes):
            # Resize and normalize silhouette
            if sil.shape != (height, width):
                sil = cv2.resize(sil, (width, height), interpolation=cv2.INTER_LINEAR)
            
            sil_normalized = torch.from_numpy(sil.astype(np.float32) / 255.0)
            
            # Use silhouette for all 3 channels (placeholder for pose)
            tensor_3ch[0, i, 0] = sil_normalized  # pose_x
            tensor_3ch[0, i, 1] = sil_normalized  # pose_y
            tensor_3ch[0, i, 2] = sil_normalized  # silhouette
        
        return tensor_3ch
    
    def _preprocess_gaitbase(self, silhouettes: List[np.ndarray], 
                            parsings: Optional[List[np.ndarray]] = None) -> torch.Tensor:
        """Preprocess inputs for GaitBase model"""
        return self._create_silhouette_tensor(silhouettes)
    
    def _create_silhouette_tensor(self, silhouettes: List[np.ndarray]) -> torch.Tensor:
        """Create silhouette tensor"""
        seq_len = len(silhouettes)
        height, width = self.model_config["expected_shape"]
        
        tensor = torch.zeros((1, seq_len, height, width), dtype=torch.float32)
        
        for i, sil in enumerate(silhouettes):
            if sil.shape != (height, width):
                sil = cv2.resize(sil, (width, height), interpolation=cv2.INTER_LINEAR)
            
            sil_normalized = sil.astype(np.float32) / 255.0
            tensor[0, i] = torch.from_numpy(sil_normalized)
        
        return tensor
    
    def _create_tensor_pair(self, silhouettes: List[np.ndarray], 
                          parsings: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create tensor pair for models that need both silhouettes and parsing"""
        seq_len = len(silhouettes)
        height, width = self.model_config["expected_shape"]
        
        sils_tensor = torch.zeros((1, seq_len, height, width), dtype=torch.float32)
        pars_tensor = torch.zeros((1, seq_len, height, width), dtype=torch.float32)
        
        for i, (sil, par) in enumerate(zip(silhouettes, parsings)):
            # Process silhouette
            if sil.shape != (height, width):
                sil = cv2.resize(sil, (width, height), interpolation=cv2.INTER_LINEAR)
            sil_normalized = sil.astype(np.float32) / 255.0
            sils_tensor[0, i] = torch.from_numpy(sil_normalized)
            
            # Process parsing
            if par.shape != (height, width):
                par = cv2.resize(par, (width, height), interpolation=cv2.INTER_NEAREST)
            par_normalized = par.astype(np.float32) / max(1, par.max()) if par.max() > 0 else par.astype(np.float32)
            pars_tensor[0, i] = torch.from_numpy(par_normalized)
        
        return pars_tensor, sils_tensor
    
    def _postprocess_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply postprocessing to embeddings"""
        
        # Handle different embedding shapes
        if embeddings.ndim == 3:  # [batch, features, parts]
            # Flatten spatial dimensions if present
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        elif embeddings.ndim > 3:
            # Flatten all dimensions except batch
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        
        # Apply L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        embeddings = embeddings / norms
        
        # Apply numerical stability checks
        embeddings = np.clip(embeddings, -10, 10)  # Prevent extreme values
        
        # Handle NaN/inf values
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return embeddings
    
    def _update_stats(self, inference_time: float, num_sequences: int):
        """Update inference statistics"""
        self.inference_stats["total_inferences"] += num_sequences
        self.inference_stats["total_time"] += inference_time
        self.inference_stats["avg_time_per_inference"] = (
            self.inference_stats["total_time"] / self.inference_stats["total_inferences"]
        )
        
        # Record memory stats
        memory_stats = self.device_manager.memory_monitor.get_memory_stats()
        self.inference_stats["memory_stats"].append(memory_stats)
    
    def benchmark_inference(self, 
                          test_sequences: List[List[np.ndarray]],
                          num_runs: int = 5) -> Dict[str, Any]:
        """
        Benchmark inference performance
        
        Args:
            test_sequences: Test sequences for benchmarking
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        if not self.enable_profiling:
            logger.warning("Profiling not enabled. Enable with enable_profiling=True")
            return {}
        
        logger.info(f"Running inference benchmark with {len(test_sequences)} sequences, {num_runs} runs")
        
        # Prepare test data
        batch_data = [{"silhouettes": seq} for seq in test_sequences]
        
        # Warmup runs
        for _ in range(2):
            self.infer_batch(batch_data)
        
        # Benchmark runs
        times = []
        memory_usage = []
        
        for run in range(num_runs):
            start_time = time.perf_counter()
            start_memory = self.device_manager.memory_monitor.get_memory_usage()
            
            results = self.infer_batch(batch_data)
            
            end_time = time.perf_counter()
            end_memory = self.device_manager.memory_monitor.get_memory_usage()
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
        
        # Calculate statistics
        benchmark_results = {
            "model_type": self.model_type,
            "device": str(self.device_manager.selected_device),
            "num_sequences": len(test_sequences),
            "num_runs": num_runs,
            "times": {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
                "per_sequence": np.mean(times) / len(test_sequences)
            },
            "memory": {
                "mean_delta_mb": np.mean(memory_usage),
                "max_delta_mb": np.max(memory_usage)
            },
            "device_performance": self.device_manager.get_performance_report(),
            "batch_stats": self.batch_processor.get_batch_statistics()
        }
        
        logger.info(f"Benchmark complete. Avg time: {benchmark_results['times']['mean']:.3f}s")
        logger.info(f"Time per sequence: {benchmark_results['times']['per_sequence']:.3f}s")
        
        return benchmark_results
    
    def get_inference_statistics(self) -> Dict[str, Any]:
        """Get comprehensive inference statistics"""
        return {
            "inference_stats": self.inference_stats,
            "device_stats": self.device_manager.get_performance_report(),
            "batch_stats": self.batch_processor.get_batch_statistics()
        }
    
    def optimize_for_deployment(self) -> Dict[str, Any]:
        """
        Apply additional optimizations for deployment
        
        Returns:
            Optimization report
        """
        logger.info("Applying deployment optimizations...")
        
        optimizations_applied = []
        
        # Model optimizations
        with self.device_manager.optimized_inference_context(self.model) as optimized_model:
            # Check if model can be optimized further
            try:
                # Try to trace the model for additional optimizations
                dummy_input = self._create_dummy_input()
                traced_model = torch.jit.trace(optimized_model, dummy_input)
                self.model = traced_model
                optimizations_applied.append("TorchScript tracing")
            except Exception as e:
                logger.warning(f"TorchScript tracing failed: {e}")
        
        # Memory optimizations
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
            optimizations_applied.append("CuDNN benchmark mode")
        
        # Set optimal thread count for CPU operations
        if self.device_manager.selected_device.type == "cpu":
            optimal_threads = min(torch.get_num_threads(), 8)  # Limit to avoid oversubscription
            torch.set_num_threads(optimal_threads)
            optimizations_applied.append(f"CPU threads: {optimal_threads}")
        
        optimization_report = {
            "optimizations_applied": optimizations_applied,
            "device": str(self.device_manager.selected_device),
            "model_type": self.model_type,
            "timestamp": time.time()
        }
        
        logger.info(f"Deployment optimizations complete: {optimizations_applied}")
        
        return optimization_report
    
    def _create_dummy_input(self) -> Tuple:
        """Create dummy input for model tracing"""
        batch_size = 1
        seq_len = 10
        height, width = self.model_config["expected_shape"]
        
        # Create dummy tensors
        sils_tensor = torch.randn(batch_size, seq_len, height, width, device=self.device_manager.selected_device)
        
        if self.model_type == "XGait":
            pars_tensor = torch.randn(batch_size, seq_len, height, width, device=self.device_manager.selected_device)
            inputs = ([pars_tensor, sils_tensor],)
        elif self.model_type == "SkeletonGaitPP":
            multimodal_tensor = torch.randn(batch_size, seq_len, 3, height, width, device=self.device_manager.selected_device)
            inputs = ([multimodal_tensor],)
        else:
            inputs = ([sils_tensor],)
        
        # Add dummy labels
        dummy_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device_manager.selected_device)
        dummy_seqL = [torch.tensor([seq_len], dtype=torch.long, device=self.device_manager.selected_device)]
        
        return inputs + (dummy_labels, dummy_labels, dummy_labels, dummy_seqL)


# Convenience functions
def create_optimized_pipeline(model: torch.nn.Module, 
                            model_type: str,
                            enable_profiling: bool = False,
                            batch_config: Optional[BatchConfig] = None) -> OptimizedInferencePipeline:
    """Create an optimized inference pipeline"""
    return OptimizedInferencePipeline(model, model_type, enable_profiling, batch_config)

def benchmark_model_performance(model: torch.nn.Module,
                              model_type: str,
                              test_sequences: List[List[np.ndarray]],
                              num_runs: int = 5) -> Dict[str, Any]:
    """Benchmark model performance with optimized pipeline"""
    pipeline = create_optimized_pipeline(model, model_type, enable_profiling=True)
    return pipeline.benchmark_inference(test_sequences, num_runs)
