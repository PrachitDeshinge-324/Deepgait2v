"""
Optimized Device Management for Maximum Inference Performance
Automatically selects the best available device and handles device-specific optimizations
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from contextlib import contextmanager
import gc
import warnings

# Suppress torch._dynamo errors for better compatibility
import os
os.environ.setdefault('TORCH_COMPILE_SUPPRESS_ERRORS', '1')

# Configure torch._dynamo for better compatibility
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.verbose = False
except ImportError:
    pass  # torch._dynamo not available in older versions

logger = logging.getLogger(__name__)

class OptimizedDeviceManager:
    """
    Intelligent device manager that automatically selects the best device
    and applies device-specific optimizations for maximum inference performance
    """
    
    def __init__(self, enable_profiling: bool = False):
        """
        Initialize the device manager
        
        Args:
            enable_profiling: Enable performance profiling and benchmarking
        """
        self.enable_profiling = enable_profiling
        self.device_info = self._analyze_available_devices()
        self.selected_device = self._select_optimal_device()
        self.device_optimizations = self._setup_device_optimizations()
        self.performance_cache = {}
        
        # Memory management
        self.memory_monitor = MemoryMonitor(self.selected_device)
        
        logger.info(f"Optimized device manager initialized with {self.selected_device}")
        logger.info(f"Device capabilities: {self.device_info[str(self.selected_device)]}")
    
    def _analyze_available_devices(self) -> Dict[str, Dict[str, Any]]:
        """Analyze all available devices and their capabilities"""
        devices = {}
        
        # CPU Analysis
        devices["cpu"] = {
            "available": True,
            "memory_gb": self._get_system_memory_gb(),
            "cores": torch.get_num_threads(),
            "supports_fp16": False,
            "supports_compilation": True,
            "tensor_cores": False,
            "performance_score": 1.0  # Baseline
        }
        
        # CUDA Analysis
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                devices[f"cuda:{i}"] = {
                    "available": True,
                    "name": device_props.name,
                    "memory_gb": device_props.total_memory / (1024**3),
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                    "cores": device_props.multi_processor_count,
                    "supports_fp16": device_props.major >= 6,  # Pascal and newer
                    "supports_compilation": True,
                    "tensor_cores": device_props.major >= 7,  # Volta and newer
                    "performance_score": self._estimate_cuda_performance(device_props)
                }
        
        # MPS (Apple Silicon) Analysis - with version compatibility
        try:
            # Check if MPS is available (PyTorch 1.12+)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                devices["mps"] = {
                    "available": True,
                    "name": "Apple Silicon GPU",
                    "memory_gb": self._estimate_mps_memory(),
                    "supports_fp16": True,
                    "supports_compilation": True,
                    "tensor_cores": True,  # Apple Silicon has specialized ML hardware
                    "performance_score": 3.0,  # Generally faster than CPU, slower than high-end CUDA
                    "mps_limitations": self._get_mps_limitations()
                }
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                # Alternative check for different PyTorch versions
                devices["mps"] = {
                    "available": True,
                    "name": "Apple Silicon GPU",
                    "memory_gb": self._estimate_mps_memory(),
                    "supports_fp16": True,
                    "supports_compilation": True,
                    "tensor_cores": True,
                    "performance_score": 3.0,
                    "mps_limitations": self._get_mps_limitations()
                }
        except AttributeError:
            # MPS not available in this PyTorch version
            logger.debug("MPS backend not available in this PyTorch version")
        
        return devices
    
    def _select_optimal_device(self) -> torch.device:
        """Select the optimal device based on capabilities and performance"""
        best_device = "cpu"
        best_score = 0
        
        for device_name, info in self.device_info.items():
            if not info["available"]:
                continue
            
            score = info["performance_score"]
            
            # Bonus for memory
            if info["memory_gb"] > 8:
                score += 0.5
            if info["memory_gb"] > 16:
                score += 0.5
            
            # Bonus for FP16 support
            if info.get("supports_fp16", False):
                score += 0.3
            
            # Bonus for tensor cores
            if info.get("tensor_cores", False):
                score += 0.2
            
            if score > best_score:
                best_score = score
                best_device = device_name
        
        return torch.device(best_device)
    
    def _setup_device_optimizations(self) -> Dict[str, Any]:
        """Setup device-specific optimizations"""
        optimizations = {
            "use_fp16": False,
            "use_compilation": False,
            "memory_efficient": False,
            "batch_size_multiplier": 1.0
        }
        
        device_str = str(self.selected_device)
        device_info = self.device_info.get(device_str, {})
        
        # Enable FP16 if supported and beneficial (but disable for MPS due to precision issues)
        if device_info.get("supports_fp16", False):
            if device_str.startswith("cuda") and device_info.get("tensor_cores", False):
                optimizations["use_fp16"] = True
                logger.info("Enabled FP16 precision for tensor core acceleration")
            elif device_str == "mps":
                # Disable FP16 for MPS due to precision mismatch issues
                optimizations["use_fp16"] = False
                logger.info("Disabled FP16 for MPS (precision compatibility)")
        
        # Disable compilation for now to focus on core optimizations
        # Enable compilation if supported (disabled for compatibility)
        if False:  # Temporarily disable compilation
            optimizations["use_compilation"] = True
        
        # Memory efficiency settings
        if device_info.get("memory_gb", 0) < 8:
            optimizations["memory_efficient"] = True
            optimizations["batch_size_multiplier"] = 0.5
        elif device_info.get("memory_gb", 0) > 16:
            optimizations["batch_size_multiplier"] = 2.0
        
        return optimizations
    
    @contextmanager
    def optimized_inference_context(self, model: torch.nn.Module):
        """
        Context manager for optimized inference
        Applies all relevant optimizations and handles cleanup
        """
        original_mode = model.training
        original_device = next(model.parameters()).device
        
        try:
            # Set model to eval mode
            model.eval()
            
            # Move model to optimal device
            if original_device != self.selected_device:
                model = model.to(self.selected_device)
                logger.info(f"Moved model from {original_device} to {self.selected_device}")
            
            # Apply device-specific optimizations
            optimized_model = self._apply_model_optimizations(model)
            
            # Set optimal inference settings with device compatibility
            with torch.inference_mode():
                if self.device_optimizations["use_fp16"] and self.selected_device.type in ['cuda', 'cpu']:
                    # Only use autocast for CUDA and CPU (MPS not supported in older PyTorch)
                    try:
                        with torch.autocast(device_type=self.selected_device.type, enabled=True):
                            yield optimized_model
                    except RuntimeError:
                        # Fallback without autocast
                        yield optimized_model
                else:
                    yield optimized_model
        
        finally:
            # Cleanup
            model.train(original_mode)
            if original_device != self.selected_device:
                model = model.to(original_device)
            self._cleanup_device_memory()
    
    def _apply_model_optimizations(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply model-specific optimizations"""
        optimized_model = model
        
        # Apply compilation if enabled and beneficial (with model-specific compatibility)
        if self.device_optimizations["use_compilation"]:
            try:
                # Skip torch.compile for XGait and complex models due to dynamic shape issues
                model_type = str(type(model))
                if ('XGait' in model_type or 'xgait' in model_type.lower() or 
                    hasattr(model, '__class__') and 'XGait' in str(model.__class__)):
                    logger.info("Skipping torch.compile for XGait model (compatibility)")
                    optimized_model = model
                elif hasattr(torch, 'compile'):
                    # Use torch.compile for other models
                    optimized_model = torch.compile(
                        model, 
                        mode="reduce-overhead",  # Focus on inference speed
                        dynamic=True  # Handle variable input sizes
                    )
                    logger.info("Applied torch.compile optimization")
                else:
                    optimized_model = model
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
                optimized_model = model
        
        return optimized_model
    
    def optimize_tensor_operations(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply device-specific tensor optimizations"""
        if tensor.device != self.selected_device:
            tensor = tensor.to(self.selected_device, non_blocking=True)
        
        # Apply precision optimization
        if self.device_optimizations["use_fp16"] and tensor.dtype == torch.float32:
            tensor = tensor.half()
        
        # Ensure optimal memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        return tensor
    
    def optimize_batch_processing(self, batch_size: int) -> int:
        """Optimize batch size based on device capabilities"""
        optimal_batch_size = int(batch_size * self.device_optimizations["batch_size_multiplier"])
        
        # Ensure minimum batch size
        optimal_batch_size = max(1, optimal_batch_size)
        
        # Check memory constraints
        available_memory = self.memory_monitor.get_available_memory_gb()
        if available_memory < 2:
            optimal_batch_size = min(optimal_batch_size, 1)
        elif available_memory < 4:
            optimal_batch_size = min(optimal_batch_size, 2)
        
        return optimal_batch_size
    
    def benchmark_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """Benchmark an operation and cache results"""
        if not self.enable_profiling:
            return operation_func(*args, **kwargs)
        
        cache_key = f"{operation_name}_{str(self.selected_device)}"
        
        start_time = time.perf_counter()
        start_memory = self.memory_monitor.get_memory_usage()
        
        result = operation_func(*args, **kwargs)
        
        end_time = time.perf_counter()
        end_memory = self.memory_monitor.get_memory_usage()
        
        timing = end_time - start_time
        memory_delta = end_memory - start_memory
        
        self.performance_cache[cache_key] = {
            "time": timing,
            "memory_delta": memory_delta,
            "timestamp": time.time()
        }
        
        logger.debug(f"{operation_name}: {timing:.4f}s, memory: {memory_delta:.2f}MB")
        
        return result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "selected_device": str(self.selected_device),
            "device_info": self.device_info[str(self.selected_device)],
            "optimizations": self.device_optimizations,
            "memory_stats": self.memory_monitor.get_memory_stats(),
            "performance_cache": self.performance_cache
        }
    
    def _cleanup_device_memory(self):
        """Clean up device memory with compatibility checks"""
        if self.selected_device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self.selected_device.type == "mps":
            # Check if MPS cache is available (PyTorch >= 2.1)
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        
        # Python garbage collection
        gc.collect()
    
    def _estimate_cuda_performance(self, props) -> float:
        """Estimate CUDA device performance score"""
        base_score = 5.0  # Better than CPU baseline
        
        # Memory bandwidth contribution
        memory_score = min(props.total_memory / (8 * 1024**3), 2.0)  # Up to 2 points for 8GB+
        
        # Compute capability contribution
        compute_score = (props.major - 6) * 0.5  # Bonus for newer architectures
        
        # Core count contribution
        core_score = min(props.multi_processor_count / 50, 1.0)  # Up to 1 point
        
        return base_score + memory_score + compute_score + core_score
    
    def _estimate_mps_memory(self) -> float:
        """Estimate MPS memory (simplified)"""
        # Apple Silicon typically has unified memory
        return 8.0  # Conservative estimate
    
    def _get_mps_limitations(self) -> List[str]:
        """Get known MPS limitations"""
        return [
            "Limited 3D convolution support",
            "Some operations fall back to CPU",
            "Different numerical precision than CUDA"
        ]
    
    def _get_system_memory_gb(self) -> float:
        """Get system memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 8.0  # Conservative default
    
    @property
    def device(self):
        """Compatibility property for selected_device"""
        return self.selected_device
    
    def get_device_capabilities(self):
        """Get capabilities of the selected device"""
        if self.selected_device.type in self.device_info:
            return self.device_info[self.selected_device.type]
        return {"available": True, "optimized": False}
    
    def create_tensor(self, data, dtype=None):
        """Create tensor on optimal device"""
        tensor = torch.tensor(data, dtype=dtype)
        return tensor.to(self.selected_device)


class MemoryMonitor:
    """Monitor and manage memory usage"""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device) / (1024**2)
        else:
            # For CPU/MPS, return process memory usage
            try:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / (1024**2)
            except ImportError:
                return 0.0
    
    def get_available_memory_gb(self) -> float:
        """Get available memory in GB"""
        if self.device.type == "cuda":
            total = torch.cuda.get_device_properties(self.device).total_memory
            allocated = torch.cuda.memory_allocated(self.device)
            return (total - allocated) / (1024**3)
        else:
            try:
                import psutil
                return psutil.virtual_memory().available / (1024**3)
            except ImportError:
                return 4.0  # Conservative default
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics"""
        stats = {
            "current_usage_mb": self.get_memory_usage(),
            "available_gb": self.get_available_memory_gb()
        }
        
        if self.device.type == "cuda":
            stats.update({
                "max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / (1024**2),
                "cached_mb": torch.cuda.memory_cached(self.device) / (1024**2)
            })
        
        return stats


# Global device manager instance
_device_manager = None

def get_device_manager(enable_profiling: bool = False) -> OptimizedDeviceManager:
    """Get global device manager instance"""
    global _device_manager
    if _device_manager is None:
        _device_manager = OptimizedDeviceManager(enable_profiling)
    return _device_manager

def get_optimal_device() -> torch.device:
    """Get the optimal device for inference"""
    return get_device_manager().selected_device

def optimize_for_inference(model: torch.nn.Module):
    """Apply inference optimizations to a model"""
    return get_device_manager().optimized_inference_context(model)
