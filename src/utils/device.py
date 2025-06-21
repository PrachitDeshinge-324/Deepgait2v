"""
Device selection utilities for optimal performance
Enhanced with optimized device management integration
"""

import torch
import config
from .optimized_device_manager import get_device_manager

def vprint(*args, **kwargs):
    """Verbose print - only prints if VERBOSE is enabled"""
    if config.VERBOSE:
        print(*args, **kwargs)

def get_best_device():
    """
    Select the best available device for running the model
    Uses the optimized device manager for intelligent selection
    
    Returns:
        torch.device: Selected device (CUDA, MPS, or CPU)
    """
    try:
        # Use optimized device manager for best selection
        device_manager = get_device_manager()
        optimal_device = device_manager.selected_device
        
        # Print device info
        device_info = device_manager.device_info.get(str(optimal_device), {})
        device_name = device_info.get('name', str(optimal_device))
        memory_gb = device_info.get('memory_gb', 'unknown')
        
        vprint(f"Optimized device selected: {optimal_device}")
        vprint(f"Device name: {device_name}")
        vprint(f"Available memory: {memory_gb}GB")
        
        if device_info.get('optimizations'):
            vprint(f"Applied optimizations: {device_info['optimizations']}")
        
        return optimal_device
        
    except Exception as e:
        # Fallback to simple device selection
        vprint(f"Optimized device selection failed: {e}")
        vprint("Falling back to simple device selection")
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            vprint(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            vprint("Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            vprint("Using CPU")
        
        return device

def get_device_info():
    """
    Get comprehensive device information
    
    Returns:
        Dict with device capabilities and optimizations
    """
    try:
        device_manager = get_device_manager()
        return device_manager.get_performance_report()
    except Exception as e:
        vprint(f"Could not get device info: {e}")
        return {"error": str(e)}

def optimize_tensor_for_device(tensor, device=None):
    """
    Optimize tensor for the target device
    
    Args:
        tensor: Input tensor
        device: Target device (uses optimal device if None)
        
    Returns:
        Optimized tensor
    """
    try:
        device_manager = get_device_manager()
        if device is None:
            device = device_manager.selected_device
        
        # Move to device if needed
        if tensor.device != device:
            tensor = tensor.to(device, non_blocking=True)
        
        # Apply device-specific optimizations
        return device_manager.optimize_tensor_operations(tensor)
        
    except Exception as e:
        vprint(f"Tensor optimization failed: {e}")
        # Fallback to simple device transfer
        if device is None:
            device = get_best_device()
        return tensor.to(device)

def get_optimal_batch_size(base_batch_size=4):
    """
    Get optimal batch size for the current device
    
    Args:
        base_batch_size: Base batch size to optimize from
        
    Returns:
        Optimized batch size
    """
    try:
        device_manager = get_device_manager()
        return device_manager.optimize_batch_processing(base_batch_size)
    except Exception as e:
        vprint(f"Batch size optimization failed: {e}")
        return base_batch_size