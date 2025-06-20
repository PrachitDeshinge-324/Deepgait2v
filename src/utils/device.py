"""
Device selection utilities for optimal performance
"""

import torch
import config

def vprint(*args, **kwargs):
    """Verbose print - only prints if VERBOSE is enabled"""
    if config.VERBOSE:
        print(*args, **kwargs)

def get_best_device():
    """
    Select the best available device for running the model
    
    Returns:
        torch.device: Selected device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device