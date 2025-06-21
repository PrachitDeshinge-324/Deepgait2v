"""
Optimized Human Parsing Module with GPU acceleration and batch processing
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class OptimizedHumanParser:
    """
    Optimized human parser with batch processing and device acceleration
    """
    
    def __init__(self, base_parser, device=None, enable_fp16=True):
        """
        Initialize optimized parser wrapper
        
        Args:
            base_parser: Original HumanParsingModel instance
            device: Target device (auto-detected if None)
            enable_fp16: Enable mixed precision inference
        """
        self.base_parser = base_parser
        self.enable_fp16 = enable_fp16 and torch.cuda.is_available()
        
        # Auto-detect optimal device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Move model to optimal device
        if hasattr(self.base_parser, 'model') and self.base_parser.model is not None:
            try:
                self.base_parser.model = self.base_parser.model.to(self.device)
                
                # Enable mixed precision if supported
                if self.enable_fp16 and self.device.type == 'cuda':
                    self.base_parser.model = self.base_parser.model.half()
                    
                logger.info(f"Human parsing model moved to {self.device}")
                
            except Exception as e:
                logger.warning(f"Failed to move parsing model to {self.device}: {e}")
                self.device = torch.device('cpu')
        
        # Cache for preprocessing transforms
        self._preprocess_cache = {}
        
    def parse_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process multiple images in a batch for efficiency
        
        Args:
            images: List of RGB images to parse
            
        Returns:
            List of parsing maps
        """
        if not images:
            return []
        
        # Fast path for single image
        if len(images) == 1:
            return [self.parse_single_optimized(images[0])]
        
        try:
            return self._batch_inference(images)
        except Exception as e:
            logger.warning(f"Batch inference failed, falling back to individual: {e}")
            # Fallback to individual processing
            return [self.parse_single_optimized(img) for img in images]
    
    def parse_single_optimized(self, image: np.ndarray) -> np.ndarray:
        """
        Optimized single image parsing with device acceleration
        
        Args:
            image: RGB image to parse
            
        Returns:
            Parsing map
        """
        try:
            # Use optimized preprocessing
            input_tensor = self._preprocess_optimized(image)
            
            # Inference with device optimization
            with torch.no_grad():
                if self.enable_fp16 and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        output = self.base_parser.model(input_tensor)
                else:
                    output = self.base_parser.model(input_tensor)
            
            # Optimized postprocessing
            parsing_map = self._postprocess_optimized(output, image.shape[:2])
            
            return parsing_map
            
        except Exception as e:
            logger.warning(f"Optimized parsing failed, using fallback: {e}")
            # Fallback to original method
            return self.base_parser.parse_human(image)
    
    def _batch_inference(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Perform batch inference for multiple images
        """
        # Preprocess all images
        input_tensors = []
        original_sizes = []
        
        for img in images:
            input_tensor = self._preprocess_optimized(img)
            input_tensors.append(input_tensor.squeeze(0))  # Remove batch dim
            original_sizes.append(img.shape[:2])
        
        # Stack into batch
        batch_tensor = torch.stack(input_tensors, dim=0)
        
        # Batch inference
        with torch.no_grad():
            if self.enable_fp16 and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    batch_output = self.base_parser.model(batch_tensor)
            else:
                batch_output = self.base_parser.model(batch_tensor)
        
        # Process each output
        results = []
        for i in range(len(images)):
            if hasattr(batch_output, 'out'):
                output = batch_output.out[i:i+1]  # Keep batch dim
            else:
                output = batch_output[i:i+1]
            
            parsing_map = self._postprocess_optimized(output, original_sizes[i])
            results.append(parsing_map)
        
        return results
    
    def _preprocess_optimized(self, image: np.ndarray) -> torch.Tensor:
        """
        Optimized preprocessing with caching
        """
        h, w = image.shape[:2]
        cache_key = (h, w)
        
        # Use cached transforms if available
        if cache_key not in self._preprocess_cache:
            # Determine target size (standard for human parsing)
            if hasattr(self.base_parser, 'input_size'):
                target_size = self.base_parser.input_size
            else:
                target_size = (512, 512)  # Default
            
            self._preprocess_cache[cache_key] = target_size
        
        target_size = self._preprocess_cache[cache_key]
        
        # Resize image
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor format [1, C, H, W]
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        # Apply mixed precision if enabled
        if self.enable_fp16 and self.device.type == 'cuda':
            tensor = tensor.half()
        
        return tensor
    
    def _postprocess_optimized(self, output: torch.Tensor, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Optimized postprocessing with GPU operations when possible
        """
        # Handle different output formats
        if hasattr(output, 'out'):
            pred = output.out
        else:
            pred = output
        
        # GPU-accelerated softmax and argmax
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        
        # Resize on GPU if possible, then move to CPU
        if self.device.type in ['cuda', 'mps']:
            # Resize using GPU interpolation
            pred_float = pred.float().unsqueeze(1)  # Add channel dim for interpolation
            pred_resized = F.interpolate(
                pred_float, 
                size=target_size, 
                mode='nearest'
            )
            pred_final = pred_resized.squeeze(1).cpu().numpy()[0]
        else:
            # CPU fallback
            pred_np = pred.cpu().numpy()[0]
            pred_final = cv2.resize(
                pred_np.astype(np.uint8), 
                (target_size[1], target_size[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        
        return pred_final.astype(np.uint8)


def optimize_human_parser(parser) -> OptimizedHumanParser:
    """
    Create optimized wrapper for existing human parser
    
    Args:
        parser: Original HumanParsingModel instance
        
    Returns:
        OptimizedHumanParser wrapper
    """
    return OptimizedHumanParser(parser)
