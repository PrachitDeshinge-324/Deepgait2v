"""
Improved XGait Inference Pipeline
Addresses common issues causing low accuracy in XGait model inference
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
import logging
import sys
import os
from pathlib import Path

# Import scipy for distance transform, with fallback
try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from processing.human_parsing import HumanParsingModel

logger = logging.getLogger(__name__)

class ImprovedXGaitInference:
    """Improved inference pipeline for XGait model with better accuracy"""
    
    def __init__(self, gait_recognizer, config):
        self.gait_recognizer = gait_recognizer
        self.config = config
        self.device = gait_recognizer.device
        
        # Enhanced preprocessing parameters
        self.target_size = (64, 44)  # XGait standard size
        self.min_sequence_length = 8
        self.max_sequence_length = 32
        self.quality_threshold = 0.3
        
        # Domain adaptation parameters
        self.apply_domain_adaptation = True
        self.normalize_features = True
        
        # Parsing enhancement
        self.enhance_parsing = True
        self.min_body_parts = 4
        
        # Initialize human parsing model
        self.human_parser = None
        self.use_pretrained_parsing = True
        self._initialize_human_parser()
        
    def _initialize_human_parser(self):
        """Initialize the human parsing model with optimizations"""
        try:
            # Use SCHP model by default as it's optimized for gait analysis
            self.human_parser = HumanParsingModel(
                model_name='schp_resnet101', 
                device=str(self.device)
            )
            if self.human_parser.model is not None:
                logger.info("Human parsing model loaded successfully")
                
                # Add optimized wrapper for better performance
                try:
                    from ..processing.optimized_human_parsing import optimize_human_parser
                    self.human_parser.optimized_parser = optimize_human_parser(self.human_parser)
                    logger.info("✅ Human parsing optimization enabled")
                except ImportError:
                    logger.warning("⚠️ Optimized human parsing not available")
                
                self.use_pretrained_parsing = True
            else:
                logger.warning("Human parsing model failed to load, using geometric fallback")
                self.use_pretrained_parsing = False
        except Exception as e:
            logger.warning(f"Failed to initialize human parsing model: {e}")
            logger.info("Using geometric parsing fallback")
            self.human_parser = None
            self.use_pretrained_parsing = False
        
    def enhanced_preprocessing(self, silhouettes: List[np.ndarray], 
                             parsings: Optional[List[np.ndarray]] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Enhanced preprocessing with better quality control and domain adaptation
        
        Args:
            silhouettes: List of silhouette arrays
            parsings: Optional list of parsing arrays
            
        Returns:
            Tuple of (parsing_tensor, silhouette_tensor, sequence_length)
        """
        # 1. Quality filtering and sequence optimization
        filtered_silhouettes, filtered_parsings = self._filter_and_optimize_sequence(
            silhouettes, parsings
        )
        
        if len(filtered_silhouettes) < self.min_sequence_length:
            logger.warning(f"Insufficient frames after filtering: {len(filtered_silhouettes)}")
            # Duplicate frames if necessary
            while len(filtered_silhouettes) < self.min_sequence_length:
                filtered_silhouettes.extend(filtered_silhouettes[:self.min_sequence_length - len(filtered_silhouettes)])
                if filtered_parsings:
                    filtered_parsings.extend(filtered_parsings[:self.min_sequence_length - len(filtered_parsings)])
        
        # 2. Enhanced silhouette preprocessing
        enhanced_silhouettes = []
        for sil in filtered_silhouettes:
            enhanced_sil = self._enhance_silhouette(sil)
            enhanced_silhouettes.append(enhanced_sil)
        
        # 3. Generate or enhance parsing maps using pretrained model
        if filtered_parsings is None:
            if self.use_pretrained_parsing and self.human_parser is not None:
                # Use the pretrained human parsing model
                enhanced_parsings = self._generate_parsing_with_model(enhanced_silhouettes)
            else:
                # Fallback to geometric parsing
                enhanced_parsings = self._generate_enhanced_parsing(enhanced_silhouettes)
        else:
            enhanced_parsings = []
            for parsing in filtered_parsings:
                enhanced_parsing = self._enhance_parsing_map(parsing)
                enhanced_parsings.append(enhanced_parsing)
        
        # 4. Convert to tensors with proper format for XGait
        seq_len = len(enhanced_silhouettes)
        
        # XGait expects 4D tensors: [n, s, h, w] where n=1 (batch size), s=sequence length
        sils_tensor = torch.zeros((1, seq_len, 64, 44), dtype=torch.float32)
        pars_tensor = torch.zeros((1, seq_len, 64, 44), dtype=torch.float32)
        
        for i, (sil, par) in enumerate(zip(enhanced_silhouettes, enhanced_parsings)):
            # Normalize silhouettes to [0, 1]
            sil_normalized = sil.astype(np.float32) / 255.0
            sils_tensor[0, i] = torch.from_numpy(sil_normalized)
            
            # Normalize parsing maps
            par_normalized = par.astype(np.float32) / max(1, par.max()) if par.max() > 0 else par.astype(np.float32)
            pars_tensor[0, i] = torch.from_numpy(par_normalized)
        
        return pars_tensor, sils_tensor, seq_len
    
    def _filter_and_optimize_sequence(self, silhouettes: List[np.ndarray], 
                                    parsings: Optional[List[np.ndarray]]) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
        """Filter and optimize sequence for better quality"""
        
        # Calculate quality scores for each frame
        quality_scores = []
        for sil in silhouettes:
            score = self._calculate_frame_quality(sil)
            quality_scores.append(score)
        
        # Select best frames
        indices_with_scores = list(enumerate(quality_scores))
        indices_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top frames but maintain temporal order
        target_length = min(self.max_sequence_length, len(silhouettes))
        selected_indices = [idx for idx, _ in indices_with_scores[:target_length]]
        selected_indices.sort()  # Maintain temporal order
        
        # Filter frames
        filtered_sils = [silhouettes[i] for i in selected_indices]
        filtered_pars = [parsings[i] for i in selected_indices] if parsings else None
        
        return filtered_sils, filtered_pars
    
    def _calculate_frame_quality(self, silhouette: np.ndarray) -> float:
        """Calculate quality score for a silhouette frame"""
        # Find bounding box
        coords = np.column_stack(np.where(silhouette > 0))
        if coords.size == 0:
            return 0.0
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Quality factors
        area_ratio = np.sum(silhouette > 0) / silhouette.size
        aspect_ratio = (y_max - y_min) / max(1, x_max - x_min)
        completeness = min(1.0, area_ratio * 10)  # Prefer larger silhouettes
        proportion = 1.0 - abs(aspect_ratio - 1.5) / 1.5  # Prefer human-like proportions
        
        # Edge quality (contour smoothness)
        edges = cv2.Canny((silhouette > 0).astype(np.uint8) * 255, 50, 150)
        edge_quality = np.sum(edges) / max(1, np.sum(silhouette > 0))
        edge_quality = min(1.0, edge_quality)
        
        quality = 0.4 * completeness + 0.3 * proportion + 0.3 * edge_quality
        return max(0.0, min(1.0, quality))
    
    def _enhance_silhouette(self, silhouette: np.ndarray) -> np.ndarray:
        """Enhance silhouette quality"""
        # Resize to target size
        enhanced = cv2.resize(silhouette, (44, 64), interpolation=cv2.INTER_LINEAR)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
        
        # Gaussian blur for smoother edges
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
        
        # Ensure binary values
        enhanced = (enhanced > 127).astype(np.uint8) * 255
        
        return enhanced
    
    def _generate_enhanced_parsing(self, silhouettes: List[np.ndarray]) -> List[np.ndarray]:
        """Generate enhanced parsing maps with better body part segmentation"""
        parsing_maps = []
        
        for sil in silhouettes:
            h, w = sil.shape
            parsing_map = np.zeros_like(sil, dtype=np.uint8)
            
            # Find silhouette boundaries
            coords = np.column_stack(np.where(sil > 0))
            if coords.size == 0:
                parsing_maps.append(parsing_map)
                continue
            
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            height = y_max - y_min + 1
            width = x_max - x_min + 1
            
            # Enhanced body part segmentation (8 parts)
            head_end = y_min + int(0.12 * height)       # Head: top 12%
            neck_end = y_min + int(0.18 * height)       # Neck: 12-18%
            shoulder_end = y_min + int(0.30 * height)   # Shoulders: 18-30%
            torso_end = y_min + int(0.60 * height)      # Torso: 30-60%
            thigh_end = y_min + int(0.80 * height)      # Thighs: 60-80%
            # Lower legs: 80-100%
            
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    if sil[y, x] > 0:
                        rel_width_pos = (x - x_min) / max(1, width)
                        rel_y = (y - y_min) / max(1, height)
                        
                        if y < head_end:
                            parsing_map[y, x] = 1  # Head
                        elif y < neck_end:
                            parsing_map[y, x] = 2  # Neck
                        elif y < shoulder_end:
                            # Shoulder area with arm detection
                            if rel_width_pos < 0.2 or rel_width_pos > 0.8:
                                parsing_map[y, x] = 3  # Arms/shoulders
                            else:
                                parsing_map[y, x] = 4  # Upper torso
                        elif y < torso_end:
                            # Main torso with better arm detection
                            if rel_width_pos < 0.15 or rel_width_pos > 0.85:
                                parsing_map[y, x] = 3  # Arms
                            else:
                                parsing_map[y, x] = 4  # Torso
                        elif y < thigh_end:
                            # Thigh area - split left/right
                            if rel_width_pos < 0.5:
                                parsing_map[y, x] = 5  # Left thigh
                            else:
                                parsing_map[y, x] = 6  # Right thigh
                        else:
                            # Lower legs - split left/right
                            if rel_width_pos < 0.5:
                                parsing_map[y, x] = 7  # Left leg
                            else:
                                parsing_map[y, x] = 8  # Right leg
            
            parsing_maps.append(parsing_map)
        
        return parsing_maps
    
    def _enhance_parsing_map(self, parsing_map: np.ndarray) -> np.ndarray:
        """Enhance existing parsing map"""
        # Resize to target size
        enhanced = cv2.resize(parsing_map, (44, 64), interpolation=cv2.INTER_NEAREST)
        
        # Clean up small disconnected regions
        unique_labels = np.unique(enhanced)
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            
            # Find connected components for this label
            mask = (enhanced == label).astype(np.uint8)
            num_labels, labels_im = cv2.connectedComponents(mask)
            
            if num_labels > 2:  # More than just background
                # Keep only the largest component
                component_sizes = [np.sum(labels_im == i) for i in range(1, num_labels)]
                largest_component = np.argmax(component_sizes) + 1
                
                # Remove small components
                for i in range(1, num_labels):
                    if i != largest_component:
                        enhanced[labels_im == i] = 0
        
        return enhanced
    
    def _generate_parsing_with_model(self, silhouettes: List[np.ndarray]) -> List[np.ndarray]:
        """Generate parsing maps using the pretrained human parsing model"""
        if self.human_parser is None or self.human_parser.model is None:
            logger.warning("Human parsing model not available, falling back to geometric parsing")
            return self._generate_enhanced_parsing(silhouettes)
        
        parsing_maps = []
        
        # Batch process parsing to improve efficiency
        try:
            # Prepare batch of RGB images for parsing
            rgb_images = []
            valid_indices = []
            
            for i, sil in enumerate(silhouettes):
                try:
                    rgb_image = self._silhouette_to_rgb(sil)
                    rgb_images.append(rgb_image)
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"Failed to convert silhouette {i} to RGB: {e}")
            
            # Batch process with optimized parser if available
            if rgb_images:
                if hasattr(self.human_parser, 'optimized_parser'):
                    # Use optimized batch processing
                    batch_results = self.human_parser.optimized_parser.parse_batch(rgb_images)
                elif hasattr(self.human_parser, 'parse_batch'):
                    # Use any available batch method
                    batch_results = self.human_parser.parse_batch(rgb_images)
                else:
                    # Fallback to individual processing
                    batch_results = []
                    for rgb_image in rgb_images:
                        parsing_result = self.human_parser.parse_human(rgb_image)
                        batch_results.append(parsing_result)
                
                # Process batch results
                for i, (idx, parsing_result) in enumerate(zip(valid_indices, batch_results)):
                    if parsing_result is not None:
                        # Convert parsing result to XGait format
                        xgait_parsing = self._convert_parsing_to_xgait_format(
                            parsing_result, silhouettes[idx]
                        )
                        parsing_maps.append(xgait_parsing)
                    else:
                        # Fallback to geometric parsing for this frame
                        geometric_parsing = self._generate_enhanced_parsing([silhouettes[idx]])[0]
                        parsing_maps.append(geometric_parsing)
            else:
                # If no valid RGB images, use geometric parsing for all
                parsing_maps = self._generate_enhanced_parsing(silhouettes)
                
        except Exception as e:
            logger.warning(f"Batch parsing failed, using geometric fallback: {e}")
            # Complete fallback to geometric parsing
            parsing_maps = self._generate_enhanced_parsing(silhouettes)
        
        return parsing_maps
    
    def _silhouette_to_rgb(self, silhouette: np.ndarray) -> np.ndarray:
        """Convert binary silhouette to RGB image for parsing model input"""
        # Create a 3-channel image
        h, w = silhouette.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Create a person-like appearance using the silhouette
        mask = silhouette > 0
        
        # Set background to a neutral color
        rgb_image[:, :] = [128, 128, 128]  # Gray background
        
        # Set person area to skin-like color
        rgb_image[mask] = [180, 150, 120]  # Skin tone
        
        # Add some texture/gradient to make it more realistic for the parsing model
        if np.any(mask):
            # Find bounding box
            coords = np.column_stack(np.where(mask))
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            height = y_max - y_min + 1
            
            # Add gradual lighting effect
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    if mask[y, x]:
                        # Add subtle gradient based on position
                        lighting_factor = 0.8 + 0.4 * (x - x_min) / max(1, x_max - x_min)
                        rgb_image[y, x] = (rgb_image[y, x] * lighting_factor).astype(np.uint8)
        
        return rgb_image
    
    def _convert_parsing_to_xgait_format(self, parsing_result: np.ndarray, original_silhouette: np.ndarray) -> np.ndarray:
        """Convert parsing model output to XGait-compatible format"""
        # Resize parsing result to match silhouette size
        h, w = original_silhouette.shape
        if parsing_result.shape[:2] != (h, w):
            parsing_resized = cv2.resize(parsing_result, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            parsing_resized = parsing_result.copy()
        
        # Create XGait-compatible parsing map with proper labels
        xgait_parsing = np.zeros_like(original_silhouette, dtype=np.uint8)
        silhouette_mask = original_silhouette > 0
        
        # Map parsing labels to XGait format (matching Gait3D expectations)
        # XGait typically expects: 0=background, 1-N=body parts
        
        if self.human_parser.model_name == 'schp_resnet101':
            # SCHP labels: {0: 'Background', 1: 'Head', 2: 'Torso', 3: 'Upper-arms', 4: 'Lower-arms', 5: 'Upper-legs', 6: 'Lower-legs'}
            label_mapping = {
                0: 0,  # Background
                1: 1,  # Head
                2: 2,  # Torso -> Upper body
                3: 3,  # Upper arms -> Arms
                4: 3,  # Lower arms -> Arms (combine with upper arms)
                5: 4,  # Upper legs -> Thighs
                6: 5   # Lower legs -> Lower legs
            }
        else:
            # For other models (LIP, ATR), create a more detailed mapping
            # Map to common body parts that XGait can use
            unique_labels = np.unique(parsing_resized)
            label_mapping = {}
            
            for label in unique_labels:
                if label == 0:
                    label_mapping[label] = 0  # Background
                elif label in [1, 2, 11, 13]:  # Hat, Hair, Face, etc.
                    label_mapping[label] = 1  # Head
                elif label in [4, 5, 6, 7, 10]:  # Upper clothes, dress, coat, etc.
                    label_mapping[label] = 2  # Torso
                elif label in [14, 15]:  # Arms
                    label_mapping[label] = 3  # Arms
                elif label in [9, 12, 13, 16, 17]:  # Pants, legs
                    label_mapping[label] = 4  # Legs
                else:
                    label_mapping[label] = 2  # Default to torso
        
        # Apply mapping only within silhouette area
        for original_label, new_label in label_mapping.items():
            mask = (parsing_resized == original_label) & silhouette_mask
            xgait_parsing[mask] = new_label
        
        # Ensure all silhouette pixels have some label (no background within silhouette)
        unlabeled_mask = (xgait_parsing == 0) & silhouette_mask
        if np.any(unlabeled_mask):
            # Assign unlabeled pixels to nearest labeled region
            xgait_parsing = self._fill_unlabeled_regions(xgait_parsing, unlabeled_mask)
        
        # Resize to target size
        final_parsing = cv2.resize(xgait_parsing, (44, 64), interpolation=cv2.INTER_NEAREST)
        
        return final_parsing
    
    def _fill_unlabeled_regions(self, parsing_map: np.ndarray, unlabeled_mask: np.ndarray) -> np.ndarray:
        """Fill unlabeled regions with nearest labeled pixels"""
        result = parsing_map.copy()
        
        if not np.any(unlabeled_mask):
            return result
        
        # Get labeled regions
        labeled_mask = (parsing_map > 0) & (~unlabeled_mask)
        
        if not np.any(labeled_mask):
            # If no labeled regions, assign default label (torso)
            result[unlabeled_mask] = 2
            return result
        
        # Simple approach: assign to most common neighboring label
        for y, x in np.column_stack(np.where(unlabeled_mask)):
            # Check 3x3 neighborhood
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < parsing_map.shape[0] and 
                        0 <= nx < parsing_map.shape[1] and 
                        parsing_map[ny, nx] > 0):
                        neighbors.append(parsing_map[ny, nx])
            
            if neighbors:
                # Assign most common neighbor label
                result[y, x] = max(set(neighbors), key=neighbors.count)
            else:
                # Default to torso if no neighbors
                result[y, x] = 2
        
        return result
    
    def enhanced_inference(self, silhouettes: List[np.ndarray], 
                         parsings: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        """
        Enhanced inference with better preprocessing and error handling
        
        Args:
            silhouettes: List of silhouette frames
            parsings: Optional parsing maps
            
        Returns:
            Feature embeddings or None if failed
        """
        try:
            # Enhanced preprocessing
            pars_tensor, sils_tensor, seq_len = self.enhanced_preprocessing(silhouettes, parsings)
            
            # Move to device
            pars_tensor = pars_tensor.to(self.device)
            sils_tensor = sils_tensor.to(self.device)
            
            # Create model inputs
            labs = torch.zeros(1).long().to(self.device)
            typs = torch.zeros(1).long().to(self.device)
            vies = torch.zeros(1).long().to(self.device)
            seqL = [torch.tensor([seq_len], dtype=torch.long).to(self.device)]
            
            # Try different input formats
            # XGait expects: ipts = [pars, sils] where both are 4D tensors [n, s, h, w]
            logger.debug(f"pars_tensor shape: {pars_tensor.shape}, sils_tensor shape: {sils_tensor.shape}")
            
            input_formats = [
                # Format 1: Standard XGait format - pars and sils as 4D tensors [1, seq_len, 64, 44]
                ([pars_tensor, sils_tensor], labs, typs, vies, seqL),
                # Format 2: Try squeezed version if needed
                ([pars_tensor.squeeze(0), sils_tensor.squeeze(0)], labs, typs, vies, seqL)
            ]
            
            embeddings = None
            for i, inputs in enumerate(input_formats):
                try:
                    with torch.no_grad():
                        outputs = self.gait_recognizer.model(inputs)
                        embeddings = outputs['inference_feat']['embeddings']
                        logger.info(f"Success with input format {i+1}")
                        break
                except Exception as e:
                    if i < len(input_formats) - 1:
                        logger.debug(f"Format {i+1} failed: {e}, trying next format...")
                    else:
                        logger.error(f"All input formats failed. Last error: {e}")
                        return None
            
            if embeddings is None:
                return None
            
            # Post-process embeddings
            embeddings = embeddings.cpu().numpy()
            
            # Apply domain adaptation if enabled
            if self.apply_domain_adaptation:
                embeddings = self._apply_domain_adaptation(embeddings)
            
            # Normalize features if enabled
            if self.normalize_features:
                embeddings = self._normalize_features(embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Enhanced inference failed: {e}")
            return None
    
    def _apply_domain_adaptation(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply simple domain adaptation techniques"""
        # Handle different embedding shapes
        original_shape = embeddings.shape
        
        # Flatten to 2D if needed: [batch, features]
        if embeddings.ndim > 2:
            embeddings_2d = embeddings.reshape(embeddings.shape[0], -1)
        else:
            embeddings_2d = embeddings
        
        # Center the features (remove mean)
        embeddings_centered = embeddings_2d - np.mean(embeddings_2d, axis=1, keepdims=True)
        
        # Apply whitening (optional) - only if we have enough features
        if embeddings_centered.shape[1] > 1 and embeddings_centered.shape[0] > 1:
            try:
                cov = np.cov(embeddings_centered.T)
                if cov.ndim == 0:  # Scalar covariance
                    cov = np.array([[cov]])
                elif cov.ndim == 1:  # 1D covariance
                    cov = np.diag(cov)
                    
                eigenvals, eigenvecs = np.linalg.eigh(cov + 1e-5 * np.eye(cov.shape[0]))
                # Avoid division by very small eigenvalues
                eigenvals = np.maximum(eigenvals, 1e-6)
                whitening_transform = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
                embeddings_whitened = embeddings_centered @ whitening_transform
                
                # Reshape back to original shape if needed
                if len(original_shape) > 2:
                    return embeddings_whitened.reshape(original_shape)
                return embeddings_whitened
            except Exception as e:
                logger.warning(f"Whitening failed, using centered features: {e}")
                if len(original_shape) > 2:
                    return embeddings_centered.reshape(original_shape)
                return embeddings_centered
        
        # If whitening is not applicable, return centered features
        if len(original_shape) > 2:
            return embeddings_centered.reshape(original_shape)
        return embeddings_centered
    
    def _normalize_features(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize feature embeddings"""
        # L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        return embeddings / norms
    
    def validate_model_compatibility(self) -> Dict[str, Any]:
        """Validate model compatibility and return diagnostic info"""
        diagnostics = {
            'model_loaded': self.gait_recognizer.model is not None,
            'device': str(self.device),
            'model_components': [],
            'test_inference': False,
            'errors': []
        }
        
        if not diagnostics['model_loaded']:
            diagnostics['errors'].append("Model not loaded")
            return diagnostics
        
        # Check model components
        required_components = ['Backbone_sil', 'Backbone_par', 'gcm', 'FCs_sil', 'FCs_par']
        for component in required_components:
            if hasattr(self.gait_recognizer.model, component):
                diagnostics['model_components'].append(component)
        
        # Test inference with dummy data
        try:
            dummy_silhouettes = self._create_dummy_silhouettes()
            result = self.enhanced_inference(dummy_silhouettes)
            diagnostics['test_inference'] = result is not None
            if result is not None:
                diagnostics['output_shape'] = result.shape
        except Exception as e:
            diagnostics['errors'].append(f"Test inference failed: {e}")
        
        return diagnostics
    
    def _create_dummy_silhouettes(self, num_frames: int = 10) -> List[np.ndarray]:
        """Create dummy silhouettes for testing"""
        dummy_silhouettes = []
        for i in range(num_frames):
            sil = np.zeros((64, 44), dtype=np.uint8)
            # Create simple human-like shape
            cv2.ellipse(sil, (22, 16), (8, 12), 0, 0, 360, 255, -1)  # Head
            cv2.rectangle(sil, (14, 28), (30, 50), 255, -1)  # Torso
            cv2.rectangle(sil, (16, 50), (21, 62), 255, -1)  # Left leg
            cv2.rectangle(sil, (23, 50), (28, 62), 255, -1)  # Right leg
            dummy_silhouettes.append(sil)
        return dummy_silhouettes
    
    def validate_preprocessing_consistency(self, probe_silhouettes, gallery_silhouettes):
        """Validate preprocessing consistency between probe and gallery"""
        probe_enhanced = [self._enhance_silhouette(sil) for sil in probe_silhouettes]
        gallery_enhanced = [self._enhance_silhouette(sil) for sil in gallery_silhouettes]

        if len(probe_enhanced) != len(gallery_enhanced):
            raise ValueError("Probe and gallery sequences have different lengths after preprocessing")

        for p, g in zip(probe_enhanced, gallery_enhanced):
            if p.shape != g.shape:
                raise ValueError("Mismatch in silhouette shapes between probe and gallery")

        print("✅ Preprocessing consistency validated")

    def validate_embedding_extraction(self, probe_embeddings, gallery_embeddings):
        """Validate embedding extraction consistency"""
        if probe_embeddings.shape != gallery_embeddings.shape:
            raise ValueError("Probe and gallery embeddings have different shapes")

        if np.isnan(probe_embeddings).any() or np.isnan(gallery_embeddings).any():
            raise ValueError("NaN values detected in embeddings")

        if np.isinf(probe_embeddings).any() or np.isinf(gallery_embeddings).any():
            raise ValueError("Infinite values detected in embeddings")

        print("✅ Embedding extraction consistency validated")

    def compare_embeddings(self, probe_embeddings, gallery_embeddings):
        """Compare probe and gallery embeddings for identity matching"""
        similarity_scores = np.dot(probe_embeddings, gallery_embeddings.T)
        return similarity_scores
