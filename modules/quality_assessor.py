"""
Quality Assessment Module for Gait Sequences
Evaluates the quality of gait sequences for robust person identification
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
import config  # Import at module level, not in method

class GaitSequenceQualityAssessor:
    """Assess the quality of gait sequences for person identification"""
    
    def __init__(self, config_override: Dict = None):
        """
        Initialize the quality assessor
        
        Args:
            config_override (dict): Configuration parameters to override defaults
        """
        self.config = self._get_default_config()
        if config_override:
            self._update_nested_config(self.config, config_override)
        
        self.logger = logging.getLogger(__name__)
        self.pose_estimator = None  # Initialize if needed
        
    def _update_nested_config(self, base_config: Dict, update_config: Dict) -> None:
        """
        Update configuration recursively while preserving nested structure
        
        Args:
            base_config: Base configuration to update
            update_config: New configuration values
        """
        for key, value in update_config.items():
            if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
                self._update_nested_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def _get_default_config(self) -> Dict:
        """Default configuration for quality assessment"""
        return {
            'min_sequence_length': 15,      # Minimum frames for reliable gait analysis
            'max_sequence_length': 120,     # Maximum frames to avoid redundancy
            'min_silhouette_area': 500,     # Minimum silhouette area in pixels
            'max_silhouette_area': 50000,   # Maximum silhouette area in pixels
            'min_aspect_ratio': 0.3,        # Minimum height/width ratio
            'max_aspect_ratio': 4.0,        # Maximum height/width ratio
            'min_motion_threshold': 10,     # Minimum motion between frames
            'max_motion_threshold': 200,    # Maximum motion (too fast movement)
            'consistency_threshold': 0.7,   # Silhouette consistency across frames
            'completeness_threshold': 0.8,  # Percentage of complete silhouettes
            'sharpness_threshold': 50,      # Minimum sharpness (Laplacian variance)
            'temporal_consistency_window': 5, # Window for temporal consistency check
            'gait_cycle_min_frames': 10,    # Minimum frames for one gait cycle
            'pose_variation_threshold': 0.3, # Minimum pose variation for gait analysis
            'min_length_threshold': 0.2,    # Minimum threshold for length score
            'min_silhouette_threshold': 0.1, # Minimum threshold for silhouette quality score
            'quality_thresholds': {         # Thresholds for quality levels
                'excellent': 0.8,
                'good': 0.65,
                'fair': 0.5,
                'poor': 0.3
            },
            'quality_weights': {            # Component weights for overall score
                'length': 0.10,
                'silhouette': 0.15,
                'motion': 0.25,
                'temporal': 0.15,
                'completeness': 0.15,
                'pose': 0.20,
                'sharpness': 0.05
            }
        }
    
    def _standardize_silhouette_format(self, silhouette: np.ndarray) -> np.ndarray:
        """
        Convert silhouette to standard uint8 format for consistent processing
        
        Args:
            silhouette: Input silhouette array
            
        Returns:
            Standardized uint8 silhouette
        """
        if silhouette is None or silhouette.size == 0:
            return None
            
        if silhouette.dtype != np.uint8:
            return (silhouette * 255).astype(np.uint8) if silhouette.max() <= 1.0 else silhouette.astype(np.uint8)
        return silhouette
    
    def assess_sequence_quality(self, silhouettes: List[np.ndarray], 
                              bboxes: List[Tuple[int, int, int, int]] = None) -> Dict:
        """
        Comprehensive quality assessment of a gait sequence
        
        Args:
            silhouettes: List of silhouette images (binary masks)
            bboxes: Optional list of bounding boxes for each frame
            
        Returns:
            Dict containing overall quality score, metrics, and assessment level
        """
        # Initialize metrics collection
        collected_metrics = {}
        
        try:
            # Basic validation
            if not silhouettes or len(silhouettes) == 0:
                return self._create_quality_result(0.0, "Empty sequence", {})
            
            # Early stopping for clearly poor sequences
            if len(silhouettes) < self.config['min_sequence_length'] // 2:
                return self._create_quality_result(0.0, "Too short for analysis", {})
            
            # Quick quality check - if too many empty silhouettes, stop early
            non_empty_count = sum(1 for s in silhouettes[:10] if s is not None and s.size > 0 and np.sum(s > 0) > 100)
            if non_empty_count < len(silhouettes[:10]) * 0.3:  # Less than 30% valid silhouettes
                return self._create_quality_result(0.0, "Too few valid silhouettes", {})
            
            # Pre-process silhouettes to standard format (do conversion once)
            std_silhouettes = [self._standardize_silhouette_format(sil) for sil in silhouettes if sil is not None]
            if not std_silhouettes:
                return self._create_quality_result(
                    0.0, 
                    "No valid silhouettes", 
                    {'error': 'All silhouettes are None or empty'}
                )

            # 1. Assess sequence length
            length_score, length_metrics = self._assess_sequence_length(std_silhouettes)
            collected_metrics['length'] = length_metrics
            
            if length_score < self.config['min_length_threshold']:
                return self._create_quality_result(
                    length_score, 
                    "Inadequate sequence length",
                    collected_metrics,
                    failed_check="length"
                )
                
            # 2. Assess silhouette quality
            silhouette_score, silhouette_metrics = self._assess_silhouette_quality(std_silhouettes)
            collected_metrics['silhouette'] = silhouette_metrics
            
            if silhouette_score < self.config['min_silhouette_threshold']:
                return self._create_quality_result(
                    silhouette_score,
                    "Poor silhouette quality", 
                    collected_metrics,
                    failed_check="silhouette"
                )
                
            # 3. Assess motion quality
            motion_score, motion_metrics = self._assess_motion_quality(std_silhouettes)
            collected_metrics['motion'] = motion_metrics
            
            # 4. Assess temporal consistency
            temporal_score, temporal_metrics = self._assess_temporal_consistency(std_silhouettes)
            collected_metrics['temporal'] = temporal_metrics
            
            # 5. Assess sequence completeness
            completeness_score, completeness_metrics = self._assess_completeness(std_silhouettes)
            collected_metrics['completeness'] = completeness_metrics
            
            # 6. Assess pose quality
            pose_score, pose_metrics = self._assess_pose_variation(std_silhouettes)
            collected_metrics['pose'] = pose_metrics
                
            # 7. Assess sharpness
            sharpness_score, sharpness_metrics = self._assess_sharpness(std_silhouettes)
            collected_metrics['sharpness'] = sharpness_metrics
            
            # Calculate weighted score based on configured weights
            component_scores = {
                'length': length_score,
                'silhouette': silhouette_score,
                'motion': motion_score,
                'temporal': temporal_score,
                'completeness': completeness_score,
                'pose': pose_score,
                'sharpness': sharpness_score
            }
            
            # Use weights from config
            weights = self.config['quality_weights']
            total_score = sum(score * weights[component] for component, score in component_scores.items())
            
            # Add component scores to metrics
            collected_metrics['component_scores'] = component_scores
            
            # Determine quality level based on score thresholds
            quality_level = self._determine_quality_level(total_score)
            
            return self._create_quality_result(
                total_score,
                quality_level,
                collected_metrics
            )
                
        except Exception as e:
            self.logger.error(f"Error in quality assessment: {str(e)}", exc_info=True)
            return self._create_quality_result(
                0.0, 
                f"Assessment error", 
                {'error': str(e)}
            )
    
    def _assess_sequence_length(self, silhouettes: List[np.ndarray]) -> Tuple[float, Dict]:
        """
        Assess sequence length quality
        
        Args:
            silhouettes: List of standardized silhouettes
            
        Returns:
            Tuple of (quality score, metrics dictionary)
        """
        length = len(silhouettes)
        min_len = self.config['min_sequence_length']
        max_len = self.config['max_sequence_length']
        
        if length < min_len:
            score = length / min_len * 0.5  # Penalize short sequences heavily
        elif length <= max_len:
            score = 1.0  # Optimal length
        else:
            # Diminishing returns for very long sequences
            score = max(0.7, 1.0 - (length - max_len) / max_len * 0.3)
        
        metrics = {
            'sequence_length': length,
            'is_adequate_length': length >= min_len,
            'length_category': 'short' if length < min_len else 'optimal' if length <= max_len else 'long'
        }
        
        return score, metrics
    
    def _assess_silhouette_quality(self, silhouettes: List[np.ndarray]) -> Tuple[float, Dict]:
        """
        Assess individual silhouette quality
        
        Args:
            silhouettes: List of standardized silhouettes
            
        Returns:
            Tuple of (quality score, metrics dictionary)
        """
        if not silhouettes:
            return 0.0, self._get_default_metrics('silhouette')
            
        valid_silhouettes = 0
        total_area = 0
        total_aspect_ratios = []
        quality_scores = []
        
        for sil in silhouettes:
            if sil is None or sil.size == 0:
                continue
                
            # Calculate silhouette area
            area = np.sum(sil > 0)
            
            # Skip silhouettes with insufficient area
            if area < self.config['min_silhouette_area']:
                continue
                
            if area <= self.config['max_silhouette_area']:
                try:
                    # Calculate aspect ratio
                    contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        x, y, w, h = cv2.boundingRect(contours[0])
                        aspect_ratio = h / w if w > 0 else 0
                        
                        if (aspect_ratio >= self.config['min_aspect_ratio'] and 
                            aspect_ratio <= self.config['max_aspect_ratio']):
                            valid_silhouettes += 1
                            total_area += area
                            total_aspect_ratios.append(aspect_ratio)
                            
                            # Calculate individual quality score based on area and aspect ratio
                            area_score = min(1.0, area / (self.config['max_silhouette_area'] * 0.25))
                            aspect_score = 1.0 - abs(aspect_ratio - 1.7) / 2.0  # Optimal is around 1.7 (human body)
                            quality_scores.append((area_score + aspect_score) / 2)
                except cv2.error as e:
                    self.logger.warning(f"OpenCV error in silhouette analysis: {e}")
                    continue
        
        if len(silhouettes) == 0:
            return 0.0, self._get_default_metrics('silhouette')
        
        quality_ratio = valid_silhouettes / len(silhouettes)
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        avg_area = total_area / max(valid_silhouettes, 1)
        avg_aspect_ratio = np.mean(total_aspect_ratios) if total_aspect_ratios else 0
        
        # Final score is a combination of quality ratio and average quality
        final_score = (quality_ratio * 0.6) + (avg_quality * 0.4)
        
        metrics = {
            'valid_silhouettes': valid_silhouettes,
            'total_silhouettes': len(silhouettes),
            'quality_ratio': quality_ratio,
            'average_area': avg_area,
            'average_aspect_ratio': avg_aspect_ratio,
            'quality_scores': quality_scores,
            'avg_quality': avg_quality
        }
        
        return final_score, metrics
    
    def _assess_motion_quality(self, silhouettes: List[np.ndarray]) -> Tuple[float, Dict]:
        """
        Assess motion quality between frames
        
        Args:
            silhouettes: List of standardized silhouettes
            
        Returns:
            Tuple of (quality score, metrics dictionary)
        """
        if len(silhouettes) < 2:
            return 0.0, self._get_default_metrics('motion')
        
        motion_vectors = []
        valid_motions = 0
        
        # Vectorized approach for motion calculation
        try:
            # Create pairs of consecutive frames
            frame_pairs = [(silhouettes[i-1], silhouettes[i]) for i in range(1, len(silhouettes))
                          if silhouettes[i-1] is not None and silhouettes[i] is not None]
            
            if not frame_pairs:
                return 0.0, self._get_default_metrics('motion')
                
            for prev_frame, curr_frame in frame_pairs:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame.astype(np.float32), curr_frame.astype(np.float32))
                motion_magnitude = np.sum(diff)
                
                motion_vectors.append(motion_magnitude)
                
                if (motion_magnitude >= self.config['min_motion_threshold'] and 
                    motion_magnitude <= self.config['max_motion_threshold']):
                    valid_motions += 1
                    
        except Exception as e:
            self.logger.warning(f"Error in motion assessment: {e}")
            return 0.3, {'error': str(e), 'motion_vectors': []}
        
        if len(motion_vectors) == 0:
            return 0.0, self._get_default_metrics('motion')
        
        # Calculate statistics once
        mean_motion = np.mean(motion_vectors)
        std_motion = np.std(motion_vectors)
        
        motion_score = valid_motions / len(motion_vectors)
        motion_consistency = 1.0 - (std_motion / (mean_motion + 1e-6))
        motion_consistency = max(0.0, min(1.0, motion_consistency))
        
        # Combine motion validity and consistency
        combined_score = (motion_score * 0.6) + (motion_consistency * 0.4)
        
        metrics = {
            'motion_vectors': motion_vectors,
            'valid_motions': valid_motions,
            'motion_score': motion_score,
            'motion_consistency': motion_consistency,
            'avg_motion': mean_motion,
            'motion_std': std_motion
        }
        
        return combined_score, metrics
    
    def _assess_temporal_consistency(self, silhouettes: List[np.ndarray]) -> Tuple[float, Dict]:
        """
        Assess temporal consistency of silhouettes
        
        Args:
            silhouettes: List of standardized silhouettes
            
        Returns:
            Tuple of (quality score, metrics dictionary)
        """
        if len(silhouettes) < self.config['temporal_consistency_window']:
            return 0.5, self._get_default_metrics('temporal')
        
        window_size = self.config['temporal_consistency_window']
        consistency_scores = []
        
        for i in range(len(silhouettes) - window_size + 1):
            window = silhouettes[i:i + window_size]
            
            # Calculate pairwise similarities within window
            similarities = self._calculate_pairwise_similarities(window)
            
            if similarities:
                consistency_scores.append(np.mean(similarities))
        
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        metrics = {
            'consistency_scores': consistency_scores,
            'overall_consistency': overall_consistency,
            'temporal_windows_analyzed': len(consistency_scores)
        }
        
        return overall_consistency, metrics
        
    def _calculate_pairwise_similarities(self, frames: List[np.ndarray]) -> List[float]:
        """
        Calculate pairwise similarities between frames in a window
        
        Args:
            frames: List of frames to compare
            
        Returns:
            List of similarity scores
        """
        similarities = []
        for j in range(len(frames)):
            for k in range(j + 1, len(frames)):
                if frames[j] is None or frames[k] is None:
                    continue
                
                try:
                    # Try template matching first (most accurate)
                    similarity = self._calculate_template_matching(frames[j], frames[k])
                    similarities.append(similarity)
                except Exception as e:
                    try:
                        # Fallback to correlation
                        similarity = self._calculate_correlation(frames[j], frames[k])
                        similarities.append(similarity)
                    except Exception as e2:
                        self.logger.warning(f"Both similarity calculation methods failed: {e}, then {e2}")
                        similarities.append(0.0)
        
        return similarities
    
    def _calculate_template_matching(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate similarity using template matching"""
        # Ensure correct data types
        img1 = img1.astype(np.float32) if img1.dtype != np.float32 else img1
        img2 = img2.astype(np.float32) if img2.dtype != np.float32 else img2
        
        # Normalize to 0-1 range if needed
        if img1.max() > 1.0:
            img1 = img1 / 255.0
        if img2.max() > 1.0:
            img2 = img2 / 255.0
        
        # Ensure same shape
        if img1.shape != img2.shape:
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]
        
        # Use template matching
        if img1.shape[0] >= img2.shape[0] and img1.shape[1] >= img2.shape[1]:
            result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        else:
            result = cv2.matchTemplate(img2, img1, cv2.TM_CCOEFF_NORMED)
        
        return float(max(0.0, np.max(result)))
    
    def _calculate_correlation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate similarity using correlation"""
        img1_flat = img1.flatten().astype(np.float64)
        img2_flat = img2.flatten().astype(np.float64)
        
        # Normalize
        img1_norm = (img1_flat - np.mean(img1_flat)) / (np.std(img1_flat) + 1e-8)
        img2_norm = (img2_flat - np.mean(img2_flat)) / (np.std(img2_flat) + 1e-8)
        
        # Calculate correlation
        corr = np.corrcoef(img1_norm, img2_norm)[0, 1]
        return float(max(0.0, corr if not np.isnan(corr) else 0.0))
    
    def _assess_completeness(self, silhouettes: List[np.ndarray]) -> Tuple[float, Dict]:
        """
        Assess completeness of silhouettes (no missing parts)
        
        Args:
            silhouettes: List of standardized silhouettes
            
        Returns:
            Tuple of (quality score, metrics dictionary)
        """
        if not silhouettes:
            return 0.0, self._get_default_metrics('completeness')
            
        complete_silhouettes = 0
        completeness_ratios = []
        
        kernel = np.ones((3, 3), np.uint8)
        
        for sil in silhouettes:
            if sil is None:
                continue
            
            try:
                # Check if silhouette has reasonable connectivity
                # Use morphological operations to check completeness
                closed = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, kernel)
                opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
                
                # Calculate ratio of filled area after morphological operations
                original_area = np.sum(sil > 0)
                processed_area = np.sum(opened > 0)
                
                if original_area > 0:
                    completeness_ratio = processed_area / original_area
                    completeness_ratios.append(completeness_ratio)
                    
                    if completeness_ratio >= self.config['completeness_threshold']:
                        complete_silhouettes += 1
            except cv2.error as e:
                self.logger.warning(f"OpenCV error in completeness assessment: {e}")
                continue
        
        completeness_score = complete_silhouettes / len(silhouettes) if silhouettes else 0.0
        avg_completeness = np.mean(completeness_ratios) if completeness_ratios else 0.0
        
        metrics = {
            'complete_silhouettes': complete_silhouettes,
            'total_silhouettes': len(silhouettes),
            'completeness_score': completeness_score,
            'avg_completeness_ratio': avg_completeness,
            'completeness_ratios': completeness_ratios
        }
        
        return completeness_score, metrics
    
    def _assess_pose_variation(self, silhouettes: List[np.ndarray]) -> Tuple[float, Dict]:
        """
        Assess pose variation to ensure gait cycle representation
        
        Args:
            silhouettes: List of standardized silhouettes
            
        Returns:
            Tuple of (quality score, metrics dictionary)
        """
        if len(silhouettes) < self.config['gait_cycle_min_frames']:
            return 0.3, self._get_default_metrics('pose')
        
        pose_features = []
        
        try:
            for sil in silhouettes:
                if sil is None:
                    continue
                    
                # Extract simple pose features (moments, contour properties)
                moments = cv2.moments(sil)
                if moments['m00'] > 0:
                    # Centroid
                    cx = moments['m10'] / moments['m00']
                    cy = moments['m01'] / moments['m00']
                    
                    # Hu moments for shape description
                    hu_moments = cv2.HuMoments(moments).flatten()
                    
                    # Combine features
                    features = [cx, cy] + hu_moments.tolist()
                    pose_features.append(features)
            
            if len(pose_features) < 2:
                return 0.3, self._get_default_metrics('pose')
            
            # Calculate variation in pose features
            pose_features = np.array(pose_features)
            pose_std = np.std(pose_features, axis=0)
            pose_variation = np.mean(pose_std)
            
            # Normalize variation score
            variation_score = min(1.0, pose_variation / self.config['pose_variation_threshold'])
            
            metrics = {
                'pose_variations': pose_std.tolist(),
                'mean_pose_variation': pose_variation,
                'variation_score': variation_score,
                'frames_analyzed': len(pose_features)
            }
            
            return variation_score, metrics
            
        except Exception as e:
            self.logger.warning(f"Error in pose variation assessment: {e}")
            return 0.3, {'error': str(e), 'pose_variations': []}
    
    def _assess_sharpness(self, silhouettes: List[np.ndarray]) -> Tuple[float, Dict]:
        """
        Assess sharpness/clarity of silhouettes
        
        Args:
            silhouettes: List of standardized silhouettes
            
        Returns:
            Tuple of (quality score, metrics dictionary)
        """
        if not silhouettes:
            return 0.0, self._get_default_metrics('sharpness')
            
        sharpness_scores = []
        
        for sil in silhouettes:
            if sil is None:
                continue
                
            try:
                # Calculate Laplacian variance as sharpness measure
                laplacian = cv2.Laplacian(sil, cv2.CV_64F)
                sharpness = laplacian.var()
                sharpness_scores.append(sharpness)
            except cv2.error as e:
                self.logger.warning(f"Laplacian calculation failed: {e}")
                try:
                    # Fallback: use Sobel operators if Laplacian fails
                    sobelx = cv2.Sobel(sil, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(sil, cv2.CV_64F, 0, 1, ksize=3)
                    sharpness = np.sqrt(sobelx**2 + sobely**2).var()
                    sharpness_scores.append(sharpness)
                except Exception as e2:
                    self.logger.warning(f"Sobel calculation failed: {e2}")
                    try:
                        # Ultimate fallback: use gradient-based measure
                        grad_x = np.gradient(sil.astype(np.float64), axis=1)
                        grad_y = np.gradient(sil.astype(np.float64), axis=0)
                        sharpness = (grad_x**2 + grad_y**2).var()
                        sharpness_scores.append(sharpness)
                    except Exception as e3:
                        self.logger.warning(f"All sharpness calculations failed: {e3}")
                        # Skip this silhouette
        
        if not sharpness_scores:
            return 0.0, self._get_default_metrics('sharpness')
        
        avg_sharpness = np.mean(sharpness_scores)
        sharpness_score = min(1.0, avg_sharpness / self.config['sharpness_threshold'])
        
        metrics = {
            'sharpness_scores': sharpness_scores,
            'average_sharpness': avg_sharpness,
            'sharpness_score': sharpness_score,
            'min_sharpness': min(sharpness_scores),
            'max_sharpness': max(sharpness_scores),
        }
        
        return sharpness_score, metrics
    
    def _get_default_metrics(self, metric_type: str) -> Dict:
        """
        Return default metrics structure for given type
        
        Args:
            metric_type: Type of metrics to generate defaults for
            
        Returns:
            Default metrics dictionary
        """
        defaults = {
            'length': {'sequence_length': 0, 'is_adequate_length': False, 'length_category': 'short'},
            'silhouette': {'valid_silhouettes': 0, 'total_silhouettes': 0, 'quality_ratio': 0.0},
            'motion': {'motion_vectors': [], 'valid_motions': 0, 'motion_consistency': 0.0},
            'temporal': {'consistency_scores': [], 'overall_consistency': 0.0},
            'completeness': {'complete_silhouettes': 0, 'total_silhouettes': 0, 'completeness_score': 0.0},
            'pose': {'pose_variations': [], 'mean_pose_variation': 0.0, 'variation_score': 0.0},
            'sharpness': {'sharpness_scores': [], 'average_sharpness': 0.0, 'sharpness_score': 0.0}
        }
        return defaults.get(metric_type, {})
    
    def _determine_quality_level(self, score: float) -> str:
        """
        Determine quality level based on overall score
        
        Args:
            score: Quality score between 0.0 and 1.0
            
        Returns:
            String quality level description
        """
        thresholds = self.config['quality_thresholds']
        
        if score >= thresholds['excellent']:
            return "Excellent"
        elif score >= thresholds['good']:
            return "Good"
        elif score >= thresholds['fair']:
            return "Fair"
        elif score >= thresholds['poor']:
            return "Poor"
        else:
            return "Very Poor"
        
    def _create_quality_result(self, score: float, level: str, metrics: Dict, failed_check: str = None) -> Dict:
        """
        Create standardized quality assessment result
        
        Args:
            score: Overall quality score
            level: Quality level description
            metrics: Dictionary of detailed metrics
            failed_check: Optional name of check that failed
            
        Returns:
            Standardized result dictionary
        """
        # Get thresholds from configuration
        high_quality_threshold = getattr(config, 'HIGH_QUALITY_THRESHOLD', 0.7)
        acceptable_threshold = 0.5  # Keep this standard
        
        result = {
            'overall_score': round(float(score), 3),
            'quality_level': level,
            'is_acceptable': score >= acceptable_threshold,
            'is_high_quality': score >= high_quality_threshold,
            'metrics': metrics,
            'recommendations': self._generate_recommendations(score, metrics),
            'timestamp': datetime.now().isoformat()
        }
        
        if failed_check:
            result['failed_check'] = failed_check
            
        return result
    
    def _generate_recommendations(self, score: float, metrics: Dict) -> List[str]:
        """
        Generate recommendations for improving sequence quality
        
        Args:
            score: Overall quality score
            metrics: Dictionary of detailed metrics
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if score < 0.5:
            recommendations.append("Sequence quality is below acceptable threshold")
        
        # Length recommendations
        if 'length' in metrics and metrics['length'].get('sequence_length', 0) < self.config['min_sequence_length']:
            recommendations.append("Collect longer gait sequence for better reliability")
        elif 'length' in metrics and metrics['length'].get('sequence_length', 0) > self.config['max_sequence_length']:
            recommendations.append("Consider using a shorter sequence to reduce redundancy")
        
        # Motion recommendations
        if 'motion' in metrics and metrics['motion'].get('motion_consistency', 1.0) < 0.5:
            recommendations.append("Ensure consistent walking speed")
            
        if 'motion' in metrics and metrics['motion'].get('motion_score', 1.0) < 0.5:
            if metrics['motion'].get('avg_motion', 0) < self.config['min_motion_threshold']:
                recommendations.append("Ensure subject is walking with sufficient motion between frames")
            elif metrics['motion'].get('avg_motion', 0) > self.config['max_motion_threshold']:
                recommendations.append("Reduce walking speed or increase frame rate to capture smoother motion")
        
        # Silhouette recommendations
        if 'silhouette' in metrics and metrics['silhouette'].get('quality_ratio', 1.0) < 0.6:
            recommendations.append("Improve silhouette extraction quality")
        
        # Completeness recommendations
        if 'completeness' in metrics and metrics['completeness'].get('completeness_score', 1.0) < 0.7:
            recommendations.append("Improve silhouette extraction to reduce missing parts")
        
        # Sharpness recommendations
        if 'sharpness' in metrics and metrics['sharpness'].get('average_sharpness', 0) < self.config['sharpness_threshold']:
            recommendations.append("Improve video quality or reduce motion blur")
        
        return recommendations

    def is_sequence_acceptable(self, quality_result: Dict) -> bool:
        """
        Check if sequence meets minimum quality standards
        
        Args:
            quality_result: Quality assessment result dictionary
            
        Returns:
            Boolean indicating if sequence is acceptable
        """
        return quality_result.get('is_acceptable', False)
    
    def is_sequence_high_quality(self, quality_result: Dict) -> bool:
        """
        Check if sequence is high quality
        
        Args:
            quality_result: Quality assessment result dictionary
            
        Returns:
            Boolean indicating if sequence is high quality
        """
        return quality_result.get('is_high_quality', False)