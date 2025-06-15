"""
Quality Assessment Module for Gait Sequences
Evaluates the quality of gait sequences for robust person identification
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging

class GaitSequenceQualityAssessor:
    """Assess the quality of gait sequences for person identification"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the quality assessor
        
        Args:
            config (dict): Configuration parameters for quality assessment
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
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
        }
    
    def assess_sequence_quality(self, silhouettes: List[np.ndarray], 
                              bboxes: List[Tuple[int, int, int, int]] = None) -> Dict:
        """
        Comprehensive quality assessment of a gait sequence
        
        Args:
            silhouettes (List[np.ndarray]): List of silhouette images
            bboxes (List[Tuple]): Optional bounding boxes for each frame
            
        Returns:
            Dict: Quality assessment results with scores and metrics
        """
        if not silhouettes or len(silhouettes) == 0:
            return self._create_quality_result(0.0, "Empty sequence", {})
        
        try:
            # Early stopping for clearly poor sequences
            if len(silhouettes) < self.config['min_sequence_length'] // 2:
                return self._create_quality_result(0.0, "Too short for analysis", {})
            
            # Quick quality check - if too many empty silhouettes, stop early
            non_empty_count = sum(1 for s in silhouettes[:10] if s is not None and s.size > 0 and np.sum(s > 0) > 100)
            if non_empty_count < len(silhouettes[:10]) * 0.3:  # Less than 30% valid silhouettes
                return self._create_quality_result(0.0, "Too few valid silhouettes", {})
            
            # 1. Sequence Length Assessment
            length_score, length_metrics = self._assess_sequence_length(silhouettes)
            
            # Early stopping if length score is too low
            if length_score < 0.2:
                return self._create_quality_result(length_score * 0.5, "Inadequate sequence length", {'length': length_metrics})
            
            # 2. Silhouette Quality Assessment
            silhouette_score, silhouette_metrics = self._assess_silhouette_quality(silhouettes)
            
            # Early stopping if silhouette quality is too low
            if silhouette_score < 0.1:
                return self._create_quality_result(silhouette_score * 0.5, "Poor silhouette quality", {
                    'length': length_metrics,
                    'silhouette': silhouette_metrics
                })
            
            # 3. Motion Quality Assessment
            motion_score, motion_metrics = self._assess_motion_quality(silhouettes)
            
            # 4. Temporal Consistency Assessment
            temporal_score, temporal_metrics = self._assess_temporal_consistency(silhouettes)
            
            # 5. Completeness Assessment
            completeness_score, completeness_metrics = self._assess_completeness(silhouettes)
            
            # 6. Pose Variation Assessment
            pose_score, pose_metrics = self._assess_pose_variation(silhouettes)
            
            # 7. Sharpness Assessment
            sharpness_score, sharpness_metrics = self._assess_sharpness(silhouettes)
            
            # Calculate overall quality score with weighted components optimized for gait
            weights = {
                'length': 0.10,        # Reduced from 0.15 - less important after minimum threshold
                'silhouette': 0.15,    # Reduced from 0.20 - basic requirement
                'motion': 0.25,        # Increased from 0.15 - critical for gait
                'temporal': 0.15,      # Maintained - important for consistency  
                'completeness': 0.15,  # Maintained - important for reliability
                'pose': 0.20,          # Increased from 0.10 - critical for gait analysis
                'sharpness': 0.05      # Reduced from 0.10 - less critical for gait
            }
            
            overall_score = (
                weights['length'] * length_score +
                weights['silhouette'] * silhouette_score +
                weights['motion'] * motion_score +
                weights['temporal'] * temporal_score +
                weights['completeness'] * completeness_score +
                weights['pose'] * pose_score +
                weights['sharpness'] * sharpness_score
            )
            
            # Compile all metrics
            all_metrics = {
                'length': length_metrics,
                'silhouette': silhouette_metrics,
                'motion': motion_metrics,
                'temporal': temporal_metrics,
                'completeness': completeness_metrics,
                'pose': pose_metrics,
                'sharpness': sharpness_metrics,
                'component_scores': {
                    'length': length_score,
                    'silhouette': silhouette_score,
                    'motion': motion_score,
                    'temporal': temporal_score,
                    'completeness': completeness_score,
                    'pose': pose_score,
                    'sharpness': sharpness_score
                }
            }
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)
            
            return self._create_quality_result(overall_score, quality_level, all_metrics)
            
        except Exception as e:
            self.logger.error(f"Error in quality assessment: {str(e)}")
            return self._create_quality_result(0.0, f"Assessment error: {str(e)}", {})
    
    def _assess_sequence_length(self, silhouettes: List[np.ndarray]) -> Tuple[float, Dict]:
        """Assess sequence length quality"""
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
        """Assess individual silhouette quality"""
        valid_silhouettes = 0
        total_area = 0
        total_aspect_ratios = []
        
        for sil in silhouettes:
            if sil is None or sil.size == 0:
                continue
                
            # Calculate silhouette area
            area = np.sum(sil > 0)
            
            if (area >= self.config['min_silhouette_area'] and 
                area <= self.config['max_silhouette_area']):
                
                # Calculate aspect ratio
                contours, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    aspect_ratio = h / w if w > 0 else 0
                    
                    if (aspect_ratio >= self.config['min_aspect_ratio'] and 
                        aspect_ratio <= self.config['max_aspect_ratio']):
                        valid_silhouettes += 1
                        total_area += area
                        total_aspect_ratios.append(aspect_ratio)
        
        if len(silhouettes) == 0:
            return 0.0, {}
        
        quality_ratio = valid_silhouettes / len(silhouettes)
        avg_area = total_area / max(valid_silhouettes, 1)
        avg_aspect_ratio = np.mean(total_aspect_ratios) if total_aspect_ratios else 0
        
        metrics = {
            'valid_silhouettes': valid_silhouettes,
            'total_silhouettes': len(silhouettes),
            'quality_ratio': quality_ratio,
            'average_area': avg_area,
            'average_aspect_ratio': avg_aspect_ratio
        }
        
        return quality_ratio, metrics
    
    def _assess_motion_quality(self, silhouettes: List[np.ndarray]) -> Tuple[float, Dict]:
        """Assess motion quality between frames"""
        if len(silhouettes) < 2:
            return 0.0, {'motion_vectors': []}
        
        motion_vectors = []
        valid_motions = 0
        
        for i in range(1, len(silhouettes)):
            if silhouettes[i-1] is None or silhouettes[i] is None:
                continue
                
            # Calculate frame difference
            diff = cv2.absdiff(silhouettes[i-1].astype(np.float32), 
                              silhouettes[i].astype(np.float32))
            motion_magnitude = np.sum(diff)
            
            motion_vectors.append(motion_magnitude)
            
            if (motion_magnitude >= self.config['min_motion_threshold'] and 
                motion_magnitude <= self.config['max_motion_threshold']):
                valid_motions += 1
        
        if len(motion_vectors) == 0:
            return 0.0, {'motion_vectors': []}
        
        motion_score = valid_motions / len(motion_vectors)
        motion_consistency = 1.0 - (np.std(motion_vectors) / (np.mean(motion_vectors) + 1e-6))
        motion_consistency = max(0.0, min(1.0, motion_consistency))
        
        # Combine motion validity and consistency
        combined_score = (motion_score + motion_consistency) / 2.0
        
        metrics = {
            'motion_vectors': motion_vectors,
            'valid_motions': valid_motions,
            'motion_score': motion_score,
            'motion_consistency': motion_consistency,
            'avg_motion': np.mean(motion_vectors),
            'motion_std': np.std(motion_vectors)
        }
        
        return combined_score, metrics
    
    def _assess_temporal_consistency(self, silhouettes: List[np.ndarray]) -> Tuple[float, Dict]:
        """Assess temporal consistency of silhouettes"""
        if len(silhouettes) < self.config['temporal_consistency_window']:
            return 0.5, {'consistency_scores': []}
        
        window_size = self.config['temporal_consistency_window']
        consistency_scores = []
        
        for i in range(len(silhouettes) - window_size + 1):
            window = silhouettes[i:i + window_size]
            
            # Calculate pairwise similarities within window
            similarities = []
            for j in range(len(window)):
                for k in range(j + 1, len(window)):
                    if window[j] is None or window[k] is None:
                        continue
                    
                    try:
                        # Ensure correct data types
                        img1 = window[j].astype(np.float32) if window[j].dtype != np.float32 else window[j]
                        img2 = window[k].astype(np.float32) if window[k].dtype != np.float32 else window[k]
                        
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
                        
                        # Use template matching with safer parameters
                        if img1.shape[0] >= img2.shape[0] and img1.shape[1] >= img2.shape[1]:
                            result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
                            corr = np.max(result)
                        else:
                            result = cv2.matchTemplate(img2, img1, cv2.TM_CCOEFF_NORMED)
                            corr = np.max(result)
                        
                        similarities.append(max(0.0, corr))
                        
                    except Exception as e:
                        # Fallback: use simple correlation
                        try:
                            img1_flat = window[j].flatten().astype(np.float64)
                            img2_flat = window[k].flatten().astype(np.float64)
                            
                            # Normalize
                            img1_norm = (img1_flat - np.mean(img1_flat)) / (np.std(img1_flat) + 1e-8)
                            img2_norm = (img2_flat - np.mean(img2_flat)) / (np.std(img2_flat) + 1e-8)
                            
                            # Calculate correlation
                            corr = np.corrcoef(img1_norm, img2_norm)[0, 1]
                            similarities.append(max(0.0, corr if not np.isnan(corr) else 0.0))
                        except:
                            similarities.append(0.0)
            
            if similarities:
                consistency_scores.append(np.mean(similarities))
        
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        metrics = {
            'consistency_scores': consistency_scores,
            'overall_consistency': overall_consistency,
            'temporal_windows_analyzed': len(consistency_scores)
        }
        
        return overall_consistency, metrics
    
    def _assess_completeness(self, silhouettes: List[np.ndarray]) -> Tuple[float, Dict]:
        """Assess completeness of silhouettes (no missing parts)"""
        complete_silhouettes = 0
        
        for sil in silhouettes:
            if sil is None:
                continue
                
            # Check if silhouette has reasonable connectivity
            # Use morphological operations to check completeness
            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(sil.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            
            # Calculate ratio of filled area after morphological operations
            original_area = np.sum(sil > 0)
            processed_area = np.sum(opened > 0)
            
            if original_area > 0:
                completeness_ratio = processed_area / original_area
                if completeness_ratio >= self.config['completeness_threshold']:
                    complete_silhouettes += 1
        
        completeness_score = complete_silhouettes / len(silhouettes) if silhouettes else 0.0
        
        metrics = {
            'complete_silhouettes': complete_silhouettes,
            'total_silhouettes': len(silhouettes),
            'completeness_score': completeness_score
        }
        
        return completeness_score, metrics
    
    def _assess_pose_variation(self, silhouettes: List[np.ndarray]) -> Tuple[float, Dict]:
        """Assess pose variation to ensure gait cycle representation"""
        if len(silhouettes) < self.config['gait_cycle_min_frames']:
            return 0.3, {'pose_variations': []}
        
        pose_features = []
        
        for sil in silhouettes:
            if sil is None:
                continue
                
            # Extract simple pose features (moments, contour properties)
            moments = cv2.moments(sil.astype(np.uint8))
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
            return 0.3, {'pose_variations': []}
        
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
    
    def _assess_sharpness(self, silhouettes: List[np.ndarray]) -> Tuple[float, Dict]:
        """Assess sharpness/clarity of silhouettes"""
        sharpness_scores = []
        
        for sil in silhouettes:
            if sil is None:
                continue
                
            # Ensure correct data type for OpenCV operations
            if sil.dtype != np.uint8:
                sil_uint8 = (sil * 255).astype(np.uint8) if sil.max() <= 1.0 else sil.astype(np.uint8)
            else:
                sil_uint8 = sil
                
            try:
                # Calculate Laplacian variance as sharpness measure
                # Use CV_64F explicitly to avoid format issues on Apple Silicon
                laplacian = cv2.Laplacian(sil_uint8, cv2.CV_64F)
                sharpness = laplacian.var()
                sharpness_scores.append(sharpness)
            except cv2.error as e:
                # Fallback: use Sobel operators if Laplacian fails
                try:
                    sobelx = cv2.Sobel(sil_uint8, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(sil_uint8, cv2.CV_64F, 0, 1, ksize=3)
                    sharpness = np.sqrt(sobelx**2 + sobely**2).var()
                    sharpness_scores.append(sharpness)
                except:
                    # Ultimate fallback: use gradient-based measure
                    grad_x = np.gradient(sil_uint8.astype(np.float64), axis=1)
                    grad_y = np.gradient(sil_uint8.astype(np.float64), axis=0)
                    sharpness = (grad_x**2 + grad_y**2).var()
                    sharpness_scores.append(sharpness)
        
        if not sharpness_scores:
            return 0.0, {'sharpness_scores': []}
        
        avg_sharpness = np.mean(sharpness_scores)
        sharpness_score = min(1.0, avg_sharpness / self.config['sharpness_threshold'])
        
        metrics = {
            'sharpness_scores': sharpness_scores,
            'average_sharpness': avg_sharpness,
            'sharpness_score': sharpness_score
        }
        
        return sharpness_score, metrics
    
    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on overall score"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.65:
            return "Good"
        elif score >= 0.5:
            return "Fair"
        elif score >= 0.3:
            return "Poor"
        else:
            return "Very Poor"
    
    def _create_quality_result(self, score: float, level: str, metrics: Dict) -> Dict:
        """Create standardized quality assessment result"""
        import config  # Import config to get thresholds
        
        # Use configurable thresholds
        high_quality_threshold = getattr(config, 'HIGH_QUALITY_THRESHOLD', 0.7)
        acceptable_threshold = 0.5  # Keep this standard
        
        return {
            'overall_score': score,
            'quality_level': level,
            'is_acceptable': score >= acceptable_threshold,
            'is_high_quality': score >= high_quality_threshold,  # Use config threshold
            'metrics': metrics,
            'recommendations': self._generate_recommendations(score, metrics)
        }
    
    def _generate_recommendations(self, score: float, metrics: Dict) -> List[str]:
        """Generate recommendations for improving sequence quality"""
        recommendations = []
        
        if score < 0.5:
            recommendations.append("Sequence quality is below acceptable threshold")
        
        if 'length' in metrics and metrics['length'].get('sequence_length', 0) < self.config['min_sequence_length']:
            recommendations.append("Collect longer gait sequence for better reliability")
        
        if 'motion' in metrics and metrics['motion'].get('motion_consistency', 1.0) < 0.5:
            recommendations.append("Ensure consistent walking speed")
        
        if 'completeness' in metrics and metrics['completeness'].get('completeness_score', 1.0) < 0.7:
            recommendations.append("Improve silhouette extraction to reduce missing parts")
        
        if 'sharpness' in metrics and metrics['sharpness'].get('average_sharpness', 0) < self.config['sharpness_threshold']:
            recommendations.append("Improve video quality or reduce motion blur")
        
        return recommendations

    def is_sequence_acceptable(self, quality_result: Dict) -> bool:
        """Check if sequence meets minimum quality standards"""
        return quality_result.get('is_acceptable', False)
    
    def is_sequence_high_quality(self, quality_result: Dict) -> bool:
        """Check if sequence is high quality"""
        return quality_result.get('is_high_quality', False)
