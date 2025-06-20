import numpy as np
import cv2
from collections import defaultdict, deque
from ultralytics import YOLO
from src.utils.device import get_best_device
from .pose_generator import PoseHeatmapGenerator, SimplifiedPoseGenerator
import os
import config

class SilhouetteExtractor:
    def __init__(self, config, model_path="../weights/yolo11m-seg.pt"):
        """
        Initialize silhouette extractor with caching capability
        
        Args:
            config (dict): Silhouette extraction configuration
        """
        self.min_track_frames = config['min_track_frames']
        self.confidence_threshold = config['confidence_threshold']
        self.min_quality_score = config['min_quality_score']
        self.resolution = config['resolution']
        self.window_size = config.get('window_size', 30)  # Default sliding window size
        self.max_cache_size = config.get('max_cache_size', 100)  # Max silhouettes per ID
        
        # Silhouette cache: track_id -> deque of (frame_num, silhouette, quality) tuples
        self.silhouette_cache = defaultdict(lambda: deque(maxlen=self.max_cache_size))
        
        # Previous frame silhouettes for temporal consistency
        self.prev_silhouettes = {}
        
        # Best sequences found so far: track_id -> list of silhouettes
        self.best_sequences = {}
        
        # Current frame number
        self.frame_num = 0
        
        # Load segmentation model
        print("Loading YOLO segmentation model...")
        self.model = YOLO(model_path)
        self.model.to(get_best_device())
        print("YOLO segmentation model loaded")
        
    def extract_silhouettes(self, frame, tracks):
        """
        Extract and cache silhouettes, then find best sequences
        
        Args:
            frame: Input video frame
            tracks: List of tracked persons
            
        Returns:
            dict: Dictionary mapping track_id to best silhouette sequence
        """
        # Increment frame counter
        self.frame_num += 1
        
        # Extract silhouettes for this frame
        for track in tracks:
            track_id = track.track_id
            
            # More permissive minimum track requirement for CCTV scenarios
            min_required_frames = min(self.min_track_frames, 12)  # Reduced further for CCTV
            
            # Skip if track hasn't been visible long enough
            if not hasattr(track, 'frame_count') or track.frame_count < min_required_frames:
                continue
                
            # Extract bounding box with simple uniform borders
            tlwh = track.tlwh
            x, y, w, h = map(int, tlwh)

            # Simple uniform border - 20% on all sides
            border_x = int(w * 0.2)
            border_y = int(h * 0.2)  # Same border percentage for vertical dimension

            # Expand the box uniformly
            x = max(0, x - border_x)
            y = max(0, y - border_y)
            x2 = min(frame.shape[1], x + w + 2*border_x)
            y2 = min(frame.shape[0], y + h + 2*border_y)

            # Skip if box is too small
            if x2 - x < 10 or y2 - y < 10:
                continue
                
            # Extract person ROI with uniform border
            person_roi = frame[y:y2, x:x2]
            
            # Generate silhouette
            silhouette, confidence_score = self._segment_person(person_roi)
            
            # Apply temporal consistency if we have a previous silhouette
            if track_id in self.prev_silhouettes:
                silhouette = self._apply_temporal_consistency(silhouette, self.prev_silhouettes[track_id])
            
            # Store for next frame
            self.prev_silhouettes[track_id] = silhouette.copy()
            
            # Evaluate silhouette quality
            quality_score = self._evaluate_silhouette_quality(silhouette)
            
            # Cache silhouette regardless of quality (for sequence analysis)
            self.silhouette_cache[track_id].append((
                self.frame_num, 
                cv2.resize(silhouette, (self.resolution[1], self.resolution[0])),
                quality_score
            ))
            
            # If we have enough silhouettes, analyze the sequences
            if len(self.silhouette_cache[track_id]) >= self.window_size:
                self._analyze_sequences(track_id)
        
        # Return the current best sequences (might be empty if none good enough yet)
        return self.best_sequences
        
    def _analyze_sequences(self, track_id):
        """
        Analyze the cached silhouettes to find the best continuous sequence
        
        Args:
            track_id: ID of the track to analyze
        """
        cache = self.silhouette_cache[track_id]
        
        # Need at least window_size samples for analysis
        if len(cache) < self.window_size:
            return
            
        # Calculate dynamic quality threshold based on the whole cache
        qualities = [q for _, _, q in cache]
        
        # More permissive dynamic threshold - use lower percentile instead of mean
        dynamic_threshold = max(
            self.min_quality_score * 0.8,  # Lower the minimum threshold by 20%
            np.percentile(qualities, 60)  # Use 60th percentile instead of mean
        )
        
        best_score = 0
        best_sequence = None
        best_start_idx = 0
        
        # Sliding window through the cache
        for start_idx in range(len(cache) - self.window_size + 1):
            # Extract the window
            window = [cache[start_idx + i] for i in range(self.window_size)]
            
            # Check temporal continuity (frames should be sequential)
            frame_nums = [f for f, _, _ in window]
            if frame_nums[-1] - frame_nums[0] > self.window_size * 3:  # More permissive gap threshold
                # Too many gaps in the sequence
                continue
                
            # Calculate quality metrics for this sequence
            qualities = [q for _, _, q in window]
            avg_quality = np.mean(qualities)
            min_quality = np.min(qualities)
            
            # Calculate completeness (how many silhouettes meet the dynamic threshold)
            completeness = sum(1 for q in qualities if q >= dynamic_threshold) / len(qualities)
            
            # More balanced scoring between quality and completeness
            sequence_score = (avg_quality * 0.6) + (completeness * 0.4)
            if min_quality < 0.2:  # Lower penalty threshold
                sequence_score *= 0.9  # Less severe penalty
                
            # Update best sequence if this one is better
            if sequence_score > best_score:
                best_score = sequence_score
                best_sequence = [silhouette for _, silhouette, _ in window]
                best_start_idx = start_idx
        
        # Lower the acceptance threshold to capture more sequences for CCTV
        if best_score > 0.45 and best_sequence:  # Lowered from 0.6 to 0.45 for CCTV scenarios
            # print(f"Found good silhouette sequence for track {track_id} with score {best_score:.2f}")
            self.best_sequences[track_id] = best_sequence

    def get_best_silhouette_sequence(self, track_id):
        """
        Get the best silhouette sequence for a given track ID
        
        Args:
            track_id: Track ID to retrieve
            
        Returns:
            list: List of silhouettes or None if not found
        """
        return self.best_sequences.get(track_id)
    
    def _segment_person(self, person_roi):
        """
        Segment person from background using more robust approach with confidence scoring
        instead of binary decisions and with standardized error handling.
        """
        # Create empty mask with confidence score
        h, w = person_roi.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        confidence_score = 0.0
        
        try:
            # Run segmentation with multiple fallback options
            results = self.model(person_roi, classes=0, verbose=False, retina_masks=True)
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                    # Score masks by confidence and size
                    best_mask_idx = 0
                    best_mask_score = 0
                    
                    for idx, box in enumerate(result.boxes):
                        if idx >= len(result.masks):
                            continue
                            
                        # Calculate score based on confidence and position
                        conf = box.conf.item()
                        # Prefer centered objects (likely to be full person)
                        box_data = box.xyxy.cpu().numpy()[0]
                        box_center_x = (box_data[0] + box_data[2]) / 2
                        box_center_y = (box_data[1] + box_data[3]) / 2
                        center_dist = np.sqrt((box_center_x/w - 0.5)**2 + (box_center_y/h - 0.5)**2)
                        position_score = 1.0 - min(1.0, 2 * center_dist)
                        
                        # Prefer larger masks (more likely complete person)
                        mask_area = result.masks[idx].data.sum().item() / (h * w)
                        area_score = min(1.0, 4 * mask_area) if mask_area < 0.25 else 1.0
                        
                        # Combined score
                        mask_score = conf * 0.6 + position_score * 0.2 + area_score * 0.2
                        
                        if mask_score > best_mask_score:
                            best_mask_score = mask_score
                            best_mask_idx = idx
                    
                    # Extract the binary mask with confidence information
                    try:
                        person_mask = result.masks[best_mask_idx].data.cpu().numpy()
                        
                        if len(person_mask.shape) == 3 and person_mask.shape[0] == 1:
                            person_mask = person_mask[0]
                        
                        # Convert to probability mask (keep values between 0-1)
                        person_mask_prob = person_mask.copy()
                        
                        # Use model confidence and mask consistency for overall confidence
                        mask_consistency = 1.0 - np.std(person_mask_prob)  # Higher consistency = lower std
                        confidence_score = best_mask_score * 0.7 + mask_consistency * 0.3
                        
                        # Create binary mask for processing
                        person_mask = (person_mask > 0.5).astype(np.uint8) * 255
                        
                        # Resize to match original ROI dimensions if needed
                        if person_mask.shape[:2] != (h, w):
                            person_mask = cv2.resize(person_mask, (w, h))
                        
                        mask = person_mask
                        
                    except Exception as e:
                        # Standardized error handling with logging
                        confidence_score *= 0.5  # Reduce confidence due to error
                        if hasattr(self, 'logger'):
                            self.logger.warning(f"Mask conversion error: {str(e)}")
        
        except Exception as e:
            # Global exception handler with informative error
            if hasattr(self, 'logger'):
                self.logger.error(f"Segmentation error: {str(e)}")
            confidence_score = 0.0
        
        # Post-processing with adaptive parameters based on confidence
        if mask.max() > 0:
            # Adaptive kernel size: more aggressive cleaning when confidence is low
            kernel_size = 3 if confidence_score > 0.7 else 5
            small_kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Fill small holes first
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small_kernel)
            
            # Remove small noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
            
            # Apply edge preservation using bilateral filter
            mask_float = mask.astype(np.float32) / 255.0
            # Use confidence score to determine filter parameters
            d = 5
            sigmaColor = 50 if confidence_score > 0.6 else 30
            sigmaSpace = 50 if confidence_score > 0.6 else 30
            mask_smooth = cv2.bilateralFilter(mask_float, d, sigmaColor, sigmaSpace)
            mask = (mask_smooth * 255).astype(np.uint8)
            
            # Final check to ensure binary values
            mask = (mask > 127).astype(np.uint8) * 255
            
            # Assess contour quality and adjust confidence
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Instead of hard cutoff on contour count, use confidence scaling
            if contours:
                # More contours = less confidence, but scale gradually
                contour_confidence = 1.0 / (1.0 + 0.2 * max(0, len(contours) - 1))
                # Weight contour confidence less as overall confidence increases
                confidence_score = confidence_score * 0.8 + contour_confidence * 0.2
            else:
                confidence_score = 0.0  # No contours = no confidence
        
        return mask , confidence_score

    def _evaluate_silhouette_quality(self, silhouette):
        """
        Advanced silhouette quality evaluation using adaptive thresholds
        
        Returns:
            float: Quality score between 0 and 1
        """
        # Ensure silhouette is a proper 2D binary mask
        if len(silhouette.shape) > 2:
            silhouette = silhouette[:, :, 0] if silhouette.shape[2] >= 1 else np.zeros((silhouette.shape[0], silhouette.shape[1]), dtype=np.uint8)
        
        # Ensure binary values (0 or 255)
        silhouette = (silhouette > 127).astype(np.uint8) * 255
        
        # Calculate basic properties
        h, w = silhouette.shape
        fg_pixels = np.count_nonzero(silhouette)
        total_pixels = h * w
        fg_ratio = fg_pixels / total_pixels
        
        # Reject extreme cases (too empty or too full)
        if fg_ratio < 0.02 or fg_ratio > 0.98:
            return 0.0
            
        # Get silhouette height/width based on non-zero columns/rows
        # (more adaptive to actual figure than using full image dimensions)
        col_sums = np.sum(silhouette > 0, axis=0)
        row_sums = np.sum(silhouette > 0, axis=1)
        non_zero_cols = np.where(col_sums > 0)[0]
        non_zero_rows = np.where(row_sums > 0)[0]
        
        if len(non_zero_cols) == 0 or len(non_zero_rows) == 0:
            return 0.0
            
        silhouette_width = non_zero_cols[-1] - non_zero_cols[0] + 1
        silhouette_height = non_zero_rows[-1] - non_zero_rows[0] + 1
        
        # Calculate aspect ratio (height/width) - human figures typically have aspect ratio > 1.5
        aspect_ratio = silhouette_height / max(1, silhouette_width)
        aspect_score = min(1.0, aspect_ratio / 1.5) * 0.15
        
        # Divide into anatomical regions using relative proportions instead of equal divisions
        # Human body proportions: head ~1/8, torso ~3/8, legs ~4/8 of height
        head_region = silhouette[0:int(silhouette_height*0.2), :]
        torso_region = silhouette[int(silhouette_height*0.2):int(silhouette_height*0.55), :]
        legs_region = silhouette[int(silhouette_height*0.55):, :]
        
        # Calculate region densities relative to expected proportions
        head_density = np.count_nonzero(head_region) / max(1, head_region.size)
        torso_density = np.count_nonzero(torso_region) / max(1, torso_region.size)
        legs_density = np.count_nonzero(legs_region) / max(1, legs_region.size)
        
        # Adaptive thresholds based on overall silhouette density
        # If overall density is low, we should expect lower density in each region
        base_density = fg_ratio * 0.7  # Base expectation
        
        # Score each region relative to the silhouette's overall characteristics
        head_expected = base_density * 0.8  # Head typically less dense than body
        head_score = min(1.0, head_density / max(0.01, head_expected)) * 0.25
        
        torso_expected = base_density * 1.2  # Torso typically more dense
        torso_score = min(1.0, torso_density / max(0.01, torso_expected)) * 0.3
        
        legs_expected = base_density * 0.9  # Legs typically less dense than torso
        legs_score = min(1.0, legs_density / max(0.01, legs_expected)) * 0.2
        
        # Balance analysis - check if silhouette is centered
        col_center = np.sum(np.arange(w) * col_sums) / max(1, np.sum(col_sums))
        balance_score = min(1.0, 1 - abs(col_center/w - 0.5) * 2) * 0.1
        
        # Contour analysis - more sophisticated than before
        try:
            contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # Calculate circularity (4*pi*area/perimeter²) - human silhouettes aren't circular
                circularity = (4 * np.pi * contour_area) / max(perimeter * perimeter, 1)
                # Good human silhouettes have moderate circularity (not too circular, not too irregular)
                contour_score = min(1.0, 1 - abs(circularity - 0.3) / 0.3) * 0.15
                
                # Connectivity score - reduce for multiple disconnected regions
                connectivity_factor = min(1.0, 3 / max(1, len(contours)))
                connectivity_score = 0.1 * connectivity_factor
            else:
                contour_score = 0
                connectivity_score = 0
        except Exception:
            contour_score = 0
            connectivity_score = 0
        
        # Combined quality score with weighted components
        quality_score = head_score + torso_score + legs_score + aspect_score + contour_score + balance_score + connectivity_score
        
        # Normalize to 0-1 range
        quality_score = min(1.0, quality_score)
        
        # Store quality metrics for potential future learning
        self._update_quality_statistics(quality_score, head_density, torso_density, legs_density)
        
        return quality_score

    def _update_quality_statistics(self, quality_score, head_density, torso_density, legs_density):
        """
        Store quality statistics for adaptive threshold learning
        
        This method lays groundwork for ML-based quality assessment by collecting statistics
        """
        # Initialize statistics containers if needed
        if not hasattr(self, 'quality_stats'):
            self.quality_stats = {
                'samples': 0,
                'quality_scores': [],
                'head_densities': [],
                'torso_densities': [],
                'legs_densities': [],
            }
        
        # Store statistics (limit to last 1000 samples)
        if self.quality_stats['samples'] < 1000:
            self.quality_stats['quality_scores'].append(quality_score)
            self.quality_stats['head_densities'].append(head_density)
            self.quality_stats['torso_densities'].append(torso_density)
            self.quality_stats['legs_densities'].append(legs_density)
        else:
            # Replace oldest entry
            idx = self.quality_stats['samples'] % 1000
            self.quality_stats['quality_scores'][idx] = quality_score
            self.quality_stats['head_densities'][idx] = head_density
            self.quality_stats['torso_densities'][idx] = torso_density
            self.quality_stats['legs_densities'][idx] = legs_density
        
        self.quality_stats['samples'] += 1
    
    def _apply_temporal_consistency(self, current_silhouette, prev_silhouette):
        """
        Apply temporal consistency to reduce flickering and maintain gait details
        
        Args:
            current_silhouette: Current frame silhouette
            prev_silhouette: Previous frame silhouette
            
        Returns:
            np.ndarray: Temporally consistent silhouette
        """
        if prev_silhouette is None or current_silhouette.shape != prev_silhouette.shape:
            return current_silhouette
            
        # Calculate similarity between current and previous silhouettes
        intersection = cv2.bitwise_and(current_silhouette, prev_silhouette)
        union = cv2.bitwise_or(current_silhouette, prev_silhouette)
        
        intersection_area = np.sum(intersection > 0)
        union_area = np.sum(union > 0)
        
        if union_area == 0:
            return current_silhouette
            
        iou = intersection_area / union_area
        
        # If silhouettes are very similar, apply weighted blending
        if iou > 0.7:
            # Weight towards current frame but maintain some temporal stability
            alpha = 0.8
            blended = cv2.addWeighted(current_silhouette.astype(np.float32), alpha,
                                    prev_silhouette.astype(np.float32), 1-alpha, 0)
            return (blended > 127).astype(np.uint8) * 255
        else:
            # If silhouettes are very different, use current frame
            return current_silhouette