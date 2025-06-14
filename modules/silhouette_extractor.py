import numpy as np
import cv2
from collections import defaultdict, deque
from ultralytics import YOLO
from utils.device import get_best_device
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
            
            # More permissive minimum track requirement
            min_required_frames = min(self.min_track_frames, 15)  # Reduced from default 20
            
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
            silhouette = self._segment_person(person_roi)
            
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
        
        # Lower the acceptance threshold to capture more sequences
        if best_score > 0.6 and best_sequence:  # Lowered from 0.7 to 0.6
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
        """Segment person from background using YOLO-Seg"""
        # Run segmentation
        results = self.model(person_roi, classes=0, verbose=False, retina_masks=True)  # Added retina_masks=True
        
        # Create empty mask
        h, w = person_roi.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Process results
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                # Get the mask with highest confidence
                best_mask_idx = 0
                if len(result.boxes) > 1:
                    # If multiple detections, take the one with highest confidence
                    confidences = [box.conf.item() for box in result.boxes]
                    best_mask_idx = np.argmax(confidences)
                
                try:
                    # Extract the binary mask and convert to desired format
                    person_mask = result.masks[best_mask_idx].data.cpu().numpy()
                    
                    # Handle different mask dimensions
                    if len(person_mask.shape) == 3 and person_mask.shape[0] == 1:
                        person_mask = person_mask[0]
                    
                    # Ensure it's a binary mask
                    person_mask = (person_mask > 0.5).astype(np.uint8) * 255
                    
                    # Resize to match original ROI dimensions if needed
                    if person_mask.shape[:2] != (h, w):
                        person_mask = cv2.resize(person_mask, (w, h))
                    
                    mask = person_mask
                    
                except Exception as e:
                    print(f"Error processing mask: {e}")
        
        # Post-process: clean up mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Final check to ensure binary values
        mask = (mask > 127).astype(np.uint8) * 255
        
        return mask
    
    def _evaluate_silhouette_quality(self, silhouette):
        """
        More permissive silhouette quality evaluation for CCTV footage
        
        Returns:
            float: Quality score between 0 and 1
        """
        # Ensure silhouette is a proper 2D binary mask
        if len(silhouette.shape) > 2:
            # Try to extract first channel if it's a multi-channel array
            if silhouette.shape[2] >= 1:
                silhouette = silhouette[:, :, 0]
            else:
                # If we can't extract, create a new empty mask
                silhouette = np.zeros((silhouette.shape[0], silhouette.shape[1]), dtype=np.uint8)
        
        # Ensure binary values (0 or 255)
        silhouette = (silhouette > 127).astype(np.uint8) * 255
        
        # Calculate ratio of foreground pixels
        h, w = silhouette.shape
        fg_pixels = np.count_nonzero(silhouette)
        total_pixels = h * w
        fg_ratio = fg_pixels / total_pixels
        
        # Size-based adaptive threshold - define size_factor
        size_factor = min(1.0, total_pixels / 10000)
        
        # More permissive size check
        if fg_ratio < 0.03 or fg_ratio > 0.97:  # Was 0.05 and 0.95
            return 0.0
        
        # Quality components calculation with more lenient thresholds
        
        # 1. Region completeness
        num_regions = 4
        region_height = h // num_regions
        region_densities = []
        
        for i in range(num_regions):
            y_start = i * region_height
            y_end = (i + 1) * region_height if i < num_regions - 1 else h
            region = silhouette[y_start:y_end, :]
            region_pixels = np.count_nonzero(region)
            region_total = region.shape[0] * region.shape[1]
            region_densities.append(region_pixels / region_total if region_total > 0 else 0)
            
        # Head presence (more lenient)
        head_threshold = max(0.005, 0.02 * size_factor)  # Lower threshold
        head_score = min(1.0, region_densities[0] / head_threshold) * 0.25
        
        # Feet presence (more lenient)
        feet_threshold = max(0.005, 0.02 * size_factor)  # Lower threshold
        feet_score = min(1.0, region_densities[-1] / feet_threshold) * 0.15
        
        # Body completeness (more lenient)
        body_score = min(1.0, min(region_densities[1:-1]) / 0.03) * 0.3  # Lower threshold
        
        # 2. Shape analysis: calculate contour quality
        try:
            contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                
                # Solidity: ratio of contour area to convex hull area
                if hull_area > 0:
                    solidity = contour_area / hull_area
                    # Good human shapes have high solidity
                    contour_score = min(1.0, solidity / 0.7) * 0.2
                else:
                    contour_score = 0
            else:
                contour_score = 0
        except Exception as e:
            print(f"Error in contour analysis: {e}")
            contour_score = 0
        
        # 3. Connectivity (penalize disjoint regions)
        connectivity_score = 0.1
        if len(contours) > 3:  # Too many disconnected parts
            connectivity_score *= (3 / len(contours))
            
        # Combined quality score with weighted components
        quality_score = head_score + feet_score + body_score + contour_score + connectivity_score
        
        # Even if some components are weak, boost the overall score
        if head_score + body_score > 0.4:  # If head and body are decent
            quality_score += 0.1  # Boost the score
            
        # Normalize to 0-1 range
        quality_score = min(1.0, quality_score)
        
        return quality_score