"""
Pose estimation module for SkeletonGait++ model
This module generates pose heatmaps from silhouettes for multimodal gait recognition
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

class PoseHeatmapGenerator:
    """
    Generates pose heatmaps from silhouettes for SkeletonGait++ model
    Since we don't have a full pose estimation pipeline, we'll generate
    simplified pose heatmaps based on silhouette analysis
    """
    
    def __init__(self, target_size=(64, 44), sigma=2.0):
        self.target_size = target_size
        self.sigma = sigma
        
    def generate_pose_from_silhouette(self, silhouette):
        """
        Generate simplified pose heatmap from silhouette
        
        Args:
            silhouette: numpy array of shape (H, W) with binary silhouette
            
        Returns:
            pose_heatmap: numpy array of shape (2, H, W) containing pose heatmaps
        """
        if len(silhouette.shape) == 3:
            silhouette = silhouette.squeeze()
            
        h, w = silhouette.shape
        
        # Initialize pose heatmaps (2 channels for simplified pose representation)
        pose_heatmap = np.zeros((2, h, w), dtype=np.float32)
        
        # Find contours to extract key points
        contours, _ = cv2.findContours(
            silhouette.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return pose_heatmap
            
        # Get the largest contour (main person silhouette)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Extract simple keypoints from silhouette
        keypoints = self._extract_keypoints_from_contour(main_contour, silhouette)
        
        # Generate heatmaps for keypoints with higher intensity
        for i, (x, y) in enumerate(keypoints):
            if x >= 0 and y >= 0:  # Valid keypoint
                channel = i % 2  # Distribute keypoints across 2 channels
                pose_heatmap[channel] = self._add_gaussian_at_point(
                    pose_heatmap[channel], x, y, self.sigma, intensity=0.8  # Increased intensity
                )
        
        return pose_heatmap
    
    def _extract_keypoints_from_contour(self, contour, silhouette):
        """
        Extract simplified keypoints from silhouette contour
        """
        h, w = silhouette.shape
        
        # Get bounding box
        x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
        
        keypoints = []
        
        # Top point (head approximation)
        top_y = y
        head_x = x + bbox_w // 2
        keypoints.append((head_x, top_y))
        
        # Center point (torso approximation)
        center_x = x + bbox_w // 2
        center_y = y + bbox_h // 3
        keypoints.append((center_x, center_y))
        
        # Bottom points (feet approximation)
        bottom_y = y + bbox_h
        left_foot_x = x + bbox_w // 3
        right_foot_x = x + 2 * bbox_w // 3
        keypoints.append((left_foot_x, bottom_y))
        keypoints.append((right_foot_x, bottom_y))
        
        # Side points (arms approximation)
        mid_y = y + bbox_h // 2
        left_arm_x = x
        right_arm_x = x + bbox_w
        keypoints.append((left_arm_x, mid_y))
        keypoints.append((right_arm_x, mid_y))
        
        return keypoints
    
    def _add_gaussian_at_point(self, heatmap, x, y, sigma, intensity=0.8):
        """
        Add a Gaussian blob at the specified point with configurable intensity
        """
        h, w = heatmap.shape
        
        # Create coordinate grids
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        
        # Calculate Gaussian with higher intensity
        gaussian = intensity * np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
        
        # Add to heatmap (use maximum to avoid overwriting)
        heatmap = np.maximum(heatmap, gaussian)
        
        return heatmap
    
    def generate_pose_sequence(self, silhouette_sequence):
        """
        Generate pose heatmaps for a sequence of silhouettes
        
        Args:
            silhouette_sequence: list of numpy arrays, each of shape (H, W)
            
        Returns:
            pose_sequence: numpy array of shape (T, 2, H, W)
        """
        pose_sequence = []
        
        for silhouette in silhouette_sequence:
            pose_heatmap = self.generate_pose_from_silhouette(silhouette)
            pose_sequence.append(pose_heatmap)
            
        return np.array(pose_sequence)
    
    def create_multimodal_input(self, silhouette_sequence):
        """
        Create multimodal input (pose + silhouette) for SkeletonGait++
        
        Args:
            silhouette_sequence: list of numpy arrays, each of shape (H, W)
            
        Returns:
            multimodal_input: numpy array of shape (T, 3, H, W)
                            where channels are [pose_x, pose_y, silhouette]
        """
        # Generate pose heatmaps
        pose_sequence = self.generate_pose_sequence(silhouette_sequence)
        
        # Prepare silhouette sequence
        silhouette_array = np.array(silhouette_sequence)
        if len(silhouette_array.shape) == 3:
            silhouette_array = silhouette_array[:, np.newaxis, :, :]  # Add channel dim
        elif len(silhouette_array.shape) == 4 and silhouette_array.shape[1] == 1:
            silhouette_array = silhouette_array.squeeze(1)  # Remove single channel
            silhouette_array = silhouette_array[:, np.newaxis, :, :]  # Add back
        
        # Combine pose and silhouette
        multimodal_input = np.concatenate([
            pose_sequence,  # Shape: (T, 2, H, W)
            silhouette_array  # Shape: (T, 1, H, W)
        ], axis=1)  # Result: (T, 3, H, W)
        
        return multimodal_input


class SimplifiedPoseGenerator:
    """
    Simplified pose generator that creates dummy pose heatmaps
    for cases where pose estimation is not available
    """
    
    def __init__(self, target_size=(64, 44)):
        self.target_size = target_size
        
    def generate_dummy_pose(self, silhouette):
        """
        Generate dummy pose heatmaps based on silhouette center of mass
        """
        if len(silhouette.shape) == 3:
            silhouette = silhouette.squeeze()
            
        h, w = silhouette.shape
        
        # Find center of mass
        y_coords, x_coords = np.where(silhouette > 0)
        if len(x_coords) == 0:
            # Empty silhouette, return zeros
            return np.zeros((2, h, w), dtype=np.float32)
            
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        # Create simple pose representation
        pose_heatmap = np.zeros((2, h, w), dtype=np.float32)
        
        # Channel 0: vertical center line
        if 0 <= center_x < w:
            pose_heatmap[0, :, center_x] = 1.0
            
        # Channel 1: horizontal center line  
        if 0 <= center_y < h:
            pose_heatmap[1, center_y, :] = 1.0
            
        # Apply Gaussian smoothing
        pose_heatmap[0] = gaussian_filter(pose_heatmap[0], sigma=1.0)
        pose_heatmap[1] = gaussian_filter(pose_heatmap[1], sigma=1.0)
        
        return pose_heatmap
    
    def create_multimodal_input(self, silhouette_sequence):
        """
        Create simplified multimodal input
        """
        multimodal_sequence = []
        
        for silhouette in silhouette_sequence:
            pose_heatmap = self.generate_dummy_pose(silhouette)
            
            # Ensure silhouette has correct shape
            if len(silhouette.shape) == 2:
                silhouette = silhouette[np.newaxis, :, :]  # Add channel dim
                
            # Combine pose and silhouette
            multimodal_frame = np.concatenate([
                pose_heatmap,  # Shape: (2, H, W)
                silhouette     # Shape: (1, H, W)
            ], axis=0)  # Result: (3, H, W)
            
            multimodal_sequence.append(multimodal_frame)
            
        return np.array(multimodal_sequence)  # Shape: (T, 3, H, W)
