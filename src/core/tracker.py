"""
Object tracking module using ByteTracker with frame counting
"""

import numpy as np
from collections import defaultdict
from src.utils.bytetracker import BYTETracker

class PersonTracker:
    def __init__(self, config):
        """
        Initialize person tracker
        
        Args:
            config (dict): Tracker configuration parameters
        """
        self.tracker = BYTETracker(
            frame_rate=config['frame_rate'],
            track_thresh=config['track_thresh'], 
            high_thresh=config['high_thresh'], 
            match_thresh=config['match_thresh']
        )
        # Track history: track_id -> frame_count
        self.track_history = defaultdict(int)
        
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections (list): List of detections [x1, y1, x2, y2, confidence]
            
        Returns:
            list: List of tracked objects
        """
        if not detections:
            return []
            
        dets = np.array(detections, dtype=np.float32)
        tracks = self.tracker.update(dets)
        
        # Update track counts
        active_tracks = set()
        for t in tracks:
            track_id = t.track_id
            self.track_history[track_id] += 1
            active_tracks.add(track_id)
            
        # Get frames tracked for each ID
        for t in tracks:
            t.frame_count = self.track_history[t.track_id]
            
        return tracks