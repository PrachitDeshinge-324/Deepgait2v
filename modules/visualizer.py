"""
Visualization module for displaying detection and tracking results
"""

import cv2

class Visualizer:
    def __init__(self, config):
        """
        Initialize visualizer
        
        Args:
            config (dict): Visualization configuration parameters
        """
        self.box_color = config['box_color']
        self.text_color = config['text_color']
        self.fps_color = config['fps_color']
        self.line_thickness = config['line_thickness']
        self.font_scale = config['font_scale']
        self.fps_font_scale = config['fps_font_scale']
        
    def draw_tracks(self, frame, tracks):
        """
        Draw tracking results on frame
        
        Args:
            frame (numpy.ndarray): Input frame
            tracks (list): List of tracking results
            
        Returns:
            numpy.ndarray: Frame with tracks drawn
        """
        for t in tracks:
            tlwh = t.tlwh
            track_id = t.track_id
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                         self.box_color, self.line_thickness)
            cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_color, 
                       self.line_thickness)
        
        return frame
        
    def draw_fps(self, frame, fps):
        """
        Draw FPS information on frame
        
        Args:
            frame (numpy.ndarray): Input frame
            fps (float): Frames per second
            
        Returns:
            numpy.ndarray: Frame with FPS drawn
        """
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.fps_font_scale, 
                   self.fps_color, self.line_thickness)
        
        return frame