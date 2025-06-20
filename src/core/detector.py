"""
Object detection module using YOLO
"""

import numpy as np
from ultralytics import YOLO
from src.utils.device import get_best_device

class PersonDetector:
    def __init__(self, model_path):
        """
        Initialize YOLO person detector
        
        Args:
            model_path (str): Path to YOLO model weights
        """
        self.model = YOLO(model_path)
        self.model.to(get_best_device())
        
    def detect(self, frame):
        """
        Detect persons in a frame
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: List of detections in format [x1, y1, x2, y2, confidence]
        """
        results = self.model(frame,verbose=False)
        detections = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    if box.cls == 0:  # Class 0 (person)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        detections.append([x1, y1, x2, y2, conf])
        
        return detections