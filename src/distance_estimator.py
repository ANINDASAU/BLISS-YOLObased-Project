from turtle import distance
import cv2
import numpy as np
from typing import Dict, List

class DistanceEstimator:
    def __init__(self):
        """Initialize distance estimator with calibration parameters"""
        # These values need to be calibrated for your specific camera
        # and object types. These are example values.
        self.focal_length = 615  # Focal length in pixels (needs calibration)
        
        # Known real-world widths for common objects (in cm)
        self.known_widths = {
            'person': 60,  # Average person width
            'car': 180,    # Average car width
            'bottle': 7,   # Standard bottle width
            'cup': 8,      # Standard cup width
            'laptop': 35,  # Standard laptop width
            'cell phone': 7, # Standard phone width
        }
    
    def calibrate_focal_length(self, known_distance: float, 
                              known_width: float, 
                              pixel_width: float) -> float:
        """Calibrate focal length using reference measurements"""
        self.focal_length = (pixel_width * known_distance) / known_width
        return self.focal_length
    
    def estimate_distance(self, object_class: str, pixel_width: float) -> float:
        if object_class not in self.known_widths:
        # Assign a default or approximate width to unknown objects, or skip
        # Optionally, log or warn
            return -1
    
    # Prevent division by zero or too small widths
        if pixel_width <= 0:
            return -1
    
        real_width = self.known_widths[object_class]
        distance = (real_width * self.focal_length) / pixel_width
        return distance

    
    def add_distance_to_detections(self, detections: List[Dict]) -> List[Dict]:
        """Add distance estimates to detection results"""
        for detection in detections:
            bbox = detection['bbox']
            pixel_width = bbox[2] - bbox[0]  # Width in pixels
            object_class = detection['class_name']
            
            distance = self.estimate_distance(object_class, pixel_width)
            detection['distance'] = distance
            
            # Convert to meters for display
            if distance > 0:
                distance_m = distance / 100  # Convert cm to meters
                detection['distance_display'] = f"{distance_m:.1f}m"
            else:
                detection['distance_display'] = "Unknown"
        
        return detections
