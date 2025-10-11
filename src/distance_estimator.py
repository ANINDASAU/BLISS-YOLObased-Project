import cv2
import numpy as np
from typing import Dict, List


class DistanceEstimator:
    def __init__(self):
        """Initialize distance estimator with calibration parameters."""
        self.focal_length = 615  # pixels; calibrate for your camera

        # Real-world widths in cm for classes you expect to detect
        self.known_widths = {
            'person': 60,
            'car': 180,
            'truck': 250,
            'bus': 250,
            'bicycle': 60,
            'motorbike': 80,
            'pothole': 50,  # approximate; adjust for your dataset
        }

        # Aliases mapping from model labels to our canonical class names
        self.class_aliases = {
            'motorcycle': 'motorbike',
            'motorbike': 'motorbike',
            'bicycle': 'bicycle',
            'car': 'car',
            'truck': 'truck',
            'bus': 'bus',
            'person': 'person',
        }

    def calibrate_focal_length(self, known_distance_m: float, known_width_cm: float, pixel_width: float) -> float:
        known_distance_cm = known_distance_m * 100.0
        self.focal_length = (pixel_width * known_distance_cm) / known_width_cm
        return self.focal_length

    def _normalize_class(self, class_name: str) -> str:
        return self.class_aliases.get(class_name.lower(), class_name.lower())

    def estimate_distance_cm(self, object_class: str, pixel_width: float) -> float:
        cls = self._normalize_class(object_class)
        if pixel_width <= 0:
            return -1
        if cls not in self.known_widths:
            return -1
        real_width_cm = self.known_widths[cls]
        distance_cm = (real_width_cm * self.focal_length) / pixel_width
        return float(distance_cm)

    def add_distance_to_detections(self, detections: List[Dict]) -> List[Dict]:
        for detection in detections:
            bbox = detection.get('bbox', [0, 0, 0, 0])
            pixel_width = max(0, bbox[2] - bbox[0])
            object_class = detection.get('class_name', '')

            distance_cm = self.estimate_distance_cm(object_class, pixel_width)
            if distance_cm > 0:
                distance_m = distance_cm / 100.0
                detection['distance'] = distance_m
                detection['distance_display'] = f"{distance_m:.2f}m"
            else:
                detection['distance'] = -1
                detection['distance_display'] = "Unknown"

        return detections
