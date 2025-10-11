from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple
import torch

class ImageClassifier:
    def __init__(self, model_path: str = "yolo11n.pt"):  # or "yolov11x.pt"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.class_names = self.model.names

        
    def detect_objects(self, image: np.ndarray, 
                      confidence_threshold: float = 0.5) -> List[Dict]:
        """Detect and classify objects in image"""
        results = self.model(image, conf=confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf.cpu().numpy()
                    class_id = int(box.cls.cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': class_name,
                        'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                    }
                    detections.append(detection)
        
        return detections
    
    def draw_detections(self, image: np.ndarray, 
                       detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on image"""
        annotated_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_image, 
                (bbox[0], bbox[1] - label_size[1] - 10),
                (bbox[0] + label_size[0], bbox[1]), 
                (0, 255, 0), -1)
            
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_image, 
                         (bbox, bbox[1] - label_size[1] - 10),
                         (bbox + label_size, bbox[1]), 
                         (0, 255, 0), -1)
            cv2.putText(annotated_image, label, 
                       (bbox, bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 0, 0), 2)
        
        return annotated_image
