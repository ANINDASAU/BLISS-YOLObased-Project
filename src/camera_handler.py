import cv2
import numpy as np
from typing import Tuple, Optional

class CameraHandler:
    def __init__(self, camera_index: int = 0):
        """Initialize camera with specified index"""
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera with index {camera_index}")
        
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
    
    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture a single frame from camera"""
        ret, frame = self.cap.read()
        return ret, frame
    
    def start_stream(self):
        """Start continuous video stream"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Display frame
            cv2.imshow('Camera Stream', frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def release(self):
        """Release camera resources"""
        self.cap.release()
        cv2.destroyAllWindows()
