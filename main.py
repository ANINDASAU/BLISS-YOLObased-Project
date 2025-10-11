import cv2
import sys
from src.camera_handler import CameraHandler
from src.image_classifier import ImageClassifier
from src.distance_estimator import DistanceEstimator
from src.alert import start_alarm, stop_alarm

# monitored classes and threshold (meters)
VEHICLE_MODEL_CLASSES = {'car', 'truck', 'bus', 'motorbike', 'motorcycle', 'bicycle', 'vehicle'}
ALLOWED_MODEL_CLASSES = VEHICLE_MODEL_CLASSES.union({'person', 'pothole'})

# For display/alarm purposes collapse vehicle family into a single 'vehicle' label
DISPLAY_TARGETS = {'person', 'vehicle', 'pothole'}

ALARM_CLASSES = DISPLAY_TARGETS
ALARM_DISTANCE_M = 2.0

class RealTimeImageClassifier:
    def __init__(self):
        """Initialize the real-time image classifier system"""
        try:
            self.camera = CameraHandler(camera_index=0)
            self.classifier = ImageClassifier()
            self.distance_estimator = DistanceEstimator()
            print("System initialized successfully!")
        except Exception as e:
            print(f"Error initializing system: {e}")
            sys.exit(1)
    
    def run(self):
        """Run the real-time classification system"""
        print("Starting real-time object detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        while True:
            # Capture frame from camera
            ret, frame = self.camera.capture_frame()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Detect objects in frame
            detections = self.classifier.detect_objects(frame)

            # Filter detections to only allowed classes (person, vehicle family, pothole)
            filtered = []
            for d in detections:
                model_cls = d.get('class_name', '').lower()
                if model_cls not in ALLOWED_MODEL_CLASSES:
                    continue

                # Keep original class_name for distance estimation, but add a display_class
                if model_cls in VEHICLE_MODEL_CLASSES:
                    d['display_class'] = 'vehicle'
                elif model_cls == 'person':
                    d['display_class'] = 'person'
                elif model_cls == 'pothole':
                    d['display_class'] = 'pothole'
                else:
                    # fallback to original
                    d['display_class'] = model_cls

                filtered.append(d)

            detections = filtered
            
            # Add distance estimation
            detections = self.distance_estimator.add_distance_to_detections(detections)

            # Alarm logic: start if any monitored class is within threshold
            threat_close = False
            for d in detections:
                # use display_class for alarm decisions (person/vehicle/pothole)
                disp = d.get('display_class', d.get('class_name', '')).lower()
                dist = d.get('distance', -1)
                if disp in ALARM_CLASSES and dist > 0 and dist <= ALARM_DISTANCE_M:
                    threat_close = True
                    break

            if threat_close:
                start_alarm()
            else:
                stop_alarm()
            
            # Draw detections with distance info
            annotated_frame = self.draw_detections_with_distance(frame, detections)
            
            # Display frame
            cv2.imshow('Real-Time Object Detection', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('screenshot.jpg', annotated_frame)
                print("Screenshot saved!")
    
    def draw_detections_with_distance(self, image, detections):
        """Draw detections with distance information"""
        annotated_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            # use display_class for overlay (shows 'vehicle' instead of 'car')
            class_name = detection.get('display_class', detection.get('class_name', ''))
            confidence = detection['confidence']
            distance_display = detection.get('distance_display', 'Unknown')

            # Label text including distance
            label = f"{class_name}: {confidence:.2f} - {distance_display}"
            
            # Calculate label size for background rectangle
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw rectangle for label background
            cv2.rectangle(annotated_image,
                        (bbox[0], bbox[1] - label_size[1] - 10),
                        (bbox[0] + label_size[0], bbox[1]),
                        (0, 255, 0), -1)
            
            # Draw the label text
            cv2.putText(annotated_image, label,
                        (bbox[0], bbox[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2)
        return annotated_image
    
    def cleanup(self):
        """Clean up resources"""
        self.camera.release()

if __name__ == "__main__":
    classifier = RealTimeImageClassifier()
    try:
        classifier.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        classifier.cleanup()
