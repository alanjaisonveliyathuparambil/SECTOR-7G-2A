import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

class FaceDetector:
    def __init__(self, scale_factor: float = 1.3):
        self.scale_factor = scale_factor
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.alt_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )

    def detect_faces(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in an image and returns bounding boxes and padded face crops.
        """
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale detection
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
        )
        
        if len(faces) == 0 and not self.alt_cascade.empty():
            faces = self.alt_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50)
            )

        results = []
        if len(faces) == 0:
            # Fallback: analyze central 70% as single focus area if no face isolated
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.15)
            bw = w - 2 * margin_x
            bh = h - 2 * margin_y
            crop = image_bgr[margin_y:margin_y+bh, margin_x:margin_x+bw]
            results.append({
                "bbox": [margin_x, margin_y, bw, bh],
                "padded_bbox": [0, 0, w, h],
                "crop": crop if crop.size > 0 else image_bgr,
                "is_fallback": True
            })
            return results

        for (x, y, bw, bh) in faces:
            # Pad bounding box by scale_factor (standard DFDC/FF++ practice)
            center_x, center_y = x + bw // 2, y + bh // 2
            size = int(max(bw, bh) * self.scale_factor)
            
            x1 = max(0, center_x - size // 2)
            y1 = max(0, center_y - size // 2)
            x2 = min(w, x1 + size)
            y2 = min(h, y1 + size)
            
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                results.append({
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "padded_bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "crop": crop,
                    "is_fallback": False
                })

        return results
