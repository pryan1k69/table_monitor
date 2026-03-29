import time
import numpy as np
from ultralytics import YOLO
from typing import Tuple

class PersonDetector:
    """Класс для детекции людей"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.25):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.detection_times = []
        
    def detect_people_in_roi(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        """Детекция людей в области столика (в оригинальных координатах)"""
        start_time = time.time()
        
        x, y, w, h = roi
        
        # Проверка границ
        if y + h > frame.shape[0] or x + w > frame.shape[1]:
            return False
        
        # Вырезаем область столика
        roi_frame = frame[y:y+h, x:x+w]
        
        if roi_frame.size == 0:
            return False
        
        # Детекция YOLO
        results = self.model(roi_frame, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 0 and conf > self.confidence_threshold:
                        detection_time = time.time() - start_time
                        self.detection_times.append(detection_time)
                        return True
        return False
    
    def get_avg_detection_time(self) -> float:
        if self.detection_times:
            return np.mean(self.detection_times)
        return 0.0