import logging
from typing import List, Dict, Any
import numpy as np
from ultralytics import YOLO
from .interfaces import VisionService

logger = logging.getLogger(__name__)

class LocalYoloService(VisionService):
    """Local implementation of VisionService using YOLOv8."""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model_path = model_path
        self.model = None
        self._load_model()
        
    def _load_model(self):
        try:
            logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            raise

    def detect_objects(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        if not self.model:
            raise RuntimeError("Model not loaded")
            
        results = self.model(image, conf=confidence_threshold, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf)
            cls_id = int(box.cls)
            class_name = self.model.names[cls_id]
            
            detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'class_name': class_name,
                'confidence': conf,
                'class_id': cls_id
            })
            
        return detections

    def detect_objects_batch(self, images: List[np.ndarray], confidence_threshold: float = 0.5) -> List[List[Dict[str, Any]]]:
        if not self.model:
            raise RuntimeError("Model not loaded")
            
        # YOLOv8 supports batch inference directly
        results_list = self.model(images, conf=confidence_threshold, verbose=False)
        
        batch_detections = []
        for results in results_list:
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                cls_id = int(box.cls)
                class_name = self.model.names[cls_id]
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'class_name': class_name,
                    'confidence': conf,
                    'class_id': cls_id
                })
            batch_detections.append(detections)
            
        return batch_detections
