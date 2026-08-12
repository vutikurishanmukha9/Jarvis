"""
Constants for the Multimodal AI System.

This module contains all constant values used throughout the application.
Centralizing constants improves maintainability and reduces magic numbers.
"""

from typing import Dict, List, Tuple

# Version Information
VERSION = "2.0.0"
VERSION_INFO = {
    "major": 2,
    "minor": 0,
    "patch": 0,
    "release": "stable"
}

# Image Processing Constants
class ImageConstants:
    """Constants related to image processing."""
    
    # Supported formats
    SUPPORTED_FORMATS: Tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"
    )
    
    # Default dimensions
    DEFAULT_WIDTH: int = 640
    DEFAULT_HEIGHT: int = 640
    
    # Size limits
    MAX_FILE_SIZE_MB: int = 50
    MAX_DIMENSION: int = 4096
    MIN_DIMENSION: int = 32
    
    # Quality thresholds
    BLUR_THRESHOLD: float = 100.0  # Laplacian variance below this is blurry
    MIN_BRIGHTNESS: float = 20.0
    MAX_BRIGHTNESS: float = 235.0


# Object Detection Constants
class DetectionConstants:
    """Constants related to object detection."""
    
    # Model names
    YOLO_NANO: str = "yolov8n.pt"
    YOLO_SMALL: str = "yolov8s.pt"
    YOLO_MEDIUM: str = "yolov8m.pt"
    YOLO_LARGE: str = "yolov8l.pt"
    YOLO_XLARGE: str = "yolov8x.pt"
    
    # Default settings
    DEFAULT_CONFIDENCE: float = 0.5
    DEFAULT_IOU: float = 0.45
    DEFAULT_MAX_DETECTIONS: int = 100
    
    # COCO class names (subset of most common)
    COMMON_CLASSES: Tuple[str, ...] = (
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    )


# LLM Constants
class LLMConstants:
    """Constants related to language model processing."""
    
    # Model IDs
    BLIP_BASE: str = "Salesforce/blip-image-captioning-base"
    BLIP_LARGE: str = "Salesforce/blip-image-captioning-large"
    BLIP_VQA_BASE: str = "Salesforce/blip-vqa-base"
    
    # Generation settings
    DEFAULT_MAX_LENGTH: int = 50
    DEFAULT_NUM_BEAMS: int = 5
    DEFAULT_TEMPERATURE: float = 1.0
    
    # Device options
    DEVICE_AUTO: str = "auto"
    DEVICE_CPU: str = "cpu"
    DEVICE_CUDA: str = "cuda"
    DEVICE_MPS: str = "mps"


# Color Analysis Constants
class ColorConstants:
    """Constants related to color analysis."""
    
    # Methods
    METHOD_KMEANS: str = "kmeans"
    METHOD_HISTOGRAM: str = "histogram"
    
    # Settings
    DEFAULT_NUM_COLORS: int = 5
    MAX_COLORS: int = 20
    MIN_COLORS: int = 1
    
    # Color names for common colors (RGB ranges)
    COLOR_NAMES: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
        "red": ((150, 0, 0), (255, 100, 100)),
        "green": ((0, 150, 0), (100, 255, 100)),
        "blue": ((0, 0, 150), (100, 100, 255)),
        "yellow": ((200, 200, 0), (255, 255, 100)),
        "orange": ((200, 100, 0), (255, 180, 100)),
        "purple": ((100, 0, 100), (200, 100, 200)),
        "pink": ((200, 100, 150), (255, 200, 220)),
        "brown": ((100, 50, 0), (180, 120, 80)),
        "black": ((0, 0, 0), (50, 50, 50)),
        "white": ((200, 200, 200), (255, 255, 255)),
        "gray": ((100, 100, 100), (180, 180, 180)),
    }


# OCR Constants
class OCRConstants:
    """Constants related to OCR text extraction."""
    
    # Default language
    DEFAULT_LANGUAGE: str = "eng"
    
    # Page segmentation modes
    PSM_AUTO: int = 3
    PSM_SINGLE_BLOCK: int = 6
    PSM_SINGLE_LINE: int = 7
    
    # Minimum confidence for word inclusion
    MIN_CONFIDENCE: float = 0.0


# Face Detection Constants
class FaceConstants:
    """Constants related to face detection."""
    
    # Haar cascade parameters
    SCALE_FACTOR: float = 1.1
    MIN_NEIGHBORS: int = 5
    MIN_SIZE: Tuple[int, int] = (30, 30)


# Web Application Constants
class WebAppConstants:
    """Constants related to the web application."""
    
    # Default settings
    DEFAULT_THEME: str = "light"
    MAX_HISTORY_ITEMS: int = 5
    
    # Upload settings
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: Tuple[str, ...] = ImageConstants.SUPPORTED_FORMATS
    
    # Suggested questions
    DEFAULT_QUESTIONS: Tuple[str, ...] = (
        "Describe this image in detail",
        "What objects do you see?",
        "What is happening in this scene?",
        "What colors are dominant?",
        "Is there any text in this image?",
        "How many people are in this image?",
    )


# Logging Constants
class LoggingConstants:
    """Constants related to logging."""
    
    FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    
    # Module names
    MODULE_MAIN: str = "multimodal_ai"
    MODULE_IMAGE: str = "multimodal_ai.image"
    MODULE_LLM: str = "multimodal_ai.llm"
    MODULE_WEB: str = "multimodal_ai.web"
