"""
Computer Vision module for Jarvis (YOLOv8 object detection, OCR, OpenCV metrics).
"""

from .vision_bridge import (
    _ACTIVE_IMAGES,
    analyze_image_deep,
    analyze_uploaded_images,
    clear_active_images,
    extract_scene_text_ocr,
    extract_text_paddleocr,
    get_and_clear_annotated_images,
    get_vision_tools,
    is_paddleocr_available,
    register_uploaded_image,
)

__all__ = [
    "get_vision_tools",
    "get_and_clear_annotated_images",
    "register_uploaded_image",
    "clear_active_images",
    "analyze_image_deep",
    "analyze_uploaded_images",
    "extract_scene_text_ocr",
    "extract_text_paddleocr",
    "is_paddleocr_available",
    "_ACTIVE_IMAGES",
]
