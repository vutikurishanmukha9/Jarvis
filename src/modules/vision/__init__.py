"""
Computer Vision module for Jarvis (YOLOv8 object detection, OCR, OpenCV metrics).
"""

from .vision_bridge import (
    get_vision_tools,
    get_and_clear_annotated_images,
    register_uploaded_image,
    clear_active_images,
    analyze_image_deep,
    _ACTIVE_IMAGES
)

__all__ = [
    "get_vision_tools",
    "get_and_clear_annotated_images",
    "register_uploaded_image",
    "clear_active_images",
    "analyze_image_deep",
    "_ACTIVE_IMAGES"
]
