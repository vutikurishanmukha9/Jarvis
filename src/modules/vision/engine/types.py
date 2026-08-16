"""
Type definitions for the Multimodal AI System.

This module provides type aliases and TypedDict definitions for 
better type checking and IDE support throughout the codebase.
"""

from typing import TypedDict, List, Dict, Any, Optional, Tuple, Union
from typing_extensions import NotRequired
import numpy as np
from PIL import Image


# Basic type aliases
ImageArray = np.ndarray
PILImage = Image.Image
BoundingBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)
RGBColor = Tuple[int, int, int]
Point = Tuple[int, int]


# Object Detection Types
class DetectedObject(TypedDict):
    """Type definition for a detected object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    center: Point
    area: int


class DetectionResult(TypedDict):
    """Type definition for object detection results."""
    objects: List[DetectedObject]
    inference_time: float
    image_size: Tuple[int, int]


# Color Analysis Types
class ExtractedColor(TypedDict):
    """Type definition for an extracted color."""
    rgb: RGBColor
    hex: str
    percentage: float
    name: NotRequired[str]


class ColorResult(TypedDict):
    """Type definition for color extraction results."""
    colors: List[ExtractedColor]
    method: str
    num_colors: int


# OCR Types
class OCRWord(TypedDict):
    """Type definition for an OCR word."""
    text: str
    confidence: float
    bbox: NotRequired[BoundingBox]


class OCRResult(TypedDict):
    """Type definition for OCR results."""
    text: str
    words: List[OCRWord]
    word_count: int
    avg_confidence: float


# Face Detection Types
class DetectedFace(TypedDict):
    """Type definition for a detected face."""
    bbox: BoundingBox
    confidence: float
    width: int
    height: int


class FaceResult(TypedDict):
    """Type definition for face detection results."""
    faces: List[DetectedFace]
    total_faces: int


# Image Quality Types
class QualityMetrics(TypedDict):
    """Type definition for image quality metrics."""
    brightness: float
    contrast: float
    sharpness: float
    is_blurry: bool
    brightness_score: float
    contrast_score: float
    sharpness_score: float
    overall_score: float


# Caption/Answer Types
class CaptionResult(TypedDict):
    """Type definition for image captioning results."""
    caption: str
    confidence: float
    caption_type: str
    conditional_text: Optional[str]
    word_count: int
    character_count: int


class AnswerResult(TypedDict):
    """Type definition for question answering results."""
    answer: str
    confidence: float
    question: str
    context_used: NotRequired[str]


# Combined Features Types
class ImageFeatures(TypedDict):
    """Type definition for combined image features."""
    objects: List[DetectedObject]
    colors: List[ExtractedColor]
    ocr_text: OCRResult
    image_stats: Dict[str, Any]


# Main Analysis Result Types
class AnalysisResult(TypedDict):
    """Type definition for complete analysis results."""
    image_path: str
    question: str
    timestamp: str
    features: ImageFeatures
    caption: CaptionResult
    answer: AnswerResult
    summary: Dict[str, Any]
    faces: NotRequired[FaceResult]
    quality: NotRequired[QualityMetrics]


# Configuration Types
class AnalysisConfig(TypedDict):
    """Type definition for analysis configuration."""
    target_size: Tuple[int, int]
    normalize: bool
    object_detection: Dict[str, Any]
    color_extraction: Dict[str, Any]
    text_extraction: Dict[str, Any]
    captioning: Dict[str, Any]
    question_answering: Dict[str, Any]


# System Info Types
class SystemInfo(TypedDict):
    """Type definition for system information."""
    yolo_model: str
    blip_model: str
    device: str
    image_processor_ready: bool
    llm_processor_ready: bool


# Export all types
__all__ = [
    # Basic types
    "ImageArray",
    "PILImage",
    "BoundingBox",
    "RGBColor",
    "Point",
    
    # Detection types
    "DetectedObject",
    "DetectionResult",
    
    # Color types
    "ExtractedColor",
    "ColorResult",
    
    # OCR types
    "OCRWord",
    "OCRResult",
    
    # Face types
    "DetectedFace",
    "FaceResult",
    
    # Quality types
    "QualityMetrics",
    
    # Caption/Answer types
    "CaptionResult",
    "AnswerResult",
    
    # Feature types
    "ImageFeatures",
    
    # Result types
    "AnalysisResult",
    
    # Config types
    "AnalysisConfig",
    
    # System types
    "SystemInfo",
]
