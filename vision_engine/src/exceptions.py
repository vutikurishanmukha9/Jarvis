"""
Custom exceptions for the Multimodal AI System.

This module provides a hierarchy of exceptions for better error handling
and more informative error messages.
"""


class MultimodalAIError(Exception):
    """Base exception for all Multimodal AI errors."""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# Image Processing Exceptions
class ImageProcessingError(MultimodalAIError):
    """Base exception for image processing errors."""
    pass


class ImageLoadError(ImageProcessingError):
    """Raised when an image cannot be loaded."""
    
    def __init__(self, path: str, reason: str = None):
        message = f"Failed to load image: {path}"
        details = {"path": path}
        if reason:
            details["reason"] = reason
        super().__init__(message, details)


class ImageFormatError(ImageProcessingError):
    """Raised when image format is not supported."""
    
    def __init__(self, format: str, supported: list = None):
        message = f"Unsupported image format: {format}"
        details = {"format": format}
        if supported:
            details["supported_formats"] = supported
        super().__init__(message, details)


class ImageSizeError(ImageProcessingError):
    """Raised when image size is invalid or too large."""
    
    def __init__(self, size: tuple, max_size: tuple = None):
        message = f"Invalid image size: {size}"
        details = {"size": size}
        if max_size:
            details["max_size"] = max_size
        super().__init__(message, details)


# Object Detection Exceptions
class ObjectDetectionError(MultimodalAIError):
    """Base exception for object detection errors."""
    pass


class ModelLoadError(ObjectDetectionError):
    """Raised when a model fails to load."""
    
    def __init__(self, model_name: str, reason: str = None):
        message = f"Failed to load model: {model_name}"
        details = {"model_name": model_name}
        if reason:
            details["reason"] = reason
        super().__init__(message, details)


class DetectionError(ObjectDetectionError):
    """Raised when object detection fails."""
    pass


# LLM Exceptions
class LLMError(MultimodalAIError):
    """Base exception for LLM-related errors."""
    pass


class ModelInitializationError(LLMError):
    """Raised when LLM model fails to initialize."""
    
    def __init__(self, model_id: str, reason: str = None):
        message = f"Failed to initialize LLM model: {model_id}"
        details = {"model_id": model_id}
        if reason:
            details["reason"] = reason
        super().__init__(message, details)


class CaptionGenerationError(LLMError):
    """Raised when caption generation fails."""
    pass


class QuestionAnsweringError(LLMError):
    """Raised when question answering fails."""
    pass


# OCR Exceptions
class OCRError(MultimodalAIError):
    """Base exception for OCR-related errors."""
    pass


class TesseractNotFoundError(OCRError):
    """Raised when Tesseract is not installed or not found."""
    
    def __init__(self):
        message = "Tesseract OCR is not installed or not found in PATH"
        details = {
            "install_instructions": {
                "windows": "Download from https://github.com/UB-Mannheim/tesseract/wiki",
                "linux": "sudo apt-get install tesseract-ocr",
                "mac": "brew install tesseract"
            }
        }
        super().__init__(message, details)


# Configuration Exceptions
class ConfigurationError(MultimodalAIError):
    """Base exception for configuration errors."""
    pass


class InvalidConfigError(ConfigurationError):
    """Raised when configuration values are invalid."""
    
    def __init__(self, key: str, value, valid_values: list = None):
        message = f"Invalid configuration value for '{key}': {value}"
        details = {"key": key, "value": value}
        if valid_values:
            details["valid_values"] = valid_values
        super().__init__(message, details)


# Resource Exceptions
class ResourceError(MultimodalAIError):
    """Base exception for resource-related errors."""
    pass


class OutOfMemoryError(ResourceError):
    """Raised when system runs out of memory."""
    
    def __init__(self, required: int = None, available: int = None):
        message = "Out of memory"
        details = {}
        if required:
            details["required_mb"] = required
        if available:
            details["available_mb"] = available
        super().__init__(message, details)


class GPUNotAvailableError(ResourceError):
    """Raised when GPU is requested but not available."""
    
    def __init__(self):
        message = "GPU requested but not available. Using CPU instead."
        super().__init__(message)


# URL Loading Exceptions
class URLLoadError(MultimodalAIError):
    """Raised when loading from URL fails."""
    
    def __init__(self, url: str, reason: str = None):
        message = f"Failed to load from URL: {url}"
        details = {"url": url}
        if reason:
            details["reason"] = reason
        super().__init__(message, details)


# Analysis Exceptions
class AnalysisError(MultimodalAIError):
    """Base exception for analysis errors."""
    pass


class FaceDetectionError(AnalysisError):
    """Raised when face detection fails."""
    pass


class ColorExtractionError(AnalysisError):
    """Raised when color extraction fails."""
    pass


class QualityAnalysisError(AnalysisError):
    """Raised when quality analysis fails."""
    pass
