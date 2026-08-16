"""
Multimodal AI System package.

This package provides a comprehensive multimodal AI system that combines 
computer vision and natural language processing for image analysis.

Modules:
- image_processor: OpenCV-based image processing and feature extraction
- llm_integration: BLIP vision-language model integration
- multimodal_system: Main system orchestrator combining all components
- utils: Shared utility functions
- web_app: Streamlit-based web interface
- config: Centralized configuration management
- exceptions: Custom exception hierarchy
- constants: Application constants

Example usage:
    from src.multimodal_system import MultimodalAI
    
    system = MultimodalAI()
    results = system.process("image.jpg", "What do you see?")
    print(results['answer'])
"""

# Version
from .constants import VERSION, VERSION_INFO

# Core classes
from .image_processor import ImageProcessor
from .llm_integration import LLMProcessor
from .multimodal_system import MultimodalAI

# Configuration
from .config import Config, get_config, set_config

# Exceptions
from .exceptions import (
    MultimodalAIError,
    ImageProcessingError,
    ImageLoadError,
    ImageFormatError,
    ObjectDetectionError,
    ModelLoadError,
    LLMError,
    ModelInitializationError,
    OCRError,
    ConfigurationError,
)

# Utilities
from .utils import (
    load_image,
    load_image_pil,
    resize_image,
    normalize_image,
    convert_bgr_to_rgb,
    convert_rgb_to_bgr,
    validate_image_format,
    draw_bounding_box,
    get_image_info
)

__version__ = VERSION
__author__ = "Multimodal AI Team"

__all__ = [
    # Version
    "VERSION",
    "VERSION_INFO",
    "__version__",
    
    # Core classes
    "ImageProcessor",
    "LLMProcessor", 
    "MultimodalAI",
    
    # Configuration
    "Config",
    "get_config",
    "set_config",
    
    # Exceptions
    "MultimodalAIError",
    "ImageProcessingError",
    "ImageLoadError",
    "ImageFormatError",
    "ObjectDetectionError",
    "ModelLoadError",
    "LLMError",
    "ModelInitializationError",
    "OCRError",
    "ConfigurationError",
    
    # Utilities
    "load_image",
    "load_image_pil",
    "resize_image",
    "normalize_image",
    "convert_bgr_to_rgb",
    "convert_rgb_to_bgr",
    "validate_image_format",
    "draw_bounding_box",
    "get_image_info",
]
