"""
Configuration management for the Multimodal AI System.

This module provides centralized configuration with:
- Environment variable support
- Default values with overrides
- Validation of configuration values
- Type-safe access to settings
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class YOLOConfig:
    """Configuration for YOLO object detection."""
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    
    # Available models from smallest to largest
    AVAILABLE_MODELS: List[str] = field(default_factory=lambda: [
        "yolov8n.pt",  # Nano
        "yolov8s.pt",  # Small
        "yolov8m.pt",  # Medium
        "yolov8l.pt",  # Large
        "yolov8x.pt",  # Extra Large
    ])
    
    def validate(self) -> bool:
        """Validate YOLO configuration."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be between 0 and 1, got {self.confidence_threshold}")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be between 0 and 1, got {self.iou_threshold}")
        return True


@dataclass
class BLIPConfig:
    """Configuration for BLIP vision-language model."""
    model_name: str = "Salesforce/blip-image-captioning-base"
    device: str = "auto"  # auto, cpu, cuda, mps
    torch_dtype: str = "auto"  # auto, float16, float32
    max_caption_length: int = 50
    max_answer_length: int = 100
    num_beams: int = 5
    temperature: float = 1.0
    
    AVAILABLE_MODELS: List[str] = field(default_factory=lambda: [
        "Salesforce/blip-image-captioning-base",
        "Salesforce/blip-image-captioning-large",
        "Salesforce/blip-vqa-base",
        "Salesforce/blip-vqa-capfilt-large",
    ])
    
    def validate(self) -> bool:
        """Validate BLIP configuration."""
        if self.device not in ["auto", "cpu", "cuda", "mps"]:
            raise ValueError(f"Invalid device: {self.device}")
        if not 0.1 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.1 and 2.0, got {self.temperature}")
        return True


@dataclass
class OCRConfig:
    """Configuration for OCR text extraction."""
    enabled: bool = True
    language: str = "eng"
    preprocess: bool = True
    min_confidence: float = 0.0


@dataclass
class ColorConfig:
    """Configuration for color extraction."""
    enabled: bool = True
    num_colors: int = 5
    method: str = "kmeans"  # kmeans, histogram
    
    def validate(self) -> bool:
        """Validate color configuration."""
        if not 1 <= self.num_colors <= 20:
            raise ValueError(f"num_colors must be between 1 and 20, got {self.num_colors}")
        if self.method not in ["kmeans", "histogram"]:
            raise ValueError(f"Invalid method: {self.method}")
        return True


@dataclass
class ImageConfig:
    """Configuration for image processing."""
    target_width: int = 640
    target_height: int = 640
    normalize: bool = True
    maintain_aspect_ratio: bool = True
    
    SUPPORTED_FORMATS: List[str] = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"
    ])
    
    MAX_FILE_SIZE_MB: int = 50
    
    @property
    def target_size(self) -> tuple:
        return (self.target_width, self.target_height)


@dataclass
class WebAppConfig:
    """Configuration for the Streamlit web application."""
    title: str = "Multimodal AI System"
    default_theme: str = "light"  # light, dark
    max_history_items: int = 5
    enable_face_detection: bool = True
    enable_quality_analysis: bool = True
    enable_url_loading: bool = True
    
    # Cache settings
    cache_ttl_seconds: int = 3600
    
    # Display settings
    show_debug_info: bool = False


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    
    VALID_LEVELS: List[str] = field(default_factory=lambda: [
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    ])
    
    def get_level(self) -> int:
        """Get logging level as integer."""
        return getattr(logging, self.level.upper(), logging.INFO)


@dataclass
class Config:
    """Main configuration class combining all settings."""
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    blip: BLIPConfig = field(default_factory=BLIPConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    webapp: WebAppConfig = field(default_factory=WebAppConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "multimodal_ai")
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        config = cls()
        
        # YOLO settings
        if os.getenv("YOLO_MODEL"):
            config.yolo.model_name = os.getenv("YOLO_MODEL")
        if os.getenv("YOLO_CONFIDENCE"):
            config.yolo.confidence_threshold = float(os.getenv("YOLO_CONFIDENCE"))
        
        # BLIP settings
        if os.getenv("BLIP_MODEL"):
            config.blip.model_name = os.getenv("BLIP_MODEL")
        if os.getenv("BLIP_DEVICE"):
            config.blip.device = os.getenv("BLIP_DEVICE")
        
        # Logging
        if os.getenv("LOG_LEVEL"):
            config.logging.level = os.getenv("LOG_LEVEL")
        
        # Web app
        if os.getenv("DEFAULT_THEME"):
            config.webapp.default_theme = os.getenv("DEFAULT_THEME")
        
        return config
    
    def validate(self) -> bool:
        """Validate all configuration settings."""
        self.yolo.validate()
        self.blip.validate()
        self.color.validate()
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "yolo": {
                "model_name": self.yolo.model_name,
                "confidence_threshold": self.yolo.confidence_threshold,
                "iou_threshold": self.yolo.iou_threshold,
            },
            "blip": {
                "model_name": self.blip.model_name,
                "device": self.blip.device,
                "max_caption_length": self.blip.max_caption_length,
            },
            "ocr": {
                "enabled": self.ocr.enabled,
                "language": self.ocr.language,
            },
            "color": {
                "enabled": self.color.enabled,
                "num_colors": self.color.num_colors,
                "method": self.color.method,
            },
            "image": {
                "target_size": self.image.target_size,
                "normalize": self.image.normalize,
            },
            "webapp": {
                "title": self.webapp.title,
                "default_theme": self.webapp.default_theme,
            },
        }


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _config
    _config = None
