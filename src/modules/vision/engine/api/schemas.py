from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class AnalysisConfig(BaseModel):
    """Configuration for the analysis job."""
    target_size: tuple[int, int] = (640, 640)
    normalize: bool = True
    reasoning_mode: str = "detailed"  # "fast" or "detailed"
    llm_model: str = "blip"  # "blip" or "llava"
    object_detection: Dict[str, Any] = Field(default_factory=lambda: {
        'enabled': True,
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45
    })
    color_extraction: Dict[str, Any] = Field(default_factory=lambda: {
        'enabled': True,
        'num_colors': 5,
        'method': 'kmeans'
    })
    text_extraction: Dict[str, Any] = Field(default_factory=lambda: {
        'enabled': True,
        'preprocessing': {
            'convert_to_gray': True,
            'apply_gaussian_blur': True,
            'apply_threshold': True
        }
    })
    captioning: Dict[str, Any] = Field(default_factory=lambda: {
        'max_length': 50,
        'num_beams': 5,
        'temperature': 1.0,
        'conditional_text': None
    })
    question_answering: Dict[str, Any] = Field(default_factory=lambda: {
        'max_length': 100,
        'num_beams': 5,
        'temperature': 1.0,
        'use_features_context': True
    })

class AnalysisRequest(BaseModel):
    """Request model for analyzing an image."""
    question: str = "Describe this image in detail"
    config: Optional[AnalysisConfig] = None

class JobResponse(BaseModel):
    """Response model for a job submission or status check."""
    job_id: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
