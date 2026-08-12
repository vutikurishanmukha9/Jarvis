from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

class VisionService(ABC):
    """Abstract base class for vision services (Object Detection)."""

    @abstractmethod
    def detect_objects(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: Input image (numpy array)
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of detected objects with bbox, class_name, confidence.
        """
        pass

    def detect_objects_batch(self, images: List[np.ndarray], confidence_threshold: float = 0.5) -> List[List[Dict[str, Any]]]:
        """
        Detect objects in multiple images (Batch Processing).
        Default implementation loops over simple detect_objects.
        """
        return [self.detect_objects(img, confidence_threshold) for img in images]

class LLMService(ABC):
    """Abstract base class for LLM/VLM services (Captioning, VQA)."""

    @abstractmethod
    def caption_image(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """
        Generate a caption for the image.
        
        Args:
            image: Input image (PIL)
            prompt: Optional conditional text
            
        Returns:
            Generated caption string.
        """
        pass

    @abstractmethod
    def answer_question(self, image: Image.Image, question: str, context: str = "") -> str:
        """
        Answer a question about the image.
        
        Args:
            image: Input image (PIL)
            question: User question
            context: Additional context (cptional)
            
        Returns:
            Answer string.
        """
        pass

    def caption_image_batch(self, images: List[Image.Image], prompt: Optional[str] = None) -> List[str]:
        """Batch caption generation."""
        return [self.caption_image(img, prompt) for img in images]

    def answer_question_batch(self, images: List[Image.Image], questions: List[str], contexts: Optional[List[str]] = None) -> List[str]:
        """Batch question answering."""
        if contexts is None:
            contexts = [""] * len(images)
        return [self.answer_question(img, q, c) for img, q, c in zip(images, questions, contexts)]
