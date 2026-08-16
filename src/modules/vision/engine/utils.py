"""
Utility functions for the multimodal AI system.

This module provides shared helper functions for image loading, preprocessing,
and common operations used across the multimodal AI pipeline.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional, Dict, Any
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file path using OpenCV.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array in BGR format, None if failed
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
            
        logger.info(f"Successfully loaded image: {image_path} with shape {image.shape}")
        return image
        
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None


def load_image_pil(image_path: str) -> Optional[Image.Image]:
    """
    Load an image using PIL/Pillow for compatibility with transformers.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object, None if failed
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        image = Image.open(image_path).convert('RGB')
        logger.info(f"Successfully loaded PIL image: {image_path} with size {image.size}")
        return image
        
    except Exception as e:
        logger.error(f"Error loading PIL image {image_path}: {str(e)}")
        return None


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image to target dimensions.
    
    Args:
        image: Input image as numpy array
        target_size: Target (width, height)
        maintain_aspect: Whether to maintain aspect ratio with padding
        
    Returns:
        Resized image
    """
    if image is None:
        logger.error("Cannot resize None image")
        return image
        
    try:
        if maintain_aspect:
            # Calculate scaling factor to fit within target size
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Avoid division by zero
            if w == 0 or h == 0:
                logger.error("Image has zero dimension")
                return image
            
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Ensure minimum size of 1
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create padded image if needed
            if new_w != target_w or new_h != target_h:
                # Create black background with same dtype
                padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
                
                # Center the resized image
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
                
                return padded
            return resized
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1] range.
    
    Args:
        image: Input image
        
    Returns:
        Normalized image as float32
    """
    if image is None:
        logger.error("Cannot normalize None image")
        return image
        
    try:
        return image.astype(np.float32) / 255.0
    except Exception as e:
        logger.error(f"Error normalizing image: {str(e)}")
        return image


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB format.
    
    Args:
        image: BGR image
        
    Returns:
        RGB image
    """
    if image is None:
        logger.error("Cannot convert None image")
        return image
        
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error converting BGR to RGB: {str(e)}")
        return image


def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to BGR format.
    
    Args:
        image: RGB image
        
    Returns:
        BGR image
    """
    if image is None:
        logger.error("Cannot convert None image")
        return image
        
    try:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Error converting RGB to BGR: {str(e)}")
        return image


def create_output_directory(output_dir: str) -> bool:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Directory path to create
        
    Returns:
        True if directory exists or was created successfully
    """
    if not output_dir:
        logger.error("Empty output directory path provided")
        return False
        
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory ready: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Error creating output directory {output_dir}: {str(e)}")
        return False


def validate_image_format(image_path: str) -> bool:
    """
    Validate if file is a supported image format.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if supported format
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
    
    if not image_path:
        logger.error("Empty image path provided")
        return False
        
    try:
        _, ext = os.path.splitext(image_path.lower())
        return ext in supported_formats
    except Exception as e:
        logger.error(f"Error validating image format: {str(e)}")
        return False


def draw_bounding_box(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                     label: str = "", color: Tuple[int, int, int] = (0, 255, 0), 
                     thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box with label on image.
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        label: Text label to display
        color: BGR color for box and text
        thickness: Line thickness
        
    Returns:
        Image with bounding box drawn
    """
    if image is None:
        logger.error("Cannot draw on None image")
        return image
        
    try:
        # Make a copy to avoid modifying original
        result = image.copy()
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw rectangle
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label if provided
        if label:
            # Get text size for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Ensure label background is within image bounds
            label_y1 = max(0, y1 - text_h - baseline - 5)
            
            # Draw background rectangle for text
            cv2.rectangle(result, (x1, label_y1), (x1 + text_w, y1), color, -1)
            
            # Draw text
            cv2.putText(result, label, (x1, y1 - baseline - 2), 
                       font, font_scale, (255, 255, 255), font_thickness)
        
        return result
        
    except Exception as e:
        logger.error(f"Error drawing bounding box: {str(e)}")
        return image


def get_image_info(image: np.ndarray) -> Dict[str, Any]:
    """
    Get basic information about an image.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image information including shape, dtype, etc.
    """
    if image is None:
        logger.error("Cannot get info from None image")
        return {}
        
    try:
        info = {
            'shape': image.shape,
            'height': image.shape[0],
            'width': image.shape[1],
            'dtype': str(image.dtype),
            'min_value': float(np.min(image)),
            'max_value': float(np.max(image)),
            'mean_value': float(np.mean(image)),
            'std_value': float(np.std(image))
        }
        
        if len(image.shape) == 3:
            info['channels'] = image.shape[2]
        else:
            info['channels'] = 1
            
        # Add size in bytes
        info['size_bytes'] = image.nbytes
            
        return info
        
    except Exception as e:
        logger.error(f"Error getting image info: {str(e)}")
        return {}


def ensure_rgb(image: np.ndarray, is_bgr: bool = True) -> np.ndarray:
    """
    Ensure image is in RGB format.
    
    Args:
        image: Input image
        is_bgr: Whether input is in BGR format (default True for OpenCV images)
        
    Returns:
        RGB image
    """
    if image is None:
        return image
        
    if is_bgr:
        return convert_bgr_to_rgb(image)
    return image


def clamp_bbox(bbox: Tuple[int, int, int, int], 
               image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    """
    Clamp bounding box coordinates to image boundaries.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        image_width: Image width
        image_height: Image height
        
    Returns:
        Clamped bounding box
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))
    return (x1, y1, x2, y2)