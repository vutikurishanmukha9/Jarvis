"""
Image processing module using OpenCV for feature extraction.

This module provides the ImageProcessor class that handles:
- Image preprocessing (resizing, normalization)
- Object detection using YOLOv8
- Color analysis and extraction
- Text extraction using Tesseract OCR
"""

import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
from typing import List, Dict, Tuple, Optional, Any
import logging
from collections import Counter

from .utils import (
    load_image, resize_image, normalize_image, convert_bgr_to_rgb,
    draw_bounding_box, get_image_info
)

# Set up logging
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    A comprehensive image processor using OpenCV for various computer vision tasks.

    This class provides methods for:
    - Image preprocessing
    - Object detection using YOLOv8
    - Color extraction and analysis
    - Optical Character Recognition (OCR)
    """

    def __init__(self, yolo_model: str = "yolov8n.pt"):
        """
        Initialize the ImageProcessor.

        Args:
            yolo_model (str): Path to YOLO model weights or model name
        """
        self.yolo_model_path = yolo_model
        self.yolo_model = None
        self.current_image = None
        self.preprocessed_image = None

        # Load YOLO model
        self._load_yolo_model()

        logger.info(f"ImageProcessor initialized with YOLO model: {yolo_model}")

    def _load_yolo_model(self) -> None:
        """Load the YOLO model for object detection."""
        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            logger.info(f"Successfully loaded YOLO model: {self.yolo_model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model {self.yolo_model_path}: {str(e)}")
            self.yolo_model = None

    def preprocess(self, image_path: str, target_size: Tuple[int, int] = (640, 640),
                  normalize: bool = True) -> Optional[np.ndarray]:
        """
        Preprocess an image for analysis.

        Args:
            image_path (str): Path to the input image
            target_size (Tuple[int, int]): Target dimensions (width, height)
            normalize (bool): Whether to normalize pixel values to [0, 1]

        Returns:
            Optional[np.ndarray]: Preprocessed image, None if failed
        """
        try:
            # Load the image
            self.current_image = load_image(image_path)
            if self.current_image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None

            # Resize image while maintaining aspect ratio
            resized_image = resize_image(self.current_image, target_size, maintain_aspect=True)

            # Normalize if requested
            if normalize:
                self.preprocessed_image = normalize_image(resized_image)
            else:
                self.preprocessed_image = resized_image

            logger.info(f"Successfully preprocessed image with shape: {self.preprocessed_image.shape}")
            return self.preprocessed_image

        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None

    def detect_objects(self, confidence_threshold: float = 0.5, 
                      iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
        """
        Detect objects in the current image using YOLOv8.

        Args:
            confidence_threshold (float): Minimum confidence score for detections
            iou_threshold (float): IoU threshold for non-maximum suppression

        Returns:
            List[Dict[str, Any]]: List of detected objects with metadata
        """
        if self.yolo_model is None:
            logger.error("YOLO model not loaded")
            return []

        if self.current_image is None:
            logger.error("No image loaded for object detection")
            return []

        try:
            # Run YOLO inference
            results = self.yolo_model(
                self.current_image,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )

            detections = []

            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]

                        detection = {
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': (x1, y1, x2, y2),
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                            'area': (x2 - x1) * (y2 - y1)
                        }

                        detections.append(detection)

            logger.info(f"Detected {len(detections)} objects")
            return detections

        except Exception as e:
            logger.error(f"Error in object detection: {str(e)}")
            return []

    def extract_colors(self, num_colors: int = 5, method: str = 'kmeans') -> List[Dict[str, Any]]:
        """
        Extract dominant colors from the current image.

        Args:
            num_colors (int): Number of dominant colors to extract
            method (str): Method to use ('kmeans' or 'histogram')

        Returns:
            List[Dict[str, Any]]: List of dominant colors with metadata
        """
        if self.current_image is None:
            logger.error("No image loaded for color extraction")
            return []

        try:
            if method == 'kmeans':
                return self._extract_colors_kmeans(num_colors)
            elif method == 'histogram':
                return self._extract_colors_histogram(num_colors)
            else:
                logger.error(f"Unknown color extraction method: {method}")
                return []

        except Exception as e:
            logger.error(f"Error extracting colors: {str(e)}")
            return []

    def _extract_colors_kmeans(self, num_colors: int) -> List[Dict[str, Any]]:
        """Extract colors using K-means clustering."""
        # Reshape image to be a list of pixels
        image_rgb = convert_bgr_to_rgb(self.current_image)
        data = image_rgb.reshape((-1, 3))
        data = np.float32(data)

        # Define criteria and apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert centers to uint8 and calculate percentages
        centers = np.uint8(centers)
        labels = labels.flatten()

        colors = []
        for i, center in enumerate(centers):
            percentage = (np.sum(labels == i) / len(labels)) * 100

            color_info = {
                'rgb': tuple(center),
                'bgr': tuple(center[::-1]),  # Reverse for BGR
                'hex': f"#{center[0]:02x}{center[1]:02x}{center[2]:02x}",
                'percentage': round(percentage, 2),
                'pixel_count': int(np.sum(labels == i))
            }
            colors.append(color_info)

        # Sort by percentage (descending)
        colors.sort(key=lambda x: x['percentage'], reverse=True)

        logger.info(f"Extracted {len(colors)} dominant colors using K-means")
        return colors

    def _extract_colors_histogram(self, num_colors: int) -> List[Dict[str, Any]]:
        """Extract colors using color histogram analysis."""
        # Convert to RGB and quantize colors to reduce complexity
        image_rgb = convert_bgr_to_rgb(self.current_image)

        # Quantize to reduce color space
        quantized = (image_rgb // 32) * 32

        # Reshape and count colors
        pixels = quantized.reshape((-1, 3))
        color_counts = Counter(map(tuple, pixels))

        # Get most common colors
        most_common = color_counts.most_common(num_colors)
        total_pixels = len(pixels)

        colors = []
        for color, count in most_common:
            percentage = (count / total_pixels) * 100

            color_info = {
                'rgb': color,
                'bgr': color[::-1],  # Reverse for BGR
                'hex': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                'percentage': round(percentage, 2),
                'pixel_count': count
            }
            colors.append(color_info)

        logger.info(f"Extracted {len(colors)} dominant colors using histogram")
        return colors

    def extract_text(self, preprocessing_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract text from the current image using Tesseract OCR.

        Args:
            preprocessing_config (Optional[Dict[str, Any]]): OCR preprocessing options

        Returns:
            Dict[str, Any]: Extracted text with metadata and bounding boxes
        """
        if self.current_image is None:
            logger.error("No image loaded for text extraction")
            return {'text': '', 'confidence': 0, 'words': []}

        try:
            # Default preprocessing configuration
            if preprocessing_config is None:
                preprocessing_config = {
                    'convert_to_gray': True,
                    'apply_gaussian_blur': True,
                    'apply_threshold': True,
                    'threshold_type': cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                    'morph_operations': True
                }

            # Preprocess image for better OCR results
            processed_image = self._preprocess_for_ocr(preprocessing_config)

            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6'

            # Extract text with detailed information
            text_data = pytesseract.image_to_data(
                processed_image, 
                config=custom_config, 
                output_type=pytesseract.Output.DICT
            )

            # Extract simple text
            full_text = pytesseract.image_to_string(processed_image, config=custom_config)

            # Process detailed results
            words = []
            confidences = []

            for i in range(len(text_data['text'])):
                if int(text_data['conf'][i]) > 0:  # Filter out low confidence
                    word_info = {
                        'text': text_data['text'][i].strip(),
                        'confidence': int(text_data['conf'][i]),
                        'bbox': (
                            text_data['left'][i],
                            text_data['top'][i],
                            text_data['left'][i] + text_data['width'][i],
                            text_data['top'][i] + text_data['height'][i]
                        ),
                        'level': text_data['level'][i]
                    }

                    if word_info['text']:  # Only add non-empty text
                        words.append(word_info)
                        confidences.append(word_info['confidence'])

            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0

            result = {
                'text': full_text.strip(),
                'confidence': round(avg_confidence, 2),
                'words': words,
                'word_count': len([w for w in words if w['text']]),
                'preprocessing_used': preprocessing_config
            }

            logger.info(f"Extracted text: '{result['text'][:50]}...' with {len(words)} words")
            return result

        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return {'text': '', 'confidence': 0, 'words': [], 'error': str(e)}

    def _preprocess_for_ocr(self, config: Dict[str, Any]) -> np.ndarray:
        """Apply preprocessing steps to improve OCR accuracy."""
        processed = self.current_image.copy()

        try:
            # Convert to grayscale
            if config.get('convert_to_gray', True):
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            if config.get('apply_gaussian_blur', True):
                processed = cv2.GaussianBlur(processed, (5, 5), 0)

            # Apply threshold for better text segmentation
            if config.get('apply_threshold', True):
                threshold_type = config.get('threshold_type', cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, processed = cv2.threshold(processed, 0, 255, threshold_type)

            # Apply morphological operations to clean up
            if config.get('morph_operations', True):
                kernel = np.ones((2, 2), np.uint8)
                processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
                processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

            return processed

        except Exception as e:
            logger.error(f"Error in OCR preprocessing: {str(e)}")
            return self.current_image

    def draw_detections(self, detections: List[Dict[str, Any]], 
                       draw_on_original: bool = True) -> np.ndarray:
        """
        Draw detection results on the image.

        Args:
            detections (List[Dict[str, Any]]): Detection results from detect_objects()
            draw_on_original (bool): Whether to draw on original or preprocessed image

        Returns:
            np.ndarray: Image with detections drawn
        """
        if draw_on_original and self.current_image is not None:
            result_image = self.current_image.copy()
        elif self.preprocessed_image is not None:
            result_image = self.preprocessed_image.copy()
            # Convert to uint8 if normalized
            if result_image.dtype == np.float32:
                result_image = (result_image * 255).astype(np.uint8)
        else:
            logger.error("No image available for drawing detections")
            return np.zeros((100, 100, 3), dtype=np.uint8)

        try:
            # Define colors for different classes
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
            ]

            for i, detection in enumerate(detections):
                color = colors[i % len(colors)]
                label = f"{detection['class_name']}: {detection['confidence']:.2f}"

                result_image = draw_bounding_box(
                    result_image, 
                    detection['bbox'], 
                    label, 
                    color
                )

            logger.info(f"Drew {len(detections)} detections on image")
            return result_image

        except Exception as e:
            logger.error(f"Error drawing detections: {str(e)}")
            return result_image if 'result_image' in locals() else np.zeros((100, 100, 3), dtype=np.uint8)

    def get_image_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the current image.

        Returns:
            Dict[str, Any]: Image statistics and metadata
        """
        if self.current_image is None:
            return {}

        try:
            stats = get_image_info(self.current_image)

            # Add more detailed statistics
            if len(self.current_image.shape) == 3:
                # Color image statistics
                for i, channel in enumerate(['Blue', 'Green', 'Red']):
                    channel_data = self.current_image[:, :, i]
                    stats[f'{channel.lower()}_mean'] = float(np.mean(channel_data))
                    stats[f'{channel.lower()}_std'] = float(np.std(channel_data))
                    stats[f'{channel.lower()}_min'] = int(np.min(channel_data))
                    stats[f'{channel.lower()}_max'] = int(np.max(channel_data))

            # Calculate brightness and contrast
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            stats['brightness'] = float(np.mean(gray))
            stats['contrast'] = float(np.std(gray))

            # Calculate sharpness (using Laplacian variance)
            stats['sharpness'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            return stats

        except Exception as e:
            logger.error(f"Error calculating image statistics: {str(e)}")
            return {}
