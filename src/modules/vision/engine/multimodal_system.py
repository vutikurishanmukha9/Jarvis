"""
Multimodal AI System - Main integration module.

This module provides the MultimodalAI class that combines:
- Image processing with OpenCV
- Vision-language modeling with BLIP
- Comprehensive multimodal analysis pipeline
"""

import logging
from typing import Dict, List, Any, Optional, Union
from PIL import Image
import cv2
import json
import os
from datetime import datetime
import numpy as np


# Import Services
from .services import LocalYoloService, LocalBlipService, LlaVAService
from .utils import load_image_pil, validate_image_format, create_output_directory

# Set up logging
logger = logging.getLogger(__name__)


class MultimodalAI:
    """
    Main multimodal AI system that combines computer vision and language understanding.
    
    Now uses a Service-based architecture for better scalability and abstraction.
    """

    def __init__(self, 
                 yolo_model: str = "yolov8n.pt",
                 blip_model: str = "Salesforce/blip-image-captioning-base",
                 llava_model: str = "llava-hf/llava-1.5-7b-hf",
                 llm_type: str = "blip",
                 device: str = "auto"):
        """
        Initialize the multimodal AI system.

        Args:
            yolo_model (str): YOLO model for object detection
            blip_model (str): BLIP model for image captioning
            llava_model (str): LlaVA model id
            llm_type (str): 'blip' or 'llava'
            device (str): Computation device ('auto', 'cpu', 'cuda')
        """
        self.device = device
        self.yolo_model_path = yolo_model
        self.blip_model_id = blip_model
        self.llava_model_id = llava_model
        self.llm_type = llm_type

        # Initialize services
        self.vision_service = None
        self.llm_service = None

        # Initialize components
        self._initialize_components()

        # Store results from last processing
        self.last_results = {}

        logger.info(f"MultimodalAI system initialized successfully (LLM: {self.llm_type})")

    def _initialize_components(self) -> None:
        """Initialize the vision and LLM services."""
        try:
            logger.info("Initializing Vision Service...")
            self.vision_service = LocalYoloService(model_path=self.yolo_model_path)

            logger.info(f"Initializing LLM Service ({self.llm_type})...")
            if self.llm_type == "llava":
                self.llm_service = LlaVAService(model_id=self.llava_model_id, device=self.device)
            else:
                self.llm_service = LocalBlipService(model_id=self.blip_model_id, device=self.device)

            logger.info("All services initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def _switch_llm_service(self, new_type: str) -> None:
        """Switch the active LLM service if needed."""
        if new_type == self.llm_type and self.llm_service is not None:
            return

        logger.info(f"Switching LLM service from {self.llm_type} to {new_type}...")
        try:
            # Unload current service to free memory (crucial for GPU)
            if self.llm_service:
                del self.llm_service
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.llm_type = new_type
            if new_type == "llava":
                self.llm_service = LlaVAService(model_id=self.llava_model_id, device=self.device)
            else:
                self.llm_service = LocalBlipService(model_id=self.blip_model_id, device=self.device)
                
            logger.info(f"Successfully switched to {new_type}")
        except Exception as e:
            logger.error(f"Failed to switch LLM service: {str(e)}")
            # Fallback to BLIP if LlaVA fails
            if new_type != "blip":
                 logger.warning("Falling back to BLIP...")
                 self._switch_llm_service("blip")
            else:
                 raise

    def process(self, image_path: str, question: str,
                analysis_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main processing pipeline for comprehensive multimodal analysis.

        Args:
            image_path (str): Path to the input image
            question (str): Question to answer about the image
            analysis_config (Optional[Dict[str, Any]]): Configuration for analysis steps

        Returns:
            Dict[str, Any]: Comprehensive analysis results
        """
        if not validate_image_format(image_path):
            return {'error': f"Unsupported image format: {image_path}"}

        if not os.path.exists(image_path):
            return {'error': f"Image file not found: {image_path}"}

        # Use default config if none provided
        if analysis_config is None:
            analysis_config = self._get_default_config()

        # Check for model switch
        requested_model = analysis_config.get('llm_model', 'blip')
        if requested_model != self.llm_type:
             self._switch_llm_service(requested_model)

        logger.info(f"Starting multimodal analysis of: {image_path}")
        
        # Load image
        try:
            pil_image = Image.open(image_path).convert('RGB')
            # Keeping cv2 image for some legacy/utility functions if needed
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) 
        except Exception as e:
            return {'error': f"Failed to load image: {str(e)}"}

        results = {
            'image_path': image_path,
            'question': question,
            'timestamp': datetime.now().isoformat(),
            'analysis_config': analysis_config,
            'features': {},
            'processing_steps': {}
        }

        try:
            # Step 1: Object Detection (Vision Service)
            logger.info("Step 1: Object Detection")
            detections = self.vision_service.detect_objects(
                cv_image, 
                confidence_threshold=analysis_config.get('object_detection', {}).get('confidence_threshold', 0.5)
            )
            results['features']['objects'] = detections
            results['processing_steps']['object_detection'] = {'status': 'success', 'count': len(detections)}
            
            # Step 2: Visual Chain-of-Thought (Advanced Reasoning)
            reasoning_context = ""
            if analysis_config.get('reasoning_mode') == 'detailed':
                logger.info("Step 2: Visual Chain-of-Thought Analysis")
                reasoning_context = self._perform_visual_cot(pil_image, detections)
                results['processing_steps']['visual_cot'] = {'status': 'success'}
            else:
                logger.info("Skipping detailed visual reasoning (Fast Mode)")

            # Step 3: Global Captioning (LLM Service)
            logger.info("Step 3: Global Captioning")
            global_caption = self.llm_service.caption_image(pil_image)
            results['caption'] = {'caption': global_caption} # Maintain legacy structure
            
            # Step 4: Question Answering (LLM Service)
            logger.info("Step 4: Question Answering")
            
            # specific logic for Q&A prompt construction
            full_context = f"Image description: {global_caption}. "
            if reasoning_context:
                full_context += f" Detailed observations: {reasoning_context}"
            else:
                # Basic features context
                obj_counts = {}
                for obj in detections:
                    obj_counts[obj['class_name']] = obj_counts.get(obj['class_name'], 0) + 1
                obj_desc = ", ".join([f"{count} {name}" for name, count in obj_counts.items()])
                if obj_desc:
                    full_context += f" Objects detected: {obj_desc}."

            logger.info(f"Context for QA: {full_context}")
            
            # Answer question using the context + image
            # Note: We pass the image to the service, but the prompt includes the context we built
            answer = self.llm_service.answer_question(pil_image, question, context=full_context)
            results['answer'] = {'answer': answer} # Maintain legacy structure

            # Summary generation (simplified)
            results['summary'] = {
                'key_findings': [
                    f"Detected {len(detections)} objects",
                    f"Caption: {global_caption}",
                    f"Answer: {answer}"
                ]
            }
            
            self.last_results = results
            return results

        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            logger.error(error_msg)
            results['error'] = error_msg
            return results

    def _perform_visual_cot(self, image: Image.Image, detections: List[Dict[str, Any]]) -> str:
        """
        Perform Visual Chain-of-Thought analysis.
        Crops interesting objects and captions them individually.
        """
        observations = []
        
        # Sort detections by confidence and take top 5
        top_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:5]
        
        for det in top_detections:
            bbox = det['bbox'] # x1, y1, x2, y2
            class_name = det['class_name']
            
            # Crop image
            try:
                crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                
                # Caption the crop
                # We prompt specifically to describe the object
                crop_caption = self.llm_service.caption_image(crop)
                
                observations.append(f"The {class_name} looks like {crop_caption}")
            except Exception as e:
                logger.warning(f"Failed to process crop for {class_name}: {e}")
                continue
                
        return ". ".join(observations)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for analysis."""
        return {
            'target_size': (640, 640),
            'normalize': True,
            'reasoning_mode': 'detailed', # Default to detailed for now as requested
            'object_detection': {
                'enabled': True,
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45
            },
            'color_extraction': {
                'enabled': True,
                'num_colors': 5,
                'method': 'kmeans'
            },
            'captioning': {
                'max_length': 50,
                'num_beams': 5,
                'temperature': 1.0,
            },
            'question_answering': {
                'max_length': 100,
                'num_beams': 5,
            }
        }

    def process_batch(self, image_paths: List[str], questions: List[str],
                      analysis_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Batch processing pipeline.
        """
        if not image_paths:
            return []
            
        # Use default config if none provided
        if analysis_config is None:
            analysis_config = self._get_default_config()

        # Check for model switch (using first config request - assume batch is uniform)
        requested_model = analysis_config.get('llm_model', 'blip')
        if requested_model != self.llm_type:
             self._switch_llm_service(requested_model)
             
        # Load images
        pil_images = []
        cv_images = []
        valid_indices = []
        results = [None] * len(image_paths) # Placeholder for results
        
        for i, path in enumerate(image_paths):
            try:
                if not os.path.exists(path):
                    results[i] = {'error': f"File not found: {path}"}
                    continue
                    
                pil_img = Image.open(path).convert('RGB')
                pil_images.append(pil_img)
                cv_images.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
                valid_indices.append(i)
            except Exception as e:
                results[i] = {'error': f"Failed to load: {str(e)}"}

        if not valid_indices:
            return results

        try:
            # Step 1: Batch Object Detection
            logger.info(f"Batch Step 1: Object Detection ({len(cv_images)} images)")
            batch_detections = self.vision_service.detect_objects_batch(
                cv_images, 
                confidence_threshold=analysis_config.get('object_detection', {}).get('confidence_threshold', 0.5)
            )
            
            # Step 2: Global Captioning
            logger.info("Batch Step 2: Global Captioning")
            global_captions = self.llm_service.caption_image_batch(pil_images)
            
            # Step 3: Question Answering (Context Construction + QA)
            logger.info("Batch Step 3: Question Answering")
            contexts = []
            valid_questions = [questions[i] for i in valid_indices]
            
            for i, detections, caption in zip(valid_indices, batch_detections, global_captions):
                # Build context (simplified for batch)
                context = f"Image description: {caption}. "
                obj_counts = {}
                for obj in detections:
                    obj_counts[obj['class_name']] = obj_counts.get(obj['class_name'], 0) + 1
                obj_desc = ", ".join([f"{count} {name}" for name, count in obj_counts.items()])
                if obj_desc:
                    context += f" Objects detected: {obj_desc}."
                contexts.append(context)
            
            answers = self.llm_service.answer_question_batch(pil_images, valid_questions, contexts=contexts)
            
            # Reassemble results
            for i, idx in enumerate(valid_indices):
                results[idx] = {
                    'image_path': image_paths[idx],
                    'question': questions[idx],
                    'timestamp': datetime.now().isoformat(),
                    'features': {'objects': batch_detections[i]},
                    'caption': {'caption': global_captions[i]},
                    'answer': {'answer': answers[i]},
                    'summary': {
                        'key_findings': [
                            f"Detected {len(batch_detections[i])} objects",
                            f"Caption: {global_captions[i]}",
                            f"Answer: {answers[i]}"
                        ]
                    }
                }
                
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            for idx in valid_indices:
                results[idx] = {'error': f"Batch processing failed: {str(e)}"}
                
        return results

    def cleanup(self) -> None:
        """Clean up services."""
        # Services might implement their own cleanup if needed
        self.last_results = {}
        logger.info("MultimodalAI system cleaned up")