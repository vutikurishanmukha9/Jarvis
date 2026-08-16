"""
Language model integration module for multimodal AI system.

This module provides the LLMProcessor class that handles:
- Image captioning using BLIP models
- Vision-language question answering
- Text generation based on image content
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import logging
from typing import Optional, Union, List, Dict, Any
import os

# Set up logging
logger = logging.getLogger(__name__)


class LLMProcessor:
    """
    Language model processor for vision-language tasks using BLIP models.

    This class provides functionality for:
    - Image captioning (conditional and unconditional)
    - Visual question answering
    - Text generation based on visual content
    """

    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base",
                 device: str = "auto", torch_dtype: str = "auto"):
        """
        Initialize the LLM processor with a BLIP model.

        Args:
            model_id (str): Hugging Face model identifier
            device (str): Device to run model on ('auto', 'cpu', 'cuda')
            torch_dtype (str): Torch data type ('auto', 'float16', 'float32')
        """
        self.model_id = model_id
        self.device = self._setup_device(device)
        self.torch_dtype = self._setup_dtype(torch_dtype)

        # Initialize model and processor
        self.processor = None
        self.model = None
        self.model_loaded = False

        # Load the model
        self._load_model()

        logger.info(f"LLMProcessor initialized with model: {model_id}")
        logger.info(f"Using device: {self.device}, dtype: {self.torch_dtype}")

    def _setup_device(self, device: str) -> str:
        """Setup the computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _setup_dtype(self, torch_dtype: str) -> torch.dtype:
        """Setup the torch data type."""
        if torch_dtype == "auto":
            if self.device == "cuda":
                return torch.float16  # Use half precision on GPU for efficiency
            else:
                return torch.float32  # Use full precision on CPU
        elif torch_dtype == "float16":
            return torch.float16
        elif torch_dtype == "float32":
            return torch.float32
        else:
            return torch.float32

    def _load_model(self) -> None:
        """Load the BLIP model and processor."""
        try:
            logger.info(f"Loading BLIP processor from {self.model_id}...")
            self.processor = BlipProcessor.from_pretrained(self.model_id)

            logger.info(f"Loading BLIP model from {self.model_id}...")
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype
            )

            # Move model to specified device
            self.model = self.model.to(self.device)

            # Set model to evaluation mode
            self.model.eval()

            self.model_loaded = True
            logger.info("Successfully loaded BLIP model and processor")

        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {str(e)}")
            self.model_loaded = False

    def caption_image(self, image: Union[str, Image.Image], 
                     conditional_text: Optional[str] = None,
                     max_length: int = 50,
                     num_beams: int = 5,
                     temperature: float = 1.0) -> Dict[str, Any]:
        """
        Generate caption for an image using BLIP model.

        Args:
            image (Union[str, Image.Image]): Image path or PIL Image object
            conditional_text (Optional[str]): Conditional text for guided captioning
            max_length (int): Maximum length of generated caption
            num_beams (int): Number of beams for beam search
            temperature (float): Sampling temperature for generation

        Returns:
            Dict[str, Any]: Caption results with metadata
        """
        if not self.model_loaded:
            logger.error("Model not loaded")
            return {
                'caption': '',
                'confidence': 0.0,
                'error': 'Model not loaded'
            }

        try:
            # Load and prepare image
            if isinstance(image, str):
                if not os.path.exists(image):
                    logger.error(f"Image file not found: {image}")
                    return {
                        'caption': '',
                        'confidence': 0.0,
                        'error': f'Image file not found: {image}'
                    }
                pil_image = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                pil_image = image.convert('RGB')
            else:
                logger.error("Invalid image type")
                return {
                    'caption': '',
                    'confidence': 0.0,
                    'error': 'Invalid image type'
                }

            # Process inputs
            if conditional_text:
                # Conditional image captioning
                inputs = self.processor(
                    images=pil_image,
                    text=conditional_text,
                    return_tensors="pt"
                ).to(self.device, dtype=self.torch_dtype)

                caption_type = "conditional"
                logger.info(f"Generating conditional caption with text: '{conditional_text}'")
            else:
                # Unconditional image captioning
                inputs = self.processor(
                    images=pil_image,
                    return_tensors="pt"
                ).to(self.device, dtype=self.torch_dtype)

                caption_type = "unconditional"
                logger.info("Generating unconditional caption")

            # Generate caption
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Decode the generated caption
            sequences = output.sequences if hasattr(output, 'sequences') else output
            caption = self.processor.decode(sequences[0], skip_special_tokens=True)

            # Calculate confidence based on generation quality metrics
            # Use sequence length ratio and beam search effectiveness
            caption_words = len(caption.split())
            if caption_words > 0:
                # Confidence based on: reasonable length (not too short or truncated)
                length_score = min(1.0, caption_words / 5) * min(1.0, max_length / (caption_words + 1))
                # Normalize to reasonable range
                confidence = min(0.95, max(0.3, length_score * 0.7 + 0.3))
            else:
                confidence = 0.1

            result = {
                'caption': caption,
                'confidence': round(confidence, 3),
                'caption_type': caption_type,
                'conditional_text': conditional_text,
                'generation_params': {
                    'max_length': max_length,
                    'num_beams': num_beams,
                    'temperature': temperature
                },
                'word_count': len(caption.split()),
                'character_count': len(caption)
            }

            logger.info(f"Generated {caption_type} caption: '{caption}'")
            return result

        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return {
                'caption': '',
                'confidence': 0.0,
                'error': str(e)
            }

    def answer_question(self, caption: str, question: str,
                       max_length: int = 100,
                       num_beams: int = 5,
                       temperature: float = 1.0) -> Dict[str, Any]:
        """
        Answer a question based on image caption using text generation.

        Args:
            caption (str): Image caption or description
            question (str): Question to answer
            max_length (int): Maximum length of generated answer
            num_beams (int): Number of beams for beam search  
            temperature (float): Sampling temperature for generation

        Returns:
            Dict[str, Any]: Answer with metadata
        """
        if not self.model_loaded:
            logger.error("Model not loaded")
            return {
                'answer': '',
                'confidence': 0.0,
                'error': 'Model not loaded'
            }

        try:
            # Create a context prompt combining caption and question
            context = f"Image description: {caption}\n\nQuestion: {question}\n\nAnswer:"

            # Tokenize the input
            inputs = self.processor.tokenizer(
                context,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Generate answer
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=inputs['input_ids'].shape[1] + max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

            # Decode the answer (remove the input context)
            full_response = self.processor.tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract just the answer part
            if "Answer:" in full_response:
                answer = full_response.split("Answer:")[-1].strip()
            else:
                answer = full_response[len(context):].strip()

            # Simple confidence estimation
            confidence = min(1.0, max(0.1, 1.0 - (len(answer.split()) / max_length)))

            result = {
                'answer': answer,
                'confidence': round(confidence, 3),
                'question': question,
                'context_caption': caption,
                'generation_params': {
                    'max_length': max_length,
                    'num_beams': num_beams,
                    'temperature': temperature
                },
                'word_count': len(answer.split()),
                'character_count': len(answer)
            }

            logger.info(f"Generated answer for question '{question}': '{answer}'")
            return result

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                'answer': '',
                'confidence': 0.0,
                'error': str(e)
            }

    def batch_caption_images(self, images: List[Union[str, Image.Image]],
                           conditional_texts: Optional[List[str]] = None,
                           **generation_kwargs) -> List[Dict[str, Any]]:
        """
        Generate captions for multiple images in batch.

        Args:
            images (List[Union[str, Image.Image]]): List of image paths or PIL Images
            conditional_texts (Optional[List[str]]): Optional conditional texts for each image
            **generation_kwargs: Additional generation parameters

        Returns:
            List[Dict[str, Any]]: List of caption results
        """
        if not self.model_loaded:
            logger.error("Model not loaded")
            return []

        results = []

        # Ensure conditional_texts has same length as images if provided
        if conditional_texts is None:
            conditional_texts = [None] * len(images)
        elif len(conditional_texts) != len(images):
            logger.warning("Conditional texts length doesn't match images length")
            conditional_texts = conditional_texts[:len(images)] + [None] * (len(images) - len(conditional_texts))

        for i, (image, conditional_text) in enumerate(zip(images, conditional_texts)):
            logger.info(f"Processing image {i+1}/{len(images)}")

            result = self.caption_image(
                image=image,
                conditional_text=conditional_text,
                **generation_kwargs
            )

            result['batch_index'] = i
            results.append(result)

        logger.info(f"Completed batch captioning of {len(images)} images")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dict[str, Any]: Model information and configuration
        """
        if not self.model_loaded:
            return {'error': 'Model not loaded'}

        try:
            info = {
                'model_id': self.model_id,
                'device': str(self.device),
                'torch_dtype': str(self.torch_dtype),
                'model_loaded': self.model_loaded,
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            }

            # Add tokenizer info
            if hasattr(self.processor, 'tokenizer'):
                info['vocab_size'] = self.processor.tokenizer.vocab_size
                info['max_position_embeddings'] = getattr(
                    self.processor.tokenizer, 'model_max_length', 'Unknown'
                )

            return info

        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {'error': str(e)}

    def cleanup(self) -> None:
        """Clean up resources and free memory."""
        try:
            if self.model is not None:
                del self.model
            if self.processor is not None:
                del self.processor

            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.model_loaded = False
            logger.info("Successfully cleaned up LLM processor resources")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
