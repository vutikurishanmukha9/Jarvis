import logging
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from typing import Optional, List
from .interfaces import LLMService

logger = logging.getLogger(__name__)

class LocalBlipService(LLMService):
    """Local implementation of LLMService using Salesforce BLIP."""
    
    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base", device: str = "auto"):
        self.model_id = model_id
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.processor = None
        self.model = None
        self._load_model()
        
    def _load_model(self):
        try:
            logger.info(f"Loading BLIP model {self.model_id} to {self.device}")
            self.processor = BlipProcessor.from_pretrained(self.model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
            logger.info("BLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {str(e)}")
            raise

    def caption_image(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        try:
            inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
            
            out = self.model.generate(
                **inputs, 
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return ""

    def answer_question(self, image: Image.Image, question: str, context: str = "") -> str:
        """
        Note: The base BLIP model is better at captioning. 
        For VQA, we leverage the captioning capability by prompting or use the generated context.
        However, for 'True' VQA we might want to swap this model for `blip-vqa-base`.
        For now, we use the conditional generation to answer.
        """
        try:
            # Simple VQA implementation using conditional generation
            # We feed the image and the question as text
            text_input = f"Question: {question} Answer:"
            
            inputs = self.processor(image, text=text_input, return_tensors="pt").to(self.device)
            
            out = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
            
            answer = self.processor.decode(out[0], skip_special_tokens=True)
            return answer
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return ""

    def caption_image_batch(self, images: List[Image.Image], prompt: Optional[str] = None) -> List[str]:
        try:
            # Batch processing
            text = [prompt] * len(images) if prompt else None
            inputs = self.processor(images=images, text=text, return_tensors="pt", padding=True).to(self.device)
            
            out = self.model.generate(
                **inputs, 
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
            
            captions = self.processor.batch_decode(out, skip_special_tokens=True)
            return captions
        except Exception as e:
            logger.error(f"Error generating batch captions: {str(e)}")
            return [""] * len(images)

    def answer_question_batch(self, images: List[Image.Image], questions: List[str], contexts: Optional[List[str]] = None) -> List[str]:
        try:
            text_inputs = [f"Question: {q} Answer:" for q in questions]
            
            inputs = self.processor(images=images, text=text_inputs, return_tensors="pt", padding=True).to(self.device)
            
            out = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
            
            answers = self.processor.batch_decode(out, skip_special_tokens=True)
            return answers
        except Exception as e:
            logger.error(f"Error answering batch questions: {str(e)}")
            return [""] * len(images)

class LlaVAService(LLMService):
    """
    State-of-the-art VLM implementation using LlaVA (Large Language-and-Vision Assistant).
    Provides 10x improvement in reasoning and detail over BLIP.
    """
    
    def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf", device: str = "auto", load_in_4bit: bool = True):
        self.model_id = model_id
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.load_in_4bit = load_in_4bit and self.device == "cuda"
        self.processor = None
        self.model = None
        self._load_model()
        
    def _load_model(self):
        try:
            logger.info(f"Loading LlaVA model {self.model_id} (4-bit={self.load_in_4bit})...")
            from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig, AutoProcessor

            quantization_config = None
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )

            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Load model
            if self.load_in_4bit:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_id, 
                    quantization_config=quantization_config, 
                    device_map="auto"
                )
            else:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_id
                ).to(self.device)
                
            logger.info("LlaVA model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LlaVA model: {str(e)}")
            raise

    def caption_image(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """
        Generate a caption or description using LlaVA.
        """
        # Default prompt for captioning if none provided
        if not prompt:
            prompt = "Describe this image in detail." # LlaVA needs a conversation prompt

        return self.answer_question(image, prompt)

    def answer_question(self, image: Image.Image, question: str, context: str = "") -> str:
        """
        Answer a question about the image.
        LlaVA expects a conversation format: USER: <image>\n<prompt>\nASSISTANT:
        """
        try:
            # Construct LlaVA prompt format
            # Note: The context arg is used by the system to pass prior details, 
            # but LlaVA is powerful enough to see them. We can prepend context to the question.
            
            full_prompt = f"USER: <image>\n"
            if context:
                 full_prompt += f"Context: {context}\n"
            full_prompt += f"{question}\nASSISTANT:"

            inputs = self.processor(text=full_prompt, images=image, return_tensors="pt").to(self.model.device)

            # Generate
            generate_ids = self.model.generate(
                **inputs, 
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            # Decode output
            output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            # Extract just the assistant's response
            # Format is usually "USER: ... ASSISTANT: <answer>"
            if "ASSISTANT:" in output:
                answer = output.split("ASSISTANT:")[-1].strip()
            else:
                answer = output

            return answer
        except Exception as e:
            logger.error(f"Error responding with LlaVA: {str(e)}")
            return "Error generating response."

    def caption_image_batch(self, images: List[Image.Image], prompt: Optional[str] = None) -> List[str]:
        # Reuse answer_question_batch logic
        if not prompt:
            prompt = "Describe this image in detail."
        questions = [prompt] * len(images)
        return self.answer_question_batch(images, questions)

    def answer_question_batch(self, images: List[Image.Image], questions: List[str], contexts: Optional[List[str]] = None) -> List[str]:
        try:
            if contexts is None:
                contexts = [""] * len(images)
                
            prompts = []
            for q, c in zip(questions, contexts):
                full_prompt = f"USER: <image>\n"
                if c:
                    full_prompt += f"Context: {c}\n"
                full_prompt += f"{q}\nASSISTANT:"
                prompts.append(full_prompt)
                
            inputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt").to(self.model.device)
            
            generate_ids = self.model.generate(
                **inputs, 
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            # Batch decode
            outputs = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            answers = []
            for output in outputs:
                if "ASSISTANT:" in output:
                    answers.append(output.split("ASSISTANT:")[-1].strip())
                else:
                    answers.append(output)
                    
            return answers
        except Exception as e:
             logger.error(f"Error answering batch questions with LlaVA: {str(e)}")
             return ["Error"] * len(images)
