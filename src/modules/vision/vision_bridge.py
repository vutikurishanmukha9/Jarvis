"""
Vision Intelligence Bridge for Jarvis Super-Intelligence.
Integrates vision_engine capabilities (YOLOv8 object detection,
OCR text extraction, color analysis, and image metrics) into Jarvis.
"""

import io
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image
from langchain_core.tools import tool, BaseTool

logger = logging.getLogger(__name__)

# Ensure Jarvis project root is at index 0 of sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Store active images in memory for agent tools
_ACTIVE_IMAGES: Dict[str, Dict[str, Any]] = {}
_ANNOTATED_IMAGE_BUFFER: List[Tuple[str, Image.Image]] = []

def get_and_clear_annotated_images() -> List[Tuple[str, Image.Image]]:
    """Retrieve and clear annotated images buffer for UI rendering."""
    global _ANNOTATED_IMAGE_BUFFER
    imgs = list(_ANNOTATED_IMAGE_BUFFER)
    _ANNOTATED_IMAGE_BUFFER.clear()
    return imgs

def register_uploaded_image(file: Any) -> Dict[str, Any]:
    """Register and preprocess an uploaded image file."""
    global _ACTIVE_IMAGES
    filename = file.name
    try:
        content = file.getvalue()
        source_image = Image.open(io.BytesIO(content))
        image_format = source_image.format or Path(filename).suffix.lstrip(".").upper()
        pil_img = source_image.convert("RGB")
        img_np = np.array(pil_img)
        # Convert RGB to BGR for OpenCV
        import cv2
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        _ACTIVE_IMAGES[filename] = {
            "pil": pil_img,
            "bgr": img_bgr,
            "rgb": img_np,
            "size": pil_img.size,
            "format": image_format
        }
        logger.info(f"Registered active image: {filename} ({pil_img.size})")
        return {"status": "success", "filename": filename, "dimensions": pil_img.size}
    except Exception as e:
        logger.error(f"Failed to register image {filename}: {str(e)}", exc_info=True)
        return {"status": "error", "filename": filename, "error": str(e)}

def clear_active_images():
    """Clear all registered images."""
    global _ACTIVE_IMAGES
    _ACTIVE_IMAGES.clear()

def analyze_image_deep(filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform complete multimodal vision analysis on an active image:
    1. YOLOv8 object detection
    2. OCR text extraction
    3. Color extraction (K-Means)
    4. Image quality statistics
    """
    global _ACTIVE_IMAGES, _ANNOTATED_IMAGE_BUFFER
    if not _ACTIVE_IMAGES:
        return {"error": "No active images uploaded to analyze."}

    target_name = filename if (filename and filename in _ACTIVE_IMAGES) else list(_ACTIVE_IMAGES.keys())[0]
    img_data = _ACTIVE_IMAGES[target_name]
    bgr_img = img_data["bgr"]
    pil_img = img_data["pil"]

    analysis_results: Dict[str, Any] = {
        "filename": target_name,
        "dimensions": f"{pil_img.width}x{pil_img.height}",
        "objects": [],
        "text_ocr": "",
        "colors": [],
        "quality": {}
    }

    try:
        # 1. YOLOv8 Object Detection
        try:
            from ultralytics import YOLO
            import cv2
            model = YOLO("yolov8n.pt")
            results = model(bgr_img, conf=0.35, verbose=False)[0]
            
            annotated_bgr = bgr_img.copy()
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls_id]
                
                analysis_results["objects"].append({
                    "object": class_name,
                    "confidence": round(conf, 3),
                    "box": [int(x1), int(y1), int(x2), int(y2)]
                })
                # Draw bounding box
                cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 240, 255), 2)
                cv2.putText(annotated_bgr, f"{class_name} {conf:.2f}", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 2)

            # Save annotated image to buffer
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            annotated_pil = Image.fromarray(annotated_rgb)
            _ANNOTATED_IMAGE_BUFFER.append((target_name, annotated_pil))
        except Exception as e:
            logger.warning(f"YOLO detection error: {str(e)}")
            analysis_results["objects_error"] = str(e)

        # 2. OCR Text Extraction
        try:
            import pytesseract
            import cv2
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            ocr_text = pytesseract.image_to_string(gray)
            analysis_results["text_ocr"] = ocr_text.strip()
        except Exception as e:
            analysis_results["ocr_note"] = "Tesseract OCR not configured or no text detected."

        # 3. Dominant Color Extraction (K-Means)
        try:
            import cv2
            rgb_data = img_data["rgb"].reshape((-1, 3)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(rgb_data, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            labels = labels.flatten()
            for i, center in enumerate(centers):
                pct = round(float(np.sum(labels == i) / len(labels) * 100), 1)
                hex_code = f"#{center[0]:02x}{center[1]:02x}{center[2]:02x}"
                analysis_results["colors"].append({"hex": hex_code, "percentage": pct})
        except Exception as e:
            logger.warning(f"Color analysis error: {str(e)}")

        # 4. Quality & Sharpness
        try:
            import cv2
            gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            analysis_results["quality"] = {
                "brightness": round(brightness, 1),
                "contrast": round(contrast, 1),
                "sharpness_score": round(sharpness, 1),
                "is_blurry": sharpness < 100.0
            }
        except Exception as e:
            logger.warning(f"Image quality stats error: {str(e)}")

    except Exception as e:
        logger.error(f"Complete vision analysis failed: {str(e)}", exc_info=True)
        analysis_results["error"] = str(e)

    return analysis_results

@tool
def analyze_uploaded_images(query: str = "Perform comprehensive vision analysis on uploaded images") -> str:
    """
    Analyzes all uploaded images to detect objects (YOLOv8), extract printed or handwritten text (OCR),
    identify colors, assess visual quality, and generate annotated bounding box figures.
    Use this tool whenever the user asks about an uploaded image, screenshot, chart, diagram, or photo.
    """
    if not _ACTIVE_IMAGES:
        return "No images have been uploaded yet. Please upload an image (PNG, JPG, WEBP) to inspect it."

    results = []
    for filename in _ACTIVE_IMAGES.keys():
        data = analyze_image_deep(filename)
        out = f"=== Vision Analysis for {filename} ===\n"
        out += f"- Dimensions: {data.get('dimensions', 'N/A')}\n"
        
        objects = data.get("objects", [])
        if objects:
            counts: Dict[str, int] = {}
            for obj in objects:
                counts[obj["object"]] = counts.get(obj["object"], 0) + 1
            obj_str = ", ".join([f"{k}: {v}" for k, v in counts.items()])
            out += f"- Detected Objects ({len(objects)} total): {obj_str}\n"
            out += f"- Detailed Bounding Boxes: {objects}\n"
        else:
            out += "- Detected Objects: None detected\n"

        ocr_text = data.get("text_ocr", "")
        if ocr_text:
            out += f"- Extracted OCR Text:\n\"\"\"\n{ocr_text}\n\"\"\"\n"
        
        colors = data.get("colors", [])
        if colors:
            color_str = ", ".join([f"{c['hex']} ({c['percentage']}%)" for c in colors])
            out += f"- Dominant Palette: {color_str}\n"
            
        quality = data.get("quality", {})
        if quality:
            out += f"- Image Quality: Sharpness {quality.get('sharpness_score')}, Brightness {quality.get('brightness')}, Blurry: {quality.get('is_blurry')}\n"
            
        results.append(out)

    return "\n\n".join(results)

def get_vision_tools() -> List[BaseTool]:
    """Retrieve vision intelligence tools."""
    return [analyze_uploaded_images]
