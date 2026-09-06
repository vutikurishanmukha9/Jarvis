"""
Vision Intelligence Bridge for Jarvis Super-Intelligence.
Integrates vision_engine capabilities (YOLOv8 object detection,
OCR text extraction, color analysis, and image metrics) into Jarvis.
"""

import io
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.tools import BaseTool, tool
from PIL import Image

logger = logging.getLogger(__name__)

# Ensure Jarvis project root and vendored PaddleOCR are discoverable on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPO_ROOT = PROJECT_ROOT.parent
_PADDLEOCR_DIR = REPO_ROOT / "PaddleOCR"
if _PADDLEOCR_DIR.exists() and str(_PADDLEOCR_DIR) not in sys.path:
    sys.path.insert(0, str(_PADDLEOCR_DIR))
_LANGCHAIN_PADDLE_DIR = _PADDLEOCR_DIR / "langchain-paddleocr"
if _LANGCHAIN_PADDLE_DIR.exists() and str(_LANGCHAIN_PADDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_LANGCHAIN_PADDLE_DIR))

# Store active images in memory for agent tools
_ACTIVE_IMAGES: Dict[str, Dict[str, Any]] = {}
_ANNOTATED_IMAGE_BUFFER: List[Tuple[str, Image.Image]] = []

_PADDLEOCR_AVAILABLE: Optional[bool] = None
_paddleocr_lock = threading.Lock()
_paddleocr_instance: Optional[Any] = None


def is_paddleocr_available() -> bool:
    """Return whether PaddleOCR engine is installed and ready."""
    global _PADDLEOCR_AVAILABLE
    if _PADDLEOCR_AVAILABLE is None:
        try:
            import ppocr  # noqa: F401

            _PADDLEOCR_AVAILABLE = True
        except ImportError:
            _PADDLEOCR_AVAILABLE = False
    return _PADDLEOCR_AVAILABLE


def extract_text_paddleocr(image_input: Any, lang: str = "en", use_angle_cls: bool = True) -> Optional[Dict[str, Any]]:
    """
    Extract structured scene text, bounding boxes, and confidence scores using PaddleOCR.
    image_input can be a numpy BGR array, PIL Image, or image bytes.
    Returns a dictionary with full_text, boxes, and metadata, or None on failure.
    """
    if not is_paddleocr_available():
        return None

    # 1. Custom runner hook (allows unit test mocking and custom pipeline injection)
    if hasattr(extract_text_paddleocr, "_runner"):
        runner = extract_text_paddleocr._runner
        if callable(runner):
            try:
                return runner(image_input, lang=lang, use_angle_cls=use_angle_cls)
            except Exception as e:
                logger.debug("PaddleOCR runner encountered error: %s", e)
                return None

    # 2. Native execution with thread safety
    with _paddleocr_lock:
        try:
            if isinstance(image_input, bytes):
                import cv2

                nparr = np.frombuffer(image_input, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif isinstance(image_input, Image.Image):
                import cv2

                img_rgb = np.array(image_input.convert("RGB"))
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            elif isinstance(image_input, np.ndarray):
                img_bgr = image_input
            else:
                return None

            if img_bgr is None or img_bgr.size == 0:
                return None

            from paddleocr import PaddleOCR

            global _paddleocr_instance
            if _paddleocr_instance is None:
                _paddleocr_instance = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

            ocr_results = _paddleocr_instance.ocr(img_bgr, cls=use_angle_cls)
            lines: List[str] = []
            boxes: List[Dict[str, Any]] = []

            if ocr_results and isinstance(ocr_results, list):
                page_res = ocr_results[0] if len(ocr_results) > 0 else []
                if page_res:
                    for item in page_res:
                        if item and len(item) == 2:
                            box_coords, (text, conf) = item
                            lines.append(text)
                            boxes.append(
                                {
                                    "text": text,
                                    "confidence": round(float(conf), 3),
                                    "polygon": box_coords,
                                }
                            )

            full_text = "\n".join(lines).strip()
            return {
                "full_text": full_text,
                "boxes": boxes,
                "line_count": len(lines),
                "engine": "paddleocr",
            }
        except Exception as e:
            logger.debug("PaddleOCR extraction encountered error: %s", e)
            return None


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
            "format": image_format,
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
        "quality": {},
    }

    try:
        # 1. YOLOv8 Object Detection
        try:
            import cv2
            from ultralytics import YOLO

            model = YOLO("yolov8n.pt")
            results = model(bgr_img, conf=0.35, verbose=False)[0]

            annotated_bgr = bgr_img.copy()
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls_id]

                analysis_results["objects"].append(
                    {"object": class_name, "confidence": round(conf, 3), "box": [int(x1), int(y1), int(x2), int(y2)]}
                )
                # Draw bounding box
                cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 240, 255), 2)
                cv2.putText(
                    annotated_bgr,
                    f"{class_name} {conf:.2f}",
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 240, 255),
                    2,
                )

            # Save annotated image to buffer
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            annotated_pil = Image.fromarray(annotated_rgb)
            _ANNOTATED_IMAGE_BUFFER.append((target_name, annotated_pil))
        except Exception as e:
            logger.warning(f"YOLO detection error: {str(e)}")
            analysis_results["objects_error"] = str(e)

        # 2. OCR Text Extraction (PaddleOCR primary SOTA, Tesseract fallback)
        paddle_res = extract_text_paddleocr(bgr_img)
        if paddle_res and paddle_res.get("full_text"):
            analysis_results["text_ocr"] = paddle_res["full_text"]
            analysis_results["text_boxes"] = paddle_res.get("boxes", [])
            analysis_results["ocr_engine"] = "paddleocr"
        else:
            try:
                import cv2
                import pytesseract

                gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
                ocr_text = pytesseract.image_to_string(gray)
                stripped_ocr = ocr_text.strip()
                if stripped_ocr:
                    analysis_results["text_ocr"] = stripped_ocr
                    analysis_results["ocr_engine"] = "tesseract"
                else:
                    analysis_results["text_ocr"] = ""
                    analysis_results["ocr_note"] = "No text detected by OCR engines."
            except Exception:
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
                "is_blurry": sharpness < 100.0,
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
            engine_str = f" ({data.get('ocr_engine', 'ocr')})" if data.get("ocr_engine") else ""
            out += f'- Extracted Text{engine_str}:\n"""\n{ocr_text}\n"""\n'

        colors = data.get("colors", [])
        if colors:
            color_str = ", ".join([f"{c['hex']} ({c['percentage']}%)" for c in colors])
            out += f"- Dominant Palette: {color_str}\n"

        quality = data.get("quality", {})
        if quality:
            out += f"- Image Quality: Sharpness {quality.get('sharpness_score')}, Brightness {quality.get('brightness')}, Blurry: {quality.get('is_blurry')}\n"

        results.append(out)

    return "\n\n".join(results)


@tool
def extract_scene_text_ocr(image_name: Optional[str] = None) -> str:
    """
    Extract scene text, fine print, receipts, labels, and structured text boxes from an uploaded image using PaddleOCR.
    If image_name is not provided, extracts text from the most recently uploaded image.
    Returns the recognized text, line count, bounding boxes, and recognition confidence scores.
    """
    if not _ACTIVE_IMAGES:
        return "No images have been uploaded yet. Please upload an image first."

    target = image_name if (image_name and image_name in _ACTIVE_IMAGES) else list(_ACTIVE_IMAGES.keys())[-1]
    img_data = _ACTIVE_IMAGES[target]
    res = extract_text_paddleocr(img_data["bgr"])
    if not res:
        # Fallback to Tesseract
        try:
            import cv2
            import pytesseract

            gray = cv2.cvtColor(img_data["bgr"], cv2.COLOR_BGR2GRAY)
            tess_text = pytesseract.image_to_string(gray).strip()
            if tess_text:
                return f"Text extracted from {target} via Tesseract OCR fallback:\n{tess_text}"
            return f"No text could be extracted from {target} using available OCR engines."
        except Exception as e:
            return f"OCR extraction failed for {target}: {str(e)}"

    out = [f"=== High-Precision Text Extraction for {target} (Engine: {res.get('engine', 'paddleocr')}) ==="]
    out.append(f"Total Detected Text Lines: {res.get('line_count', 0)}")
    out.append("Recognized Content:")
    out.append(res.get("full_text", ""))
    if res.get("boxes"):
        out.append(f"\nDetailed Bounding Boxes and Confidence:\n{res.get('boxes')}")
    return "\n".join(out)


def get_vision_tools() -> List[BaseTool]:
    """Retrieve vision intelligence tools."""
    return [analyze_uploaded_images, extract_scene_text_ocr]
