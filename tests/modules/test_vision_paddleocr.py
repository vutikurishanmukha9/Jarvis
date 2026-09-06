"""
Unit and integration tests for PaddleOCR vision intelligence engine in B.H.A.I.R.A.V.A.
Verifies scene text extraction, bounding box detection, multi-tier fallback mechanisms,
LangChain tool invocation, and thread safety.
"""

import io
import threading
from typing import Any, Dict, Optional
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.modules.vision.vision_bridge import (
    analyze_image_deep,
    clear_active_images,
    extract_scene_text_ocr,
    extract_text_paddleocr,
    get_vision_tools,
    is_paddleocr_available,
    register_uploaded_image,
)
from src.tools import is_paddleocr_available as tools_is_paddleocr_available


@pytest.fixture(autouse=True)
def reset_vision_state():
    """Ensure image buffer and custom runners are reset around every test."""
    clear_active_images()
    if hasattr(extract_text_paddleocr, "_runner"):
        delattr(extract_text_paddleocr, "_runner")
    yield
    clear_active_images()
    if hasattr(extract_text_paddleocr, "_runner"):
        delattr(extract_text_paddleocr, "_runner")


def test_is_paddleocr_available() -> None:
    """Validate that is_paddleocr_available returns a valid boolean contract."""
    avail = is_paddleocr_available()
    assert isinstance(avail, bool)
    assert tools_is_paddleocr_available() == avail


def test_extract_text_paddleocr_unsupported_input() -> None:
    """Validate that extract_text_paddleocr handles invalid inputs gracefully."""
    assert extract_text_paddleocr(None) is None
    assert extract_text_paddleocr(12345) is None
    assert extract_text_paddleocr("not_an_image") is None


def test_extract_text_paddleocr_with_mock_runner() -> None:
    """Validate that extract_text_paddleocr formats structured OCR outputs correctly."""
    sample_response: Dict[str, Any] = {
        "full_text": "B.H.A.I.R.A.V.A. Vision Engine\nDeep Text Detection",
        "boxes": [
            {
                "text": "B.H.A.I.R.A.V.A. Vision Engine",
                "confidence": 0.985,
                "polygon": [[10, 10], [200, 10], [200, 40], [10, 40]],
            },
            {
                "text": "Deep Text Detection",
                "confidence": 0.962,
                "polygon": [[10, 45], [180, 45], [180, 75], [10, 75]],
            },
        ],
        "line_count": 2,
        "engine": "paddleocr",
    }

    def mock_runner(image_input: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return sample_response

    extract_text_paddleocr._runner = mock_runner  # type: ignore[attr-defined]

    with patch("src.modules.vision.vision_bridge.is_paddleocr_available", return_value=True):
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        res = extract_text_paddleocr(img)

        assert res is not None
        assert res["full_text"] == "B.H.A.I.R.A.V.A. Vision Engine\nDeep Text Detection"
        assert res["line_count"] == 2
        assert res["engine"] == "paddleocr"
        assert len(res["boxes"]) == 2
        assert res["boxes"][0]["confidence"] == 0.985


def test_extract_text_paddleocr_runner_exception_handling() -> None:
    """Validate that runner exceptions do not crash and return None gracefully."""

    def failing_runner(image_input: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        raise RuntimeError("GPU memory exhausted during inference")

    extract_text_paddleocr._runner = failing_runner  # type: ignore[attr-defined]

    with patch("src.modules.vision.vision_bridge.is_paddleocr_available", return_value=True):
        img = Image.new("RGB", (50, 50), color=(0, 0, 0))
        res = extract_text_paddleocr(img)
        assert res is None


def test_extract_text_paddleocr_input_formats() -> None:
    """Validate that extract_text_paddleocr accepts bytes, PIL Image, and numpy arrays."""
    invoked_types = []

    def tracking_runner(image_input: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        invoked_types.append(type(image_input))
        return {"full_text": "Sample", "boxes": [], "line_count": 1, "engine": "paddleocr"}

    extract_text_paddleocr._runner = tracking_runner  # type: ignore[attr-defined]

    with patch("src.modules.vision.vision_bridge.is_paddleocr_available", return_value=True):
        # 1. PIL Image
        pil_img = Image.new("RGB", (64, 64), color=(200, 200, 200))
        extract_text_paddleocr(pil_img)

        # 2. Numpy Array
        np_arr = np.zeros((64, 64, 3), dtype=np.uint8)
        extract_text_paddleocr(np_arr)

        # 3. Raw Bytes
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        extract_text_paddleocr(buf.getvalue())

    assert len(invoked_types) == 3


def test_extract_text_paddleocr_thread_safety() -> None:
    """Validate that concurrent invocations across worker threads execute safely."""
    results: list = []

    def mock_runner(image_input: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return {"full_text": "Thread Text", "boxes": [], "line_count": 1, "engine": "paddleocr"}

    extract_text_paddleocr._runner = mock_runner  # type: ignore[attr-defined]

    def worker():
        with patch("src.modules.vision.vision_bridge.is_paddleocr_available", return_value=True):
            img = Image.new("RGB", (32, 32), color=(128, 128, 128))
            res = extract_text_paddleocr(img)
            results.append(res)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    assert all(r is not None and r["full_text"] == "Thread Text" for r in results)


def test_analyze_image_deep_paddleocr_priority(sample_image_file) -> None:
    """Validate that analyze_image_deep uses PaddleOCR as primary engine when text is detected."""
    sample_response = {
        "full_text": "Invoice #9821\nTotal: $420.00",
        "boxes": [{"text": "Invoice #9821", "confidence": 0.99}],
        "line_count": 2,
        "engine": "paddleocr",
    }

    def mock_runner(image_input: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return sample_response

    extract_text_paddleocr._runner = mock_runner  # type: ignore[attr-defined]

    with patch("src.modules.vision.vision_bridge.is_paddleocr_available", return_value=True):
        register_uploaded_image(sample_image_file)
        analysis = analyze_image_deep("test_frame.jpg")

        assert analysis["text_ocr"] == "Invoice #9821\nTotal: $420.00"
        assert analysis["ocr_engine"] == "paddleocr"
        assert len(analysis["text_boxes"]) == 1
        assert analysis["filename"] == "test_frame.jpg"


def test_analyze_image_deep_paddleocr_fallback_to_tesseract(sample_image_file) -> None:
    """Validate that analyze_image_deep falls back cleanly if PaddleOCR returns None."""

    def mock_runner(image_input: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return None

    extract_text_paddleocr._runner = mock_runner  # type: ignore[attr-defined]

    register_uploaded_image(sample_image_file)
    analysis = analyze_image_deep("test_frame.jpg")

    assert "text_ocr" in analysis
    # Even if OCR finds no text in blank test frame, dimensions and quality are still analyzed
    assert analysis["dimensions"] == "320x240"
    assert "quality" in analysis


def test_extract_scene_text_ocr_tool_no_images() -> None:
    """Validate extract_scene_text_ocr tool returns prompt when no images are uploaded."""
    clear_active_images()
    tool_output = extract_scene_text_ocr.invoke({})
    assert "No images have been uploaded yet" in tool_output


def test_extract_scene_text_ocr_tool_with_image(sample_image_file) -> None:
    """Validate extract_scene_text_ocr tool returns structured report with line count and boxes."""
    sample_response = {
        "full_text": "STORE RECEIPT\nITEM 1: 10.00\nTHANK YOU",
        "boxes": [
            {"text": "STORE RECEIPT", "confidence": 0.97, "polygon": [[0, 0], [50, 0], [50, 10], [0, 10]]},
            {"text": "ITEM 1: 10.00", "confidence": 0.95, "polygon": [[0, 15], [50, 15], [50, 25], [0, 25]]},
        ],
        "line_count": 3,
        "engine": "paddleocr",
    }

    def mock_runner(image_input: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return sample_response

    extract_text_paddleocr._runner = mock_runner  # type: ignore[attr-defined]

    with patch("src.modules.vision.vision_bridge.is_paddleocr_available", return_value=True):
        register_uploaded_image(sample_image_file)
        tool_output = extract_scene_text_ocr.invoke({"image_name": "test_frame.jpg"})

        assert "=== High-Precision Text Extraction for test_frame.jpg" in tool_output
        assert "Total Detected Text Lines: 3" in tool_output
        assert "STORE RECEIPT" in tool_output
        assert "ITEM 1: 10.00" in tool_output
        assert "Detailed Bounding Boxes and Confidence:" in tool_output


def test_vision_tools_includes_paddleocr() -> None:
    """Validate that get_vision_tools exposes both analyze_uploaded_images and extract_scene_text_ocr."""
    tools = get_vision_tools()
    tool_map = {t.name: t for t in tools}

    assert "analyze_uploaded_images" in tool_map
    assert "extract_scene_text_ocr" in tool_map

    scene_tool = tool_map["extract_scene_text_ocr"]
    assert "scene text" in scene_tool.description.lower() or "paddleocr" in scene_tool.description.lower()
