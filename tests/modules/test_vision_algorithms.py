"""
Tests for Vision Algorithms: Laplacian variance sharpness, blur classification, and K-Means color hex conversion.
"""

import io

import numpy as np
import pytest
from PIL import Image

from src.modules.vision.vision_bridge import analyze_image_deep, clear_active_images, register_uploaded_image
from tests.conftest import MockUploadedFile


@pytest.fixture(autouse=True)
def reset_images():
    clear_active_images()
    yield
    clear_active_images()


def test_vision_deep_analysis_structure(sample_image_file):
    """Verify deep vision analysis returns dimensions, colors, and quality metrics."""
    register_uploaded_image(sample_image_file)
    analysis = analyze_image_deep("test_frame.jpg")

    assert "filename" in analysis
    assert "dimensions" in analysis
    assert "quality" in analysis
    assert "colors" in analysis

    quality = analysis["quality"]
    assert "brightness" in quality
    assert "contrast" in quality
    assert "sharpness_score" in quality
    assert "is_blurry" in quality


def test_vision_kmeans_color_percentages(sample_image_file):
    """Verify dominant color percentages sum approximately to 100%."""
    register_uploaded_image(sample_image_file)
    analysis = analyze_image_deep("test_frame.jpg")
    colors = analysis["colors"]

    assert len(colors) == 4
    total_pct = sum(c["percentage"] for c in colors)
    assert total_pct == pytest.approx(100.0, abs=1.0)
    for c in colors:
        assert c["hex"].startswith("#")
        assert len(c["hex"]) == 7


def test_vision_quality_sharpness_calculation():
    """Verify sharp high-contrast image produces high sharpness score."""
    # Create checkerboard high-frequency pattern
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[::2, ::2] = 255
    arr[1::2, 1::2] = 255

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    mock_file = MockUploadedFile("sharp_pattern.png", buf.getvalue())

    register_uploaded_image(mock_file)
    analysis = analyze_image_deep("sharp_pattern.png")
    quality = analysis["quality"]
    assert quality["sharpness_score"] > 500
    assert quality["is_blurry"] is False


def test_vision_no_images_uploaded():
    """Verify analyze_image_deep returns clean error message when no images are active."""
    clear_active_images()
    analysis = analyze_image_deep()
    assert "error" in analysis
