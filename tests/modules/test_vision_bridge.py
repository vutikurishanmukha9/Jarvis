"""
Tests for Vision Bridge: image registration, dimension extraction, memory cache, and tool retrieval.
"""

import io

from PIL import Image

from src.modules.vision import _ACTIVE_IMAGES, clear_active_images, get_vision_tools, register_uploaded_image
from tests.conftest import MockUploadedFile


def test_vision_image_registration_jpg(sample_image_file):
    """Verify registering a JPEG image populates _ACTIVE_IMAGES and returns dimensions."""
    clear_active_images()
    assert len(_ACTIVE_IMAGES) == 0

    res = register_uploaded_image(sample_image_file)
    assert res["status"] == "success"
    assert res["filename"] == "test_frame.jpg"
    assert res["dimensions"] == (320, 240)
    assert "test_frame.jpg" in _ACTIVE_IMAGES


def test_vision_image_registration_png():
    """Verify registering a PNG image with alpha transparency."""
    img = Image.new("RGBA", (200, 150), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_file = MockUploadedFile("overlay.png", buf.getvalue())

    res = register_uploaded_image(png_file)
    assert res["status"] == "success"
    assert res["dimensions"] == (200, 150)
    assert "overlay.png" in _ACTIVE_IMAGES


def test_vision_clear_active_images():
    """Verify clear_active_images empties the image cache."""
    assert len(_ACTIVE_IMAGES) > 0
    clear_active_images()
    assert len(_ACTIVE_IMAGES) == 0


def test_vision_tools_suite_registration():
    """Verify vision tools are registered in module suite."""
    tools = get_vision_tools()
    assert len(tools) == 1
    assert tools[0].name == "analyze_uploaded_images"
