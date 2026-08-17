"""
Unit tests for Vision Intelligence Sub-Agent tools.
"""

import io

from PIL import Image

from src.modules.vision import (
    analyze_image_deep,
    clear_active_images,
    get_vision_tools,
    register_uploaded_image,
)


class MockImageFile:
    def __init__(self, name: str) -> None:
        self.name = name
        img = Image.new("RGB", (100, 100), color=(73, 109, 137))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        self._data = buf.getvalue()

    def getvalue(self) -> bytes:
        return self._data


def test_vision_tools_collection() -> None:
    """Ensure get_vision_tools returns functional LangChain tools."""
    tools = get_vision_tools()
    assert len(tools) >= 1
    tool_names = [t.name for t in tools]
    assert "analyze_uploaded_images" in tool_names


def test_register_and_clear_uploaded_image() -> None:
    """Test image registration into active cache and clearing buffer."""
    clear_active_images()
    fake_file = MockImageFile("test_pic.jpg")

    meta = register_uploaded_image(fake_file)
    assert meta["filename"] == "test_pic.jpg"
    assert "dimensions" in meta
    assert meta["dimensions"] == (100, 100)


def test_analyze_image_deep_with_no_images() -> None:
    """Test behavior when analyze_image_deep is called without any uploaded image."""
    clear_active_images()
    result = analyze_image_deep("find cars")
    assert "error" in result
    assert "No active images" in result["error"] or "uploaded" in result["error"]
