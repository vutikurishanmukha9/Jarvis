"""
Unit tests validating Browser Capability 5: Browser Interaction.
Tests viewport scrolling, screenshot telemetry capture, and tool collection registration.
"""

from src.tools.browser_tools import (
    browser_capture_screenshot,
    browser_scroll,
    get_browser_session,
    get_browser_tools,
)


def test_browser_scroll_down_and_up() -> None:
    """Test viewport scrolling in downward and upward directions."""
    session = get_browser_session()
    session.reset()
    session.current_url = "https://example.com/long-page"

    res_down = browser_scroll.invoke({"direction": "down", "pixels": 600})
    assert "position 600px" in res_down
    assert session.scroll_position == 600

    res_up = browser_scroll.invoke({"direction": "up", "pixels": 200})
    assert "position 400px" in res_up
    assert session.scroll_position == 400


def test_browser_capture_screenshot_telemetry() -> None:
    """Test browser viewport capture telemetry."""
    session = get_browser_session()
    session.reset()
    session.current_url = "https://example.com/dashboard"
    session.page_title = "Executive Dashboard"
    session.scroll_position = 300

    res = browser_capture_screenshot.invoke({})
    assert "Browser Viewport Captured" in res
    assert "https://example.com/dashboard" in res
    assert "Executive Dashboard" in res
    assert "Scroll Y: 300px" in res


def test_get_browser_tools_suite() -> None:
    """Test complete browser tool suite registration."""
    tools = get_browser_tools()
    tool_names = [t.name for t in tools]
    assert "navigate_to_url" in tool_names
    assert "browser_click_element" in tool_names
    assert "browser_fill_form" in tool_names
    assert "browser_submit_form" in tool_names
    assert "browser_scrape_page" in tool_names
    assert "browser_extract_table" in tool_names
    assert "browser_scroll" in tool_names
    assert "browser_capture_screenshot" in tool_names
