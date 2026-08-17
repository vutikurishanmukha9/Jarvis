"""
Unit tests validating Browser Capability 2: Clicking.
Tests clicking CSS selectors, text anchors, and automatic link redirection.
"""

from unittest.mock import MagicMock, patch

from src.tools.browser_tools import (
    browser_click_element,
    get_browser_session,
)


def test_click_element_no_page_loaded() -> None:
    """Test clicking when no active page is loaded."""
    session = get_browser_session()
    session.reset()
    res = browser_click_element.invoke({"selector_or_text": "#submit-button"})
    assert "Error: No active web page loaded" in res


def test_click_element_by_css_selector() -> None:
    """Test clicking an element matching a CSS selector."""
    session = get_browser_session()
    session.reset()
    session.current_url = "https://example.com/app"
    session.page_content = "<html><body><button id='btn-action'>Perform Action</button></body></html>"

    res = browser_click_element.invoke({"selector_or_text": "#btn-action"})
    assert "Clicked element <button> successfully" in res


def test_click_element_by_text_anchor() -> None:
    """Test clicking an element matching visible text anchor."""
    session = get_browser_session()
    session.reset()
    session.current_url = "https://example.com/app"
    session.page_content = "<html><body><button class='cta'>Apply Now</button></body></html>"

    res = browser_click_element.invoke({"selector_or_text": "Apply Now"})
    assert "Clicked element <button> successfully" in res


def test_click_link_triggers_navigation() -> None:
    """Test clicking a link follows the href attribute."""
    session = get_browser_session()
    session.reset()
    session.current_url = "https://example.com/home"
    session.page_content = "<html><body><a href='/careers' class='nav-link'>Careers</a></body></html>"

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.text = "<html><head><title>Careers Portal</title></head><body>Open Positions</body></html>"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        res = browser_click_element.invoke({"selector_or_text": ".nav-link"})
        assert "Successfully navigated to: https://example.com/careers" in res


def test_click_nonexistent_element() -> None:
    """Test clicking an element that does not exist."""
    session = get_browser_session()
    session.reset()
    session.current_url = "https://example.com/home"
    session.page_content = "<html><body><p>Hello World</p></body></html>"

    res = browser_click_element.invoke({"selector_or_text": "#missing-id"})
    assert "Element '#missing-id' not found" in res
