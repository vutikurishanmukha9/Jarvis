"""
Unit tests validating Browser Capability 1: Web Navigation.
Tests URL navigation, SSRF policy enforcement, history management, and tab switching.
"""

from unittest.mock import MagicMock, patch

from src.tools.browser_tools import (
    browser_go_back,
    browser_switch_tab,
    get_browser_session,
    navigate_to_url,
)


def test_navigate_to_url_empty_or_whitespace() -> None:
    """Test navigation with empty input."""
    res = navigate_to_url.invoke({"url": "   "})
    assert "Error" in res
    assert "empty" in res.lower()


def test_navigate_to_url_blocks_ssrf_private_ips() -> None:
    """Test SSRF guard blocks internal loopback and private IPs."""
    res = navigate_to_url.invoke({"url": "http://127.0.0.1:8000/admin"})
    assert "Blocked" in res or "Security Policy" in res

    res2 = navigate_to_url.invoke({"url": "http://169.254.169.254/latest/meta-data"})
    assert "Blocked" in res2 or "Security Policy" in res2


def test_navigate_to_url_success_updates_session() -> None:
    """Test successful URL navigation parses DOM and updates session state."""
    session = get_browser_session()
    session.reset()

    html_content = (
        "<html><head><title>Acme Portal</title></head><body><a href='/jobs'>Jobs</a><input name='q' /></body></html>"
    )

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.text = html_content
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        res = navigate_to_url.invoke({"url": "https://acme.org/home"})
        assert "Successfully navigated" in res
        assert "Acme Portal" in res
        assert session.current_url == "https://acme.org/home"
        assert session.page_title == "Acme Portal"


def test_browser_go_back_empty_history() -> None:
    """Test go_back when history is empty."""
    session = get_browser_session()
    session.reset()
    res = browser_go_back.invoke({})
    assert "Cannot navigate back" in res


def test_browser_switch_tab_lifecycle() -> None:
    """Test opening and switching between multiple browser tabs."""
    session = get_browser_session()
    session.reset()

    res = browser_switch_tab.invoke({"tab_id": "tab-2", "new_url": ""})
    assert "Switched to tab-2" in res
    assert session.current_tab_id == "tab-2"
