"""
Unit tests validating Brave Browser & Brave Shields Integration.
Tests Brave binary path auto-detection, ad/tracker domain filtering,
DOM ad-stripping token reduction, and the toggle_brave_shields tool.
"""

from unittest.mock import MagicMock, patch

from src.tools.browser import (
    BRAVE_SHIELDS_BLOCKED_DOMAINS,
    get_brave_executable_path,
    get_browser_session,
    get_browser_tools,
    is_domain_blocked_by_shields,
    navigate_to_url,
    strip_ad_elements_from_dom,
    toggle_brave_shields,
)


def test_brave_executable_path_detection() -> None:
    """Test auto-detection of Brave Browser binary via env var and system paths."""
    with patch.dict("os.environ", {"BRAVE_PATH": "/custom/path/to/brave"}):
        with patch("os.path.isfile", side_effect=lambda p: p == "/custom/path/to/brave"):
            detected = get_brave_executable_path()
            assert detected == "/custom/path/to/brave"

    # Test fallback when not installed
    with patch.dict("os.environ", {}, clear=True):
        with patch("os.path.isfile", return_value=False):
            with patch("shutil.which", return_value=None):
                detected = get_brave_executable_path()
                assert detected is None


def test_is_domain_blocked_by_shields() -> None:
    """Test Brave Shields domain filtering against known ad and tracker networks."""
    # Direct domain match
    assert is_domain_blocked_by_shields("doubleclick.net") is True
    assert is_domain_blocked_by_shields("google-analytics.com") is True
    assert is_domain_blocked_by_shields("amazon-adsystem.com") is True

    # Subdomain match
    assert is_domain_blocked_by_shields("adservice.google-analytics.com") is True
    assert is_domain_blocked_by_shields("https://stats.doubleclick.net/tracker.js") is True

    # Legitimate non-ad domains
    assert is_domain_blocked_by_shields("https://github.com/torvalds/linux") is False
    assert is_domain_blocked_by_shields("https://en.wikipedia.org/wiki/Main_Page") is False
    assert is_domain_blocked_by_shields("https://pytorch.org") is False


def test_strip_ad_elements_from_dom() -> None:
    """Test Brave Shields DOM ad-stripper removes ad slots, cookie modals, and banners."""
    sample_dirty_html = """
    <html>
        <head><title>Research Article</title></head>
        <body>
            <div id="cookie-notice" class="cookie-banner">Accept cookies to continue</div>
            <ins class="adsbygoogle" style="display:block" data-ad-client="ca-pub-12345"></ins>
            <div class="ad-container banner-ad">
                <a href="https://buy-now.example">Discount Shoes 50% Off</a>
            </div>
            <article class="main-content">
                <h1>Autonomous AI Systems</h1>
                <p>Neural networks and agentic planning are transforming robotics.</p>
            </article>
            <div id="google_ads_frame_1">Ad Sponsor Content</div>
            <iframe src="https://adservice.google.com/frame"></iframe>
            <footer>Copyright 2026 AI Journal</footer>
        </body>
    </html>
    """

    cleaned_html = strip_ad_elements_from_dom(sample_dirty_html)

    # Legitimate article content preserved
    assert "<h1>Autonomous AI Systems</h1>" in cleaned_html
    assert "Neural networks and agentic planning" in cleaned_html
    assert "Copyright 2026 AI Journal" in cleaned_html

    # Ad slots, cookie banners, and sponsor containers purged
    assert "cookie-notice" not in cleaned_html
    assert "adsbygoogle" not in cleaned_html
    assert "Discount Shoes" not in cleaned_html
    assert "google_ads_frame_1" not in cleaned_html
    assert "adservice.google.com" not in cleaned_html


def test_toggle_brave_shields_tool() -> None:
    """Test toggling Brave Shields active state via tool invocation."""
    session = get_browser_session()
    session.reset()

    assert session.brave_shields_enabled is True

    # Disable
    res_disable = toggle_brave_shields.invoke({"enable": False})
    assert "DISABLED" in res_disable
    assert session.brave_shields_enabled is False

    # Re-enable
    res_enable = toggle_brave_shields.invoke({"enable": True})
    assert "ENABLED" in res_enable
    assert session.brave_shields_enabled is True


def test_navigate_to_url_with_brave_shields_active() -> None:
    """Test that navigate_to_url blocks ad network URLs when shields are enabled."""
    session = get_browser_session()
    session.reset()
    session.brave_shields_enabled = True

    # Ad network domain should be blocked
    res = navigate_to_url.invoke({"url": "https://stats.doubleclick.net/ad_pixel.gif"})
    assert "Navigation Blocked (Brave Shields)" in res
    assert session.shields_blocked_count >= 1

    # Legitimate domain with ads in DOM should have ads purged
    page_with_ads = (
        "<html><head><title>Tech News</title></head><body>"
        "<div class='banner-ad'>Ad 50% off</div>"
        "<p>Real news report here.</p></body></html>"
    )

    with patch("src.tools.browser._validate_url", return_value=None):
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = page_with_ads
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            nav_res = navigate_to_url.invoke({"url": "https://technews.example/article"})
            assert "Successfully navigated" in nav_res
            assert "Brave Shields: Active" in nav_res
            assert "banner-ad" not in session.page_content
            assert "Real news report here." in session.page_content


def test_navigate_to_url_with_brave_shields_disabled() -> None:
    """Test that when shields are disabled, ad network domain check is skipped."""
    session = get_browser_session()
    session.reset()
    session.brave_shields_enabled = False

    with patch("src.tools.browser._validate_url", return_value=None):
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = "<html><body>Ad tracking payload</body></html>"
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            res = navigate_to_url.invoke({"url": "https://stats.doubleclick.net/data"})
            assert "Navigation Blocked (Brave Shields)" not in res
            assert "Successfully navigated" in res


def test_browser_tools_includes_toggle_brave_shields() -> None:
    """Verify toggle_brave_shields is registered in get_browser_tools."""
    tools = get_browser_tools()
    tool_names = [t.name for t in tools]
    assert "toggle_brave_shields" in tool_names
