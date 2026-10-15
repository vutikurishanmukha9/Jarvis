"""
Unit tests for J.A.R.V.I.S. 3-Tier Web Intelligence Matrix
(Scrapling Adaptive Scraper, Crawl4AI Dynamic Crawler, Firecrawl Site Mapper).
"""

from unittest.mock import MagicMock, patch

from src.tools.web import (
    crawl_web_dynamic,
    get_web_tools,
    map_website_urls,
    scrape_web_adaptive,
)


def test_get_web_tools_registration():
    """Verify that all 6 web research and intelligence tools are registered."""
    tools = get_web_tools()
    tool_names = [t.name for t in tools]
    assert "duckduckgo_search" in tool_names or any("search" in n for n in tool_names)
    assert "wikipedia_lookup" in tool_names
    assert "read_webpage_content" in tool_names
    assert "scrape_web_adaptive" in tool_names
    assert "crawl_web_dynamic" in tool_names
    assert "map_website_urls" in tool_names
    assert len(tools) == 6


def test_scrape_web_adaptive_ssrf_blocked():
    """Verify that private IP addresses are blocked by scrape_web_adaptive."""
    res = scrape_web_adaptive.invoke({"url": "http://127.0.0.1:8080/admin"})
    assert "[Security Blocked]" in res
    assert "private/internal IP" in res


def test_scrape_web_adaptive_success():
    """Verify that scrape_web_adaptive correctly extracts structured selector text."""
    sample_html = """
    <html>
        <head><title>Quantum Computing Lab</title></head>
        <body>
            <div class="target-card">
                <h2>Breakthrough Qubit Scaling</h2>
                <p>New topological qubits achieve 99.9% gate fidelity.</p>
            </div>
            <div class="footer">Noise footer</div>
        </body>
    </html>
    """.encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.content = sample_html
    mock_resp.raise_for_status.return_value = None

    with (
        patch("src.tools.web._fetch_with_safe_redirects", return_value=mock_resp),
        patch("src.tools.web._validate_url", return_value=""),
    ):
        res = scrape_web_adaptive.invoke(
            {
                "url": "https://example.com/quantum",
                "css_selector": ".target-card",
            }
        )

    assert "Quantum Computing Lab" in res
    assert "Breakthrough Qubit Scaling" in res
    assert "https://example.com/quantum" in res


def test_crawl_web_dynamic_ssrf_blocked():
    """Verify that private IP addresses are blocked by crawl_web_dynamic."""
    res = crawl_web_dynamic.invoke({"url": "http://192.168.1.1/router"})
    assert "[Security Blocked]" in res


def test_crawl_web_dynamic_fallback():
    """Verify that crawl_web_dynamic falls back safely to reader when dynamic crawler is uninitialized."""
    html_bytes = b"<html><head><title>Test App</title></head><body><p>Mocked safe reader content</p></body></html>"
    mock_resp = MagicMock()
    mock_resp.content = html_bytes
    mock_resp.iter_content.return_value = [html_bytes]
    mock_resp.headers = {"Content-Length": str(len(html_bytes))}
    mock_resp.raise_for_status.return_value = None

    with (
        patch("src.tools.web._validate_url", return_value=""),
        patch("src.tools.web._fetch_with_safe_redirects", return_value=mock_resp),
    ):
        res = crawl_web_dynamic.invoke({"url": "https://example.com/spa-app"})

    assert "Test App" in res
    assert "Mocked safe reader content" in res


def test_map_website_urls_ssrf_blocked():
    """Verify that private IP addresses are blocked by map_website_urls."""
    res = map_website_urls.invoke({"root_url": "http://10.0.0.1/dashboard"})
    assert "[Security Blocked]" in res


def test_map_website_urls_native_sitemapper():
    """Verify native sitemap discovery extracts and filters internal URLs."""
    sample_html = """
    <html>
        <body>
            <a href="/docs/intro">Intro</a>
            <a href="/docs/distributed">Distributed Guide</a>
            <a href="/docs/api">API Reference</a>
            <a href="https://external-site.com/other">External</a>
        </body>
    </html>
    """.encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.content = sample_html
    mock_resp.raise_for_status.return_value = None

    with (
        patch("src.tools.web._fetch_with_safe_redirects", return_value=mock_resp),
        patch("src.tools.web._validate_url", return_value=""),
    ):
        # Filter for 'distributed'
        res = map_website_urls.invoke(
            {
                "root_url": "https://pytorch.org",
                "search_query": "distributed",
            }
        )

    assert "https://pytorch.org/docs/distributed" in res
    assert "external-site.com" not in res
