"""
Unit tests for Web Research tools and SSRF URL protection.
"""

from src.tools.web_tools import _validate_url, get_web_tools


def test_web_tools_collection() -> None:
    """Ensure get_web_tools returns DuckDuckGo, Wikipedia, and Scraper tools."""
    tools = get_web_tools()
    assert len(tools) >= 3
    tool_names = [t.name for t in tools]
    assert "duckduckgo_search" in tool_names
    assert "wikipedia_lookup" in tool_names
    assert "read_webpage_content" in tool_names


def test_validate_url_blocks_internal_ips() -> None:
    """Ensure internal/localhost IPs are blocked to prevent SSRF attacks."""
    res1 = _validate_url("http://127.0.0.1:8000/admin")
    assert "Blocked" in res1 and "127.0.0.1" in res1

    res2 = _validate_url("http://192.168.1.1/secret")
    assert "Blocked" in res2 and "192.168.1.1" in res2

    res3 = _validate_url("http://10.0.0.5/api")
    assert "Blocked" in res3 and "10.0.0.5" in res3


def test_validate_url_blocks_disallowed_schemes() -> None:
    """Ensure file://, ftp://, and gopher:// schemes are rejected."""
    res1 = _validate_url("file:///etc/passwd")
    assert "Blocked URL scheme" in res1

    res2 = _validate_url("ftp://ftp.example.com")
    assert "Blocked URL scheme" in res2
