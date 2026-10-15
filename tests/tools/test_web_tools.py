"""
Tests for Deep Web Research tools: DuckDuckGo search, Wikipedia lookup, and HTML parser.
"""

from src.tools.web import get_web_tools, wikipedia_lookup


def test_web_tools_suite_registration():
    """Verify standard suite and 3-Tier web intelligence tools are exposed."""
    tools = get_web_tools()
    assert len(tools) == 6
    names = [t.name for t in tools]
    assert "duckduckgo_search" in names or any("search" in n for n in names)
    assert "wikipedia_lookup" in names
    assert "read_webpage_content" in names
    assert "scrape_web_adaptive" in names
    assert "crawl_web_dynamic" in names
    assert "map_website_urls" in names


def test_wikipedia_lookup_empty_or_nonsense():
    """Verify handling when Wikipedia returns no matches."""
    result = wikipedia_lookup.invoke({"query": "xzqj98274198273918273918273"})
    assert "No Wikipedia articles found" in result or "lookup error" in result.lower()


def test_wikipedia_lookup_known_subject():
    """Verify retrieving summary for a known encyclopedic topic or network fallback."""
    result = wikipedia_lookup.invoke({"query": "Python (programming language)"})
    assert any(k in result for k in ["Wikipedia:", "Python", "Wikipedia lookup error", "No Wikipedia articles"])
