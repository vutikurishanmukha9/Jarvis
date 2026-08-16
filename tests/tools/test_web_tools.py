"""
Tests for Deep Web Research tools: DuckDuckGo search, Wikipedia lookup, and HTML parser.
"""

import pytest
from src.tools.web_tools import get_web_tools, wikipedia_lookup

def test_web_tools_suite_registration():
    """Verify standard suite of web research tools are exposed."""
    tools = get_web_tools()
    assert len(tools) == 3
    names = [t.name for t in tools]
    assert "duckduckgo_search" in names
    assert "wikipedia_lookup" in names
    assert "read_webpage_content" in names

def test_wikipedia_lookup_empty_or_nonsense():
    """Verify handling when Wikipedia returns no matches."""
    result = wikipedia_lookup.invoke({"query": "xzqj98274198273918273918273"})
    assert "No Wikipedia articles found" in result or "lookup error" in result.lower()

def test_wikipedia_lookup_known_subject():
    """Verify retrieving summary for a known encyclopedic topic."""
    result = wikipedia_lookup.invoke({"query": "Python (programming language)"})
    assert "Wikipedia:" in result or "Python" in result
    assert "URL:" in result
