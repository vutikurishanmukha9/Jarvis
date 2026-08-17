"""
Unit tests validating Browser Capability 4: Scraping.
Tests page content extraction, noise script/style filtering, and markdown table synthesis.
"""

from src.tools.browser_tools import (
    browser_extract_table,
    browser_scrape_page,
    get_browser_session,
)


def test_scrape_page_no_page_loaded() -> None:
    """Test scraping when no active page is loaded."""
    session = get_browser_session()
    session.reset()
    res = browser_scrape_page.invoke({"max_length": 1000})
    assert "Error: No active web page loaded" in res


def test_scrape_page_cleans_scripts_and_styles() -> None:
    """Test scraping filters out javascript and styling blocks."""
    session = get_browser_session()
    session.reset()
    session.current_url = "https://example.com/article"
    session.page_title = "AI Research"
    session.page_content = (
        "<html><head><style>body { color: red; }</style></head>"
        "<body>"
        "<script>console.log('secret');</script>"
        "<h1>Autonomous Agents</h1>"
        "<p>Agents plan and execute multi-step tasks.</p>"
        "</body></html>"
    )

    res = browser_scrape_page.invoke({"max_length": 2000})
    assert "Autonomous Agents" in res
    assert "Agents plan and execute multi-step tasks." in res
    assert "console.log" not in res
    assert "color: red" not in res


def test_extract_table_to_markdown() -> None:
    """Test extracting HTML table into formatted markdown table."""
    session = get_browser_session()
    session.reset()
    session.current_url = "https://example.com/data"
    session.page_content = (
        "<html><body>"
        "<table>"
        "<tr><th>Candidate</th><th>Role</th><th>Score</th></tr>"
        "<tr><td>Alice</td><td>Engineer</td><td>95</td></tr>"
        "<tr><td>Bob</td><td>Scientist</td><td>88</td></tr>"
        "</table>"
        "</body></html>"
    )

    res = browser_extract_table.invoke({"table_index": 0})
    assert "| Candidate | Role | Score |" in res
    assert "| Alice | Engineer | 95 |" in res
    assert "| Bob | Scientist | 88 |" in res


def test_extract_nonexistent_table() -> None:
    """Test extracting table when no tables exist on the page."""
    session = get_browser_session()
    session.reset()
    session.current_url = "https://example.com/no-tables"
    session.page_content = "<html><body><p>No tabular data here.</p></body></html>"

    res = browser_extract_table.invoke({"table_index": 0})
    assert "No table found at index 0" in res
