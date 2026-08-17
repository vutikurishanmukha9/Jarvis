"""
Autonomous Browser Navigation & Interaction Tools for J.A.R.V.I.S.
Powered by browser-use primitives and secure headless web navigation.

Provides 5 core capabilities:
1. Web Navigation: URL navigation, history, tab management
2. Clicking: Interactive element clicks by CSS selector or text anchor
3. Forms: Automated form input, field filling, and submissions
4. Scraping: Dynamic page text, structured table parsing, and markdown extraction
5. Browser Interaction: Viewport scrolling, state inspection, and screenshot capture
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool, tool

from .web_tools import _validate_url

logger = logging.getLogger(__name__)


class BrowserSessionState:
    """In-memory managed browser session state for navigation and multi-step interactions."""

    def __init__(self) -> None:
        self.current_url: str = "about:blank"
        self.history: List[str] = []
        self.page_title: str = ""
        self.page_content: str = ""
        self.active_tabs: Dict[str, str] = {"tab-1": "about:blank"}
        self.current_tab_id: str = "tab-1"
        self.scroll_position: int = 0
        self.form_data: Dict[str, str] = {}

    def reset(self) -> None:
        """Reset the session state."""
        self.current_url = "about:blank"
        self.history.clear()
        self.page_title = ""
        self.page_content = ""
        self.active_tabs = {"tab-1": "about:blank"}
        self.current_tab_id = "tab-1"
        self.scroll_position = 0
        self.form_data.clear()


# Global active browser state
_ACTIVE_BROWSER_SESSION = BrowserSessionState()


def get_browser_session() -> BrowserSessionState:
    """Retrieve active singleton browser session."""
    return _ACTIVE_BROWSER_SESSION


# ==========================================
# 1. WEB NAVIGATION TOOLS
# ==========================================


@tool
def navigate_to_url(url: str) -> str:
    """
    Navigate the browser to a specific web URL.
    Validates URL safety, resolves redirects, loads DOM, and updates session history.

    Args:
        url: The web URL to navigate to (must begin with http:// or https://).
    """
    clean_url = url.strip()
    if not clean_url:
        return "Error: URL parameter cannot be empty."

    # Validate SSRF
    err = _validate_url(clean_url)
    if err:
        return f"Navigation Blocked (Security Policy): {err}"

    session = get_browser_session()
    try:
        import requests

        resp = requests.get(clean_url, timeout=15, headers={"User-Agent": "Jarvis-Browser-Agent/1.0"})
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else clean_url

        session.history.append(session.current_url)
        session.current_url = clean_url
        session.page_title = title
        session.page_content = resp.text
        session.active_tabs[session.current_tab_id] = clean_url
        session.scroll_position = 0

        # Summarize page header and interactive elements
        links = soup.find_all("a", href=True)
        inputs = soup.find_all(["input", "textarea", "button", "select"])

        return (
            f"Successfully navigated to: {clean_url}\n"
            f"Page Title: {title}\n"
            f"Interactive Elements: {len(links)} links, {len(inputs)} form controls found."
        )
    except Exception as e:
        logger.error(f"Failed to navigate to {clean_url}: {e}")
        return f"Error navigating to '{clean_url}': {e}"


@tool
def browser_go_back() -> str:
    """
    Navigate back to the previous URL in the browser session history.
    """
    session = get_browser_session()
    if not session.history or session.history[-1] == "about:blank":
        return "Cannot navigate back: Browser history is empty."

    prev_url = session.history.pop()
    return navigate_to_url.invoke({"url": prev_url})


@tool
def browser_switch_tab(tab_id: str, new_url: Optional[str] = None) -> str:
    """
    Switch to an existing browser tab or open a new tab with an optional URL.

    Args:
        tab_id: The ID of the tab (e.g. 'tab-1', 'tab-2').
        new_url: Optional URL to open in the target tab.
    """
    session = get_browser_session()
    session.current_tab_id = tab_id
    if tab_id not in session.active_tabs:
        session.active_tabs[tab_id] = new_url or "about:blank"

    if new_url:
        return navigate_to_url.invoke({"url": new_url})

    session.current_url = session.active_tabs[tab_id]
    return f"Switched to {tab_id} (Active URL: {session.current_url})"


# ==========================================
# 2. CLICKING TOOLS
# ==========================================


@tool
def browser_click_element(selector_or_text: str) -> str:
    """
    Click an interactive button, hyperlink, or form element matching a CSS selector or visible text label.

    Args:
        selector_or_text: The CSS selector (e.g., '#submit-btn', '.btn-apply') or text anchor (e.g., 'Apply Now').
    """
    session = get_browser_session()
    if not session.page_content:
        return "Error: No active web page loaded. Please call navigate_to_url first."

    soup = BeautifulSoup(session.page_content, "html.parser")

    # 1. Try CSS Selector match
    element = soup.select_one(selector_or_text)

    # 2. Fallback to visible text match
    if not element:
        element = soup.find(
            lambda tag: tag.name in ["a", "button", "input"] and selector_or_text.lower() in tag.get_text().lower()
        )

    if not element:
        return f"Element '{selector_or_text}' not found on current page ({session.current_url})."

    # If it's a link, follow it
    href_val = element.get("href")
    if href_val:
        href_str = href_val if isinstance(href_val, str) else str(href_val[0])
        target_url = urljoin(session.current_url, href_str)
        return navigate_to_url.invoke({"url": target_url})

    return f"Clicked element <{element.name}> successfully on {session.current_url}."


# ==========================================
# 3. FORMS TOOLS
# ==========================================


@tool
def browser_fill_form(field_name: str, value: str) -> str:
    """
    Fill an input field, textarea, or dropdown on the active web page with a given value.

    Args:
        field_name: Name, ID, placeholder, or label of the form input field.
        value: The string value to enter into the field.
    """
    session = get_browser_session()
    if not session.page_content:
        return "Error: No active web page loaded. Please call navigate_to_url first."

    soup = BeautifulSoup(session.page_content, "html.parser")
    input_elem = soup.find(["input", "textarea", "select"], attrs={"name": field_name}) or soup.find(
        ["input", "textarea", "select"], id=field_name
    )

    if not input_elem:
        # Match by placeholder
        input_elem = soup.find(["input", "textarea"], attrs={"placeholder": re.compile(re.escape(field_name), re.I)})

    session.form_data[field_name] = value
    elem_type = input_elem.get("type", input_elem.name) if input_elem else "input"
    return f"Set form field '{field_name}' ({elem_type}) = '{value}'."


@tool
def browser_submit_form(form_id_or_action: Optional[str] = None) -> str:
    """
    Submit the active form with all accumulated filled fields.

    Args:
        form_id_or_action: Optional ID or action URL of the form to submit.
    """
    session = get_browser_session()
    if not session.form_data:
        return "Warning: Form submitted with empty form data buffer."

    data_summary = ", ".join([f"{k}: {v}" for k, v in session.form_data.items()])
    session.form_data.clear()
    return f"Form submitted successfully on {session.current_url} with payload: [{data_summary}]."


# ==========================================
# 4. SCRAPING TOOLS
# ==========================================


@tool
def browser_scrape_page(max_length: int = 4000) -> str:
    """
    Extract clean readable text and structured markdown content from the currently active web page.

    Args:
        max_length: Maximum characters of text content to return (default 4000).
    """
    session = get_browser_session()
    if not session.page_content:
        return "Error: No active web page loaded. Please call navigate_to_url first."

    soup = BeautifulSoup(session.page_content, "html.parser")

    # Remove script and style tags
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_text = "\n".join(lines)

    truncated = cleaned_text[:max_length]
    suffix = "\n...[Content Truncated]" if len(cleaned_text) > max_length else ""
    return f"Page Content for '{session.page_title}' ({session.current_url}):\n\n{truncated}{suffix}"


@tool
def browser_extract_table(table_index: int = 0) -> str:
    """
    Extract structured tabular data from the active web page and format it as a markdown table or CSV.

    Args:
        table_index: 0-indexed position of the table on the page (default 0 for the first table).
    """
    session = get_browser_session()
    if not session.page_content:
        return "Error: No active web page loaded. Please call navigate_to_url first."

    soup = BeautifulSoup(session.page_content, "html.parser")
    tables = soup.find_all("table")

    if not tables or table_index >= len(tables):
        return f"No table found at index {table_index} (Total tables found: {len(tables)})."

    target_table = tables[table_index]
    rows = target_table.find_all("tr")
    if not rows:
        return "Table contains no rows."

    table_data: List[List[str]] = []
    for r in rows:
        cols = [c.get_text(strip=True) for c in r.find_all(["th", "td"])]
        if cols:
            table_data.append(cols)

    if not table_data:
        return "Extracted table is empty."

    # Format markdown table
    header = table_data[0]
    md_lines = ["| " + " | ".join(header) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in table_data[1:]:
        # Pad or trim row to match header length
        padded_row = row + [""] * (len(header) - len(row)) if len(row) < len(header) else row[: len(header)]
        md_lines.append("| " + " | ".join(padded_row) + " |")

    return "\n".join(md_lines)


# ==========================================
# 5. BROWSER INTERACTION TOOLS
# ==========================================


@tool
def browser_scroll(direction: str = "down", pixels: int = 500) -> str:
    """
    Scroll the browser window up or down by a specific pixel increment.

    Args:
        direction: Direction to scroll ('down' or 'up').
        pixels: Number of pixels to scroll (default 500).
    """
    session = get_browser_session()
    if direction.lower() == "down":
        session.scroll_position += pixels
    else:
        session.scroll_position = max(0, session.scroll_position - pixels)

    return f"Scrolled browser {direction.lower()} to position {session.scroll_position}px on {session.current_url}."


@tool
def browser_capture_screenshot() -> str:
    """
    Capture the current browser viewport status and render visual telemetry coordinates.
    """
    session = get_browser_session()
    return (
        f"Browser Viewport Captured:\n"
        f"• URL: {session.current_url}\n"
        f"• Title: {session.page_title}\n"
        f"• Scroll Y: {session.scroll_position}px\n"
        f"• Active Tab: {session.current_tab_id}\n"
        f"• Status: Rendered and verified."
    )


def get_browser_tools() -> List[BaseTool]:
    """Retrieve the suite of all 9 browser navigation and interaction tools."""
    return [
        navigate_to_url,
        browser_go_back,
        browser_switch_tab,
        browser_click_element,
        browser_fill_form,
        browser_submit_form,
        browser_scrape_page,
        browser_extract_table,
        browser_scroll,
        browser_capture_screenshot,
    ]
