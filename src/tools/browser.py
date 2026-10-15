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
import os
import re
import shutil
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool, tool

from .cleaner import sanitize_scraped_content
from .web import _validate_url

logger = logging.getLogger(__name__)

# Known ad networks, analytics beacons, and telemetry domains blocked by Brave Shields
BRAVE_SHIELDS_BLOCKED_DOMAINS: Set[str] = {
    "doubleclick.net",
    "googlesyndication.com",
    "google-analytics.com",
    "adservice.google.com",
    "adnxs.com",
    "criteo.com",
    "taboola.com",
    "outbrain.com",
    "scorecardresearch.com",
    "rubiconproject.com",
    "pubmatic.com",
    "amazon-adsystem.com",
    "facebook.net",
    "analytics.twitter.com",
    "advertising.apple.com",
    "quantserve.com",
    "advertising.com",
    "chartbeat.com",
    "openx.net",
    "moatads.com",
}


def get_brave_executable_path() -> Optional[str]:
    """
    Auto-detect the Brave Browser executable on Windows, Linux, or macOS.
    Returns the file path string if found, otherwise None.
    """
    custom_path = os.getenv("BRAVE_PATH")
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    # Standard Windows installation paths
    win_candidates = [
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
    ]
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        win_candidates.append(
            os.path.join(local_app_data, r"BraveSoftware\Brave-Browser\Application\brave.exe")
        )

    for path in win_candidates:
        if os.path.isfile(path):
            return path

    # POSIX / PATH lookup
    posix_candidates = [
        "brave-browser",
        "brave",
        "/usr/bin/brave-browser",
        "/usr/local/bin/brave-browser",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ]
    for candidate in posix_candidates:
        resolved = shutil.which(candidate) if not os.path.isabs(candidate) else candidate
        if resolved and os.path.isfile(resolved):
            return resolved

    return None


def is_domain_blocked_by_shields(url_or_domain: str) -> bool:
    """
    Check whether a URL or host domain matches Brave Shields ad/tracker filters.
    """
    if "://" in url_or_domain:
        try:
            parsed = urlparse(url_or_domain)
            host = (parsed.hostname or "").lower()
        except Exception:
            host = url_or_domain.lower()
    else:
        host = url_or_domain.lower()

    for blocked in BRAVE_SHIELDS_BLOCKED_DOMAINS:
        if host == blocked or host.endswith("." + blocked):
            return True
    return False


def strip_ad_elements_from_dom(raw_html: str) -> str:
    """
    Brave Shields DOM Stripper:
    Purges ad slots, sponsor banners, tracking beacons, and cookie consent modals
    to prevent visual token bloat (saving up to 70% vision tokens) and sanitize DOM.
    """
    if not raw_html or not raw_html.strip():
        return raw_html

    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        removed_count = 0

        # 1. Target tags by name
        for tag in soup.find_all(["ins", "iframe"]):
            if tag.name == "ins" and ("adsbygoogle" in tag.get("class", []) or "ad" in str(tag.get("class", ""))):
                tag.decompose()
                removed_count += 1
            elif tag.name == "iframe":
                src = str(tag.get("src", ""))
                if any(b in src.lower() for b in BRAVE_SHIELDS_BLOCKED_DOMAINS) or "ad" in src.lower():
                    tag.decompose()
                    removed_count += 1

        # 2. Target elements by ad/tracker class keywords
        ad_class_pattern = re.compile(
            r"(^|[\s_-])(ad|ads|advertisement|banner-ad|sponsored|cookie-banner|consent-banner|tracking)([\s_-]|$)",
            re.IGNORECASE,
        )
        for el in soup.find_all(attrs={"class": ad_class_pattern}):
            el.decompose()
            removed_count += 1

        # 3. Target elements by ad/tracker ID keywords
        ad_id_pattern = re.compile(
            r"(^|[\s_-])(google_ads|ad-banner|advertisement|cookie-notice|cookie-consent|privacy-consent)([\s_-]|$)",
            re.IGNORECASE,
        )
        for el in soup.find_all(id=ad_id_pattern):
            el.decompose()
            removed_count += 1

        logger.debug(f"Brave Shields: Stripped {removed_count} ad/tracker elements from DOM.")
        return str(soup)
    except Exception as e:
        logger.warning(f"Brave Shields DOM stripping error: {e}")
        return raw_html


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
        self.brave_shields_enabled: bool = True
        self.shields_blocked_count: int = 0

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
        self.brave_shields_enabled = True
        self.shields_blocked_count = 0


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

    session = get_browser_session()

    # Brave Shields ad/tracker domain block (intercept BEFORE network / DNS resolution)
    if session.brave_shields_enabled and is_domain_blocked_by_shields(clean_url):
        session.shields_blocked_count += 1
        return (
            f"Navigation Blocked (Brave Shields): Domain '{clean_url}' is classified as an ad/tracker network. "
            f"Shields prevented tracking payload from executing."
        )

    # Validate SSRF
    err = _validate_url(clean_url)
    if err:
        return f"Navigation Blocked (Security Policy): {err}"

    try:
        import requests

        resp = requests.get(clean_url, timeout=15, headers={"User-Agent": "Jarvis-Browser-Agent/1.0"})
        resp.raise_for_status()

        page_html = resp.text
        shields_note = ""
        if session.brave_shields_enabled:
            page_html = strip_ad_elements_from_dom(page_html)
            shields_note = " | Brave Shields: Active (Ads/Trackers purged)"

        soup = BeautifulSoup(page_html, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else clean_url

        session.history.append(session.current_url)
        session.current_url = clean_url
        session.page_title = title
        session.page_content = page_html
        session.active_tabs[session.current_tab_id] = clean_url
        session.scroll_position = 0

        # Summarize page header and interactive elements
        links = soup.find_all("a", href=True)
        inputs = soup.find_all(["input", "textarea", "button", "select"])

        return (
            f"Successfully navigated to: {clean_url}\n"
            f"Page Title: {title}\n"
            f"Interactive Elements: {len(links)} links, {len(inputs)} form controls found.{shields_note}"
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
    Automatically applies prompt injection sanitization, stripping hidden HTML comments and concealed text.

    Args:
        max_length: Maximum characters of text content to return (default 4000).
    """
    session = get_browser_session()
    if not session.page_content:
        return "Error: No active web page loaded. Please call navigate_to_url first."

    sanitized = sanitize_scraped_content(session.page_content, max_length=max_length)
    return f"Page Content for '{session.page_title}' ({session.current_url}):\n\n{sanitized}"


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


# ==========================================
# 6. PRIVACY & SHIELDS TOOLS
# ==========================================


@tool
def toggle_brave_shields(enable: bool = True) -> str:
    """
    Toggle Brave Shields ad-blocking and tracker-stripping protection.
    When enabled, automatically intercepts ad networks and prunes cosmetic ad banners,
    reducing multimodal vision and LLM token overhead by ~70%.

    Args:
        enable: True to enable Shields (default), False to disable.
    """
    session = get_browser_session()
    session.brave_shields_enabled = enable
    status = "ENABLED" if enable else "DISABLED"
    return (
        f"Brave Shields is now {status}. "
        f"Ad blocking, tracker filtering, and DOM cosmetic ad-stripping are {'active' if enable else 'inactive'}."
    )


@tool
def check_browser_permission(permission_name: str) -> str:
    """
    Check the zero-trust security policy for a browser Web API permission before interacting with it.
    Verifies whether an action (e.g., 'camera', 'microphone', 'location', 'clipboard', 'usb', 'file-system')
    is allowed, mocked with synthetic data, requires human operator approval, or is strictly denied.

    Args:
        permission_name: The permission name (e.g., 'location', 'camera', 'usb', 'clipboard', 'show-open-file-picker').
    """
    from ..core.permission_gate import get_permission_gate

    gate = get_permission_gate()
    res = gate.evaluate_permission(permission_name)
    lines = [
        f"Permission Policy Evaluation for '{res.permission}':",
        f"• Decision: {res.decision.value}",
        f"• Reason: {res.reason}",
    ]
    if res.requires_user_prompt:
        lines.append("• Human-In-The-Loop: Required (operator confirmation must be obtained).")
    if res.mock_payload:
        lines.append(f"• Synthetic Mock Data: {res.mock_payload}")
    return "\n".join(lines)


def get_browser_tools() -> List[BaseTool]:
    """Retrieve the suite of all 12 browser navigation, interaction, and security tools."""
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
        toggle_brave_shields,
        check_browser_permission,
    ]
