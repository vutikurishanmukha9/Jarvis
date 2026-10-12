"""
Content Sanitizer & Prompt Injection Guard for J.A.R.V.I.S.
Defends against Indirect Prompt Injection (VULN-01) by stripping hidden directives,
zero-width Unicode payloads, HTML comments, and styling-concealed text from external web content.
"""

from __future__ import annotations

import logging
import re
from typing import List

from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)

# Zero-width spaces, joiners, formatting and directional override Unicode codepoints
INVISIBLE_UNICODE_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF\u200E\u200F\u202A-\u202E\u2060-\u206F\u00AD]")

# Regex pattern to identify concealed CSS styles
HIDDEN_STYLE_PATTERN = re.compile(
    r"(display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|opacity\s*:\s*0|height\s*:\s*0px|width\s*:\s*0px)",
    re.IGNORECASE,
)

# System override patterns commonly used in indirect prompt injection attacks
SUSPICIOUS_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"<!--\s*(?:system|priority override|ignore previous|execute command)", re.IGNORECASE),
    re.compile(r"\[SYSTEM(?:\s+UPDATE|\s+DIRECTIVE|\s+OVERRIDE)?\]", re.IGNORECASE),
    re.compile(r"Ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions", re.IGNORECASE),
]


def strip_html_comments(soup: BeautifulSoup) -> None:
    """Find and remove all HTML comments in-place."""
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()


def strip_hidden_elements(soup: BeautifulSoup) -> None:
    """Find and remove elements explicitly concealed via CSS or hidden attributes."""
    for tag in soup.find_all(True):
        # 1. Elements with 'hidden' attribute
        if tag.has_attr("hidden") or tag.get("aria-hidden") == "true":
            tag.decompose()
            continue

        # 2. Elements with inline concealment styling
        style_attr = tag.get("style", "")
        if isinstance(style_attr, str) and HIDDEN_STYLE_PATTERN.search(style_attr):
            tag.decompose()
            continue

        # 3. High-risk non-rendered tags
        if tag.name in ["script", "style", "noscript", "svg", "template"]:
            tag.decompose()


def strip_invisible_unicode(text: str) -> str:
    """Strip zero-width characters, RTL overrides, and invisible control codes."""
    return INVISIBLE_UNICODE_PATTERN.sub("", text)


def sanitize_scraped_content(raw_html: str, max_length: int = 4000) -> str:
    """
    Sanitize raw scraped HTML content before passing it to LLM context or tool outputs.

    Steps:
    1. Parse HTML structure.
    2. Strip HTML comments (which frequently contain hidden instructions).
    3. Remove hidden/invisible DOM elements.
    4. Extract clean text.
    5. Clean invisible Unicode characters.
    6. Truncate to maximum allowed length.
    7. Wrap in trusted boundary markers.
    """
    if not raw_html or not raw_html.strip():
        return "[EXTERNAL_WEB_CONTENT_START]\n(Empty content)\n[EXTERNAL_WEB_CONTENT_END]"

    soup = BeautifulSoup(raw_html, "html.parser")

    # Step 1: Strip comments & hidden elements
    strip_html_comments(soup)
    strip_hidden_elements(soup)

    # Step 2: Extract text
    raw_text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    cleaned_text = "\n".join(lines)

    # Step 3: Strip invisible Unicode
    cleaned_text = strip_invisible_unicode(cleaned_text)

    # Step 4: Truncate safely
    truncated = cleaned_text[:max_length]
    suffix = "\n...[Content Truncated by Security Policy]" if len(cleaned_text) > max_length else ""

    # Step 5: Wrap in boundary markers
    return f"[EXTERNAL_WEB_CONTENT_START]\n{truncated}{suffix}\n[EXTERNAL_WEB_CONTENT_END]"
