"""
Deep Web Research & Encyclopedic Retrieval Tools for Jarvis.
Includes DuckDuckGo Search, Wikipedia Search, and Direct Web URL Scraper.

SECURITY: URL fetching validates schemes, blocks private/internal IPs,
limits response size, and restricts redirect chains.
"""

import ipaddress
import logging
import socket
from typing import List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)

# Security constants
MAX_RESPONSE_BYTES = 500_000  # 500 KB
MAX_REDIRECTS = 3
ALLOWED_SCHEMES = {"http", "https"}

# Private/internal IP ranges that must be blocked (SSRF protection)
BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("10.0.0.0/8"),  # Private Class A
    ipaddress.ip_network("172.16.0.0/12"),  # Private Class B
    ipaddress.ip_network("192.168.0.0/16"),  # Private Class C
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local
    ipaddress.ip_network("0.0.0.0/8"),  # Unspecified
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),  # IPv6 private
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
]


def _validate_url(url: str) -> str:
    """
    Validate a URL for security before fetching.
    Returns an error string if the URL is rejected, empty string if safe.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return f"Invalid URL format: '{url}'"

    # 1. Scheme validation
    if parsed.scheme not in ALLOWED_SCHEMES:
        return f"Blocked URL scheme '{parsed.scheme}://'. Only {', '.join(ALLOWED_SCHEMES)} are allowed."

    # 2. Hostname must exist
    hostname = parsed.hostname
    if not hostname:
        return f"URL has no hostname: '{url}'"

    # 3. DNS resolution and private IP check
    try:
        addr_infos = socket.getaddrinfo(hostname, parsed.port or 80, proto=socket.IPPROTO_TCP)
        for _family, _, _, _, sockaddr in addr_infos:
            ip = ipaddress.ip_address(sockaddr[0])
            for network in BLOCKED_IP_NETWORKS:
                if ip in network:
                    return (
                        f"Blocked: URL '{hostname}' resolves to private/internal IP {ip}. "
                        f"Fetching internal network resources is not permitted."
                    )
    except socket.gaierror:
        return f"DNS resolution failed for hostname: '{hostname}'"
    except Exception as e:
        return f"URL validation error for '{hostname}': {str(e)}"

    return ""  # Safe


def _fetch_with_safe_redirects(url: str, headers: dict) -> requests.Response:
    """Fetch a URL while validating every redirect destination against SSRF."""
    current_url = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        validation_error = _validate_url(current_url)
        if validation_error:
            raise ValueError(validation_error)

        response = requests.get(
            current_url,
            headers=headers,
            timeout=12,
            allow_redirects=False,
            stream=True,
        )
        if not response.is_redirect:
            return response

        if redirect_count == MAX_REDIRECTS:
            response.close()
            raise ValueError(f"Too many redirects. Maximum allowed: {MAX_REDIRECTS}.")
        location = response.headers.get("Location")
        if not location:
            response.close()
            raise ValueError("Redirect response did not contain a Location header.")
        current_url = requests.compat.urljoin(current_url, location)
        response.close()

    raise ValueError("Redirect handling failed unexpectedly.")


@tool
def wikipedia_lookup(query: str) -> str:
    """
    Search Wikipedia for encyclopedic knowledge, background summaries,
    historical facts, biographies, scientific concepts, and structured information.

    Args:
        query: The search term or subject.
    """
    try:
        import wikipedia

        # Search for page titles
        search_results = wikipedia.search(query, results=3)
        if not search_results:
            return f"No Wikipedia articles found for '{query}'."

        # Fetch summary of best match
        best_title = search_results[0]
        page = wikipedia.page(best_title, auto_suggest=False)
        summary = wikipedia.summary(best_title, sentences=5, auto_suggest=False)

        output = f"=== Wikipedia: {page.title} ===\nURL: {page.url}\n\n{summary}\n"
        if len(search_results) > 1:
            output += f"\nRelated Topics: {', '.join(search_results[1:])}"
        return output
    except Exception as e:
        logger.warning(f"Wikipedia lookup error: {str(e)}")
        return f"Wikipedia lookup error for '{query}': {str(e)}"


@tool
def read_webpage_content(url: str) -> str:
    """
    Fetch and read the full text content from a specific web URL or article link.
    Useful when a web search returns an interesting link and full details are required.

    Security: Only HTTP/HTTPS URLs are allowed. Private/internal IPs are blocked.
    Response size is capped at 500KB with a maximum of 3 redirects.

    Args:
        url: Complete HTTP or HTTPS web address.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = _fetch_with_safe_redirects(url, headers)
        resp.raise_for_status()

        # Check content size before reading full body
        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_RESPONSE_BYTES:
            return (
                f"[Security] Response too large ({int(content_length) // 1024}KB). "
                f"Maximum allowed: {MAX_RESPONSE_BYTES // 1024}KB."
            )

        # Read incrementally; accessing ``resp.content`` would download an unbounded body.
        chunks = []
        total_bytes = 0
        for chunk in resp.iter_content(chunk_size=16_384):
            if not chunk:
                continue
            total_bytes += len(chunk)
            if total_bytes > MAX_RESPONSE_BYTES:
                return f"[Security] Response exceeded {MAX_RESPONSE_BYTES // 1024}KB."
            chunks.append(chunk)
        content = b"".join(chunks)
        text = content.decode("utf-8", errors="ignore")

        soup = BeautifulSoup(text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else url
        paragraphs = [
            p.get_text().strip() for p in soup.find_all(["p", "h1", "h2", "h3", "li"]) if p.get_text().strip()
        ]

        page_content = "\n\n".join(paragraphs[:40])  # limit to top paragraphs
        if not page_content:
            page_content = soup.get_text(separator="\n", strip=True)[:3000]

        return f"=== Web Page: {title} ===\nURL: {url}\n\n{page_content[:4000]}"
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        return f"Failed to retrieve web page from '{url}': {str(e)}"


def get_web_tools() -> List[BaseTool]:
    """Retrieve the standard suite of web research tools."""
    return [
        DuckDuckGoSearchRun(
            description="Search the web for up-to-date information, news, current events, and live data."
        ),
        wikipedia_lookup,
        read_webpage_content,
    ]
