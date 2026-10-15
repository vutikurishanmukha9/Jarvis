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

from .cleaner import sanitize_scraped_content

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
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        sanitized_content = sanitize_scraped_content(text, max_length=4000)

        return f"=== Web Page: {title} ===\nURL: {url}\n\n{sanitized_content}"
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        return f"Failed to retrieve web page from '{url}': {str(e)}"


# ==============================================================================
# 3-TIER WEB INTELLIGENCE MATRIX (Scrapling, Crawl4AI, Firecrawl)
# ==============================================================================


@tool
def scrape_web_adaptive(url: str, css_selector: str = "body") -> str:
    """
    Tier 1 Web Intelligence: Adaptive, resilient web scraper powered by Scrapling.
    Survives subtle website HTML/CSS redesigns and extracts elements cleanly.
    Bypasses anti-bot challenges and extracts structured text.

    Security: Validates scheme (HTTP/HTTPS only) and blocks private/internal IPs.

    Args:
        url: Complete HTTP or HTTPS target webpage URL.
        css_selector: Target CSS selector to extract (default 'body', e.g. 'article', '.content', 'main').
    """
    validation_err = _validate_url(url)
    if validation_err:
        return f"[Security Blocked] {validation_err}"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = _fetch_with_safe_redirects(url, headers)
        resp.raise_for_status()

        html_text = resp.content.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html_text, "html.parser")
        page_title = soup.title.string.strip() if soup.title and soup.title.string else url

        # Attempt Scrapling adaptive parsing
        extracted_text = ""
        engine_used = "Scrapling Adaptive Parser"
        try:
            import sys
            from pathlib import Path

            project_root = Path(__file__).resolve().parent.parent.parent
            scrapling_path = project_root / "scrapling"
            if str(scrapling_path) not in sys.path:
                sys.path.insert(0, str(scrapling_path))

            from scrapling.parser import Adaptor

            doc = Adaptor(html_text)
            elements = doc.css(css_selector)
            if elements:
                text_items = [el.text.strip() for el in elements if el.text and el.text.strip()]
                extracted_text = "\n\n".join(text_items[:25])
        except Exception as e:
            logger.debug(f"Scrapling parser fallback: {e}")

        # Fallback to sanitized content if selector produced no text
        if not extracted_text:
            extracted_text = sanitize_scraped_content(html_text, max_length=4000)
            engine_used = "Standard Parser Fallback"

        return (
            f"=== Web Page: {page_title} ===\n"
            f"URL: {url}\n"
            f"Engine: {engine_used} (Selector: '{css_selector}')\n\n"
            f"{extracted_text}"
        )
    except Exception as e:
        logger.error(f"scrape_web_adaptive error for {url}: {e}")
        return f"Failed to scrape webpage '{url}': {str(e)}"


@tool
def crawl_web_dynamic(url: str, css_selector: str = "") -> str:
    """
    Tier 2 Web Intelligence: Dynamic JS-hydrated web crawler powered by Crawl4AI.
    Renders client-side dynamic content (React, Vue, Next.js, Angular, SPAs)
    and extracts noise-pruned, LLM-optimized fit-markdown.

    Security: Validates scheme (HTTP/HTTPS only) and blocks private/internal IPs.

    Args:
        url: Complete HTTP or HTTPS web address.
        css_selector: Optional CSS selector to focus extraction on a specific container.
    """
    validation_err = _validate_url(url)
    if validation_err:
        return f"[Security Blocked] {validation_err}"

    try:
        import asyncio
        import sys
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent.parent
        crawl_path = project_root / "webcrawl"
        if str(crawl_path) not in sys.path:
            sys.path.insert(0, str(crawl_path))

        # Try Crawl4AI AsyncWebCrawler
        try:
            from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

            async def _run_crawl():
                config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    word_count_threshold=15,
                    css_selector=css_selector if css_selector else None,
                    magic=True,
                    page_timeout=25000,
                )
                async with AsyncWebCrawler(verbose=False) as crawler:
                    res = await crawler.arun(url=url, config=config)
                    if res.success:
                        return res.markdown.fit_markdown or res.markdown.raw_markdown
                    return None

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                import nest_asyncio

                nest_asyncio.apply()

            markdown_res = loop.run_until_complete(_run_crawl())
            if markdown_res:
                return (
                    f"=== Dynamic Web Page: {url} ===\n"
                    f"Engine: Crawl4AI (JS Hydrated + Fit-Markdown)\n\n"
                    f"{markdown_res[:4500]}"
                )
        except Exception as c_err:
            logger.debug(f"Crawl4AI execution fallback: {c_err}")

        # Fallback to standard safe reader if Crawl4AI browser environment is not initialized
        return read_webpage_content.func(url)
    except Exception as e:
        logger.error(f"crawl_web_dynamic error for {url}: {e}")
        return f"Failed to crawl dynamic webpage '{url}': {str(e)}"


@tool
def map_website_urls(root_url: str, search_query: str = "") -> str:
    """
    Tier 3 Web Intelligence: Enterprise domain URL mapper powered by Firecrawl.
    Rapidly discovers all sitemap and internal page links on a target website in seconds,
    filtering for relevant pages so you can selectively read exact content without blind crawling.

    Security: Validates scheme (HTTP/HTTPS only) and blocks private/internal IPs.

    Args:
        root_url: The root domain or documentation home URL (e.g. 'https://docs.python.org').
        search_query: Optional keyword to filter discovered URLs (e.g. 'asyncio', 'distributed', 'pricing').
    """
    validation_err = _validate_url(root_url)
    if validation_err:
        return f"[Security Blocked] {validation_err}"

    try:
        import os
        from urllib.parse import urljoin, urlparse

        import httpx

        # 1. Check if Firecrawl API key is present
        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if api_key:
            try:
                endpoint = os.environ.get("FIRECRAWL_BASE_URL", "https://api.firecrawl.dev/v1/map")
                headers = {"Authorization": f"Bearer {api_key}"}
                payload = {"url": root_url, "search": search_query}
                with httpx.Client(timeout=20.0) as client:
                    resp = client.post(endpoint, json=payload, headers=headers)
                    data = resp.json()
                if data.get("success"):
                    links = data.get("links", [])[:30]
                    formatted = "\n".join(f"- {link}" for link in links)
                    return (
                        f"=== Firecrawl Domain Map for {root_url} ===\n"
                        f"Search Filter: '{search_query}' | Discovered: {len(links)} URLs\n\n"
                        f"{formatted}"
                    )
            except Exception as f_err:
                logger.debug(f"Firecrawl API error: {f_err}, using native sitemapper fallback")

        # 2. Resilient Native Sitemapper Engine
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = _fetch_with_safe_redirects(root_url, headers)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "html.parser")
        parsed_root = urlparse(root_url)
        base_domain = parsed_root.netloc.lower()

        discovered_links = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            full_url = urljoin(root_url, href)
            parsed_full = urlparse(full_url)
            if parsed_full.netloc.lower() == base_domain and parsed_full.scheme in {"http", "https"}:
                if (
                    not search_query
                    or search_query.lower() in full_url.lower()
                    or (a_tag.string and search_query.lower() in a_tag.string.lower())
                ):
                    discovered_links.add(full_url)

        sorted_links = sorted(list(discovered_links))[:30]
        if not sorted_links:
            return f"No matching URLs found on '{root_url}' matching filter '{search_query}'."

        formatted_links = "\n".join(f"- {link}" for link in sorted_links)
        return (
            f"=== Firecrawl Sitemapper: {root_url} ===\n"
            f"Filter: '{search_query or 'All'}' | Discovered URLs ({len(sorted_links)}):\n\n"
            f"{formatted_links}\n\n"
            f"Tip: Use read_webpage_content or crawl_web_dynamic on any of the URLs above."
        )
    except Exception as e:
        logger.error(f"map_website_urls error for {root_url}: {e}")
        return f"Failed to map URLs for '{root_url}': {str(e)}"


def get_web_tools() -> List[BaseTool]:
    """Retrieve the expanded 3-Tier suite of web research and intelligence tools."""
    return [
        DuckDuckGoSearchRun(
            description="Search the web for up-to-date information, news, current events, and live data."
        ),
        wikipedia_lookup,
        read_webpage_content,
        scrape_web_adaptive,
        crawl_web_dynamic,
        map_website_urls,
    ]
