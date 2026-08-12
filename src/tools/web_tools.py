"""
Deep Web Research & Encyclopedic Retrieval Tools for Jarvis.
Includes DuckDuckGo Search, Wikipedia Search, and Direct Web URL Scraper.
"""

import logging
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from typing import List

logger = logging.getLogger(__name__)

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
    
    Args:
        url: Complete HTTP or HTTPS web address.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()

        title = soup.title.string.strip() if soup.title else url
        paragraphs = [p.get_text().strip() for p in soup.find_all(["p", "h1", "h2", "h3", "li"]) if p.get_text().strip()]
        
        content = "\n\n".join(paragraphs[:40])  # limit to top paragraphs
        if not content:
            content = soup.get_text(separator="\n", strip=True)[:3000]

        return f"=== Web Page: {title} ===\nURL: {url}\n\n{content[:4000]}"
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        return f"Failed to retrieve web page from '{url}': {str(e)}"

def get_web_tools() -> List[BaseTool]:
    """Retrieve the standard suite of web research tools."""
    return [
        DuckDuckGoSearchRun(description="Search the web for up-to-date information, news, current events, and live data."),
        wikipedia_lookup,
        read_webpage_content
    ]
