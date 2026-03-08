"""
DuckDuckGo web search integration.
Uses the duckduckgo-search library for free web search.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class DuckDuckGoSearcher:
    """
    Web search using DuckDuckGo's free search API.
    No API key required.
    """

    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay

    def search(self, query: str, max_results: int = 8) -> list[dict]:
        """
        Search the web using DuckDuckGo.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of dicts with title, url, and snippet
        """
        logger.info(f"DuckDuckGo search: '{query}' (max={max_results})")

        for attempt in range(self.max_retries):
            try:
                results = self._ddg_search(query, max_results)
                logger.info(f"Search returned {len(results)} results")
                return results
            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))

        logger.error(f"All search attempts failed for: {query}")
        return []

    def _ddg_search(self, query: str, max_results: int) -> list[dict]:
        """Execute DuckDuckGo search."""
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
            return results
        except ImportError:
            logger.warning("duckduckgo_search not installed, using mock results")
            return self._mock_search(query)

    def _mock_search(self, query: str) -> list[dict]:
        """Mock search results when DDG is unavailable."""
        return [
            {
                "title": f"Regulatory Framework for {query}",
                "url": "https://example.com/framework",
                "snippet": f"Overview of regulatory requirements for {query}. "
                           "Includes compliance standards and best practices."
            }
        ]

    def search_for_document(self, framework_name: str) -> list[dict]:
        """Search specifically for official regulatory documents."""
        queries = [
            f"{framework_name} official standard PDF download",
            f"{framework_name} compliance document specifications",
            f"site:iso.org OR site:nist.gov OR site:pcisecuritystandards.org {framework_name}",
        ]

        all_results = []
        for query in queries[:2]:  # Limit queries
            results = self.search(query, max_results=5)
            all_results.extend(results)
            time.sleep(0.5)

        # Deduplicate by URL
        seen_urls = set()
        unique = []
        for r in all_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append(r)

        return unique[:8]
