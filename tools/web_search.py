"""
Date-filtered web search tool using SerpAPI.

Ensures no future data leakage by filtering search results to before the sample date.
"""

import os
import logging
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


class DateFilteredWebSearch:
    """
    Web search tool that filters results by date to prevent future data leakage.

    Uses SerpAPI with date range filtering (tbs parameter).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
    ):
        """
        Initialize the date-filtered web search.

        Args:
            api_key: SerpAPI API key (defaults to SERPAPI_API_KEY env var)
            max_results: Maximum number of search results to return (1-100)
        """
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        # Validate and clamp max_results to reasonable range
        self.max_results = max(1, min(100, max_results))

        if not self.api_key:
            logger.warning("SERPAPI_API_KEY not set. Web search will not work.")

    def create_tool(self, sample_date: datetime):
        """
        Create a search tool bound to a specific sample date.

        All search results will be filtered to be from before the sample_date
        to prevent future data leakage.

        Args:
            sample_date: The date of the sample. Results will be filtered
                        to be from before this date.

        Returns:
            A LangChain tool that can be used for web search
        """
        api_key = self.api_key
        max_results = self.max_results

        # Format date for SerpAPI tbs parameter: M/D/YYYY
        # Using strftime with platform-independent formatting
        max_date_str = f"{sample_date.month}/{sample_date.day}/{sample_date.year}"

        @tool
        def web_search(query: str) -> str:
            """
            Search the web for information relevant to making a prediction.

            Use this tool to find news, data, and information that would help
            you make an informed forecast. Results are automatically filtered
            to avoid future data leakage.

            Args:
                query: The search query to find relevant information

            Returns:
                A formatted string containing search results with titles,
                snippets, and sources
            """
            if not api_key:
                return "Error: SERPAPI_API_KEY not configured. Cannot perform web search."

            try:
                from serpapi import GoogleSearch
            except ImportError:
                return "Error: google-search-results package not installed. Run: pip install google-search-results"

            try:
                # Configure SerpAPI search with date filtering
                params = {
                    "q": query,
                    "api_key": api_key,
                    # tbs=cdr:1,cd_min:START,cd_max:END for custom date range
                    "tbs": f"cdr:1,cd_min:1/1/2020,cd_max:{max_date_str}",
                    "num": max_results,
                    "hl": "en",
                    "gl": "us",
                }

                logger.info(f"Web search: '{query}' (before {max_date_str})")

                search = GoogleSearch(params)
                results = search.get_dict()

                # Check for errors
                if "error" in results:
                    logger.error(f"Search API error: {results['error']}")
                    return f"Search error: {results['error']}"

                # Extract organic results
                organic_results = results.get("organic_results", [])
                logger.info(f"Found {len(organic_results)} results")

                if not organic_results:
                    return f"No results found for query: {query}"

                # Format results
                formatted_results = []
                for i, result in enumerate(organic_results[:max_results], 1):
                    title = result.get("title", "No title")
                    snippet = result.get("snippet", "No description available")
                    link = result.get("link", "")
                    date = result.get("date", "")

                    result_str = f"[{i}] {title}"
                    if date:
                        result_str += f" ({date})"
                    result_str += f"\n    {snippet}"
                    if link:
                        result_str += f"\n    Source: {link}"

                    formatted_results.append(result_str)

                header = f"Search results for '{query}' (filtered to before {max_date_str}):\n"
                return header + "\n\n".join(formatted_results)

            except Exception as e:
                logger.error(f"Web search error: {e}")
                return f"Error performing web search: {str(e)}"

        return web_search


def create_search_tool(
    sample_date: datetime,
    api_key: Optional[str] = None,
    max_results: int = 5,
):
    """
    Convenience function to create a date-filtered search tool.

    Args:
        sample_date: The sample date for filtering results
        api_key: Optional SerpAPI key
        max_results: Maximum results to return

    Returns:
        A LangChain tool for web search
    """
    searcher = DateFilteredWebSearch(api_key=api_key, max_results=max_results)
    return searcher.create_tool(sample_date)
