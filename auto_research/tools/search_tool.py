"""Web search tool wrapper for LangChain agents.
Requires a TAVILY_API_KEY environment variable.
"""
from __future__ import annotations

import os
from typing import Optional

try:
    # LangChain split community / core may differ by version
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
except ImportError:  # pragma: no cover
    from langchain.tools.tavily_search import TavilySearchResults  # type: ignore


def load_tavily_search_tool() -> TavilySearchResults:
    """Return a configured Tavily search tool instance.

    The tool returns the top *k* organic web results for a query.  These
    are usually sufficient context for further reasoning or summarisation.
    """
    api_key: str = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "TAVILY_API_KEY environment variable not set. Please export it to use the search tool."
        )

    # Instantiate wrapper and plug into tool.
    api_wrapper = TavilySearchAPIWrapper(tavily_api_key=api_key)

    # `max_results` controls number of search results returned.
    return TavilySearchResults(api_wrapper=api_wrapper, max_results=10) 