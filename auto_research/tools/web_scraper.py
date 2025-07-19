"""Optional Web page scraper / summariser tool.

WebPilot is useful for retrieving and cleaning the HTML of a given URL
so that an LLM can read the content.  It is *optional* for classification –
if WebPilot or similar dependency is unavailable, downstream agents can
still function using search results alone.
"""
from __future__ import annotations

import logging
from typing import Optional

try:
    from langchain_community.tools.web_pilot import WebPilot  # Available in LC community >=0.0.34
except ImportError:  # pragma: no cover
    WebPilot = None  # type: ignore

logger = logging.getLogger(__name__)


def load_web_pilot_tool() -> Optional["WebPilot"]:
    """Return an instance of WebPilot if the dependency is present.

    Returns
    -------
    WebPilot | None
        Configured tool or ``None`` when WebPilot could not be imported.
    """
    if WebPilot is None:
        logger.warning("WebPilot could not be imported – proceeding without it.")
        return None

    return WebPilot() 