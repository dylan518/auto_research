"""Company research agent built with LangChain.

Given a company name, the agent performs web research using the provided
search/scraping tools and returns a structured evidence summary to feed
into the quadrant classifier.
"""
from __future__ import annotations

import json
from typing import List, Optional

from langchain.agents import AgentExecutor, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_core.language_models import BaseLanguageModel

from ..tools.search_tool import load_tavily_search_tool
from ..tools.web_scraper import load_web_pilot_tool


class CompanyResearchAgent:
    """Wrapper around a ReAct-style LangChain agent that gathers evidence."""

    SYSTEM_PROMPT = (
        "You are an investigative market analyst. Given the name of a private SaaS "
        "company you must gather publicly available information to answer four questions:\n"
        "1. What primary user task or workflow does the product support?\n"
        "2. Is that user task *highly automatable* or *low automatable*?\n"
        "3. Across the broader industry, is AI penetration for this workflow *high* or *low*?\n"
        "4. Does the company itself use or integrate AI today (Yes/No)?\n\n"
        "Search the web, read webpages and then RETURN ONLY the JSON dictionary, no markdown, no text before or after.\\n"
        "The dictionary MUST have exactly these keys: 'task', 'automation_level', 'ai_penetration', 'uses_ai'.\\n"
        "Example format (output exactly this schema, values are illustrative):\\n"
        "{\\n  \"task\": \"invoice processing\",\\n  \"automation_level\": \"high\",\\n  \"ai_penetration\": \"low\",\\n  \"uses_ai\": \"Yes\"\\n}\\n"
    )

    def __init__(self, llm: BaseLanguageModel, verbose: bool = False):
        tools = [load_tavily_search_tool()]
        web_pilot_tool = load_web_pilot_tool()
        if web_pilot_tool is not None:
            tools.append(web_pilot_tool)

        self.agent: AgentExecutor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            system=self.SYSTEM_PROMPT,
        )

    def research(self, company_name: str) -> dict[str, str]:
        """Run the agent and return evidence dict.

        Parameters
        ----------
        company_name : str
            Name of the SaaS vendor to research.
        """
        query = f"Provide answers in JSON only. Company: {company_name}"
        output: str = self.agent.run(query)

        try:
            evidence: dict[str, str] = json.loads(output)
        except json.JSONDecodeError:
            # When the agent spills non-JSON text, attempt to recover by extracting the first
            # JSON blob. This keeps the pipeline resilient to minor formatting errors.
            import re

            match = re.search(r"\{[\s\S]*\}", output)
            if not match:
                raise ValueError(
                    f"Could not find JSON in agent output for {company_name}: {output}"
                )
            evidence = json.loads(match.group(0))
        return evidence 