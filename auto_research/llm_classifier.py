"""LLM-based quadrant classifier.

Given evidence about a SaaS company's task, automation level, industry AI penetration and
whether the product uses AI, decide which disruption quadrant the company belongs to.
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel

# Locate prompt file relative to this module
_PROMPT_PATH = pathlib.Path(__file__).resolve().parent / "prompts" / "quadrant_classifier_prompt.txt"
_QUADRANT_TEXT = _PROMPT_PATH.read_text(encoding="utf-8")

# Build a simple chat prompt template with placeholders
_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _QUADRANT_TEXT),
        ("user", "Here is the evidence we have about {company}:\n{evidence}\n\nClassify the company now."),
    ]
)


class QuadrantClassifier:
    """Wraps an LLM call that returns quadrant + rationale."""

    def __init__(self, llm: BaseLanguageModel | None = None):
        # Default to GPT-4-o or GPT-4 if none supplied
        self.llm = llm or ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    def classify(self, company: str, evidence: dict[str, str]) -> Dict[str, str]:
        serialized_evidence = json.dumps(evidence, indent=2)
        messages = _CLASSIFIER_PROMPT.format_messages(
            company=company, evidence=serialized_evidence
        )
        response = self.llm.invoke(messages)

        import re

        # Attempt direct JSON parse.
        try:
            data: Dict[str, str] = json.loads(response.content)
        except json.JSONDecodeError:
            # Strip markdown fences or extract first JSON object.
            match = re.search(r"\{[\s\S]*\}", response.content)
            if not match:
                raise ValueError(
                    f"Classifier LLM did not return valid JSON for {company}: {response.content}"
                )
            data = json.loads(match.group(0))

        quadrant = data.get("Quadrant")
        rationale = data.get("Rationale")
        if not quadrant or not rationale:
            raise ValueError(
                f"Classifier JSON missing required keys for {company}: {data}"
            )
        return {"Company": company, "Quadrant": quadrant, "Rationale": rationale} 