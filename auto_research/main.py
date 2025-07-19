"""Entry point to classify SaaS companies into AI disruption quadrants.

Usage
-----
$ python -m auto_research.main --input data/input.csv --output data/output.csv

Set environment variables beforehand:
    export OPENAI_API_KEY=...
    export TAVILY_API_KEY=...
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List

import pandas as pd
from langchain.chat_models import ChatOpenAI

from .agents.company_classifier_agent import CompanyResearchAgent
from .llm_classifier import QuadrantClassifier


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify SaaS companies by AI disruption quadrant")
    parser.add_argument("--input", type=pathlib.Path, default=pathlib.Path("data/input.csv"), help="Input CSV with a 'Company' column")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data/output.csv"), help="Path to write results CSV")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model name to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose agent logging")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])

    if not args.input.exists():
        raise FileNotFoundError(f"Input file {args.input} does not exist")

    df_input = pd.read_csv(args.input)
    if "Company" not in df_input.columns:
        raise ValueError("Input CSV must contain a 'Company' column")

    llm = ChatOpenAI(model_name=args.openai_model, temperature=0)

    research_agent = CompanyResearchAgent(llm=llm, verbose=args.verbose)
    classifier = QuadrantClassifier(llm=llm)

    results = []
    for company in df_input["Company"].dropna().unique():
        print(f"\n=== Processing {company} ===")
        evidence = research_agent.research(company)
        classification = classifier.classify(company, evidence)
        results.append(classification)
        print(f"Quadrant: {classification['Quadrant']}")

    df_out = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"\nResults written to {args.output.resolve()}")


if __name__ == "__main__":
    main() 