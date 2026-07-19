"""LLM Normalizer Agent — rewrites raw scraped code into standard
`strategy(df)` functions via Groq, with an execute-to-validate step before
acceptance (the concrete fix for the v1 gap where any output merely
containing "def strategy(" was accepted unchecked). Falls back to direct
validation (no LLM call) when GROQ_API_KEY isn't configured, since the mock
strategy pool is already in the target format.
"""
from __future__ import annotations

import os

from scaata.agents.state import AgentState
from scaata.strategies.normalizer import normalize_and_validate, validate_mock_strategies


def normalizer_node(state: AgentState) -> dict:
    raw_scripts = state["raw_scripts"]

    if os.environ.get("GROQ_API_KEY"):
        normalized = normalize_and_validate(raw_scripts)
    else:
        normalized = validate_mock_strategies(raw_scripts)

    return {"normalized_strategies": normalized}
