"""LLM Normalizer Agent — rewrites raw scraped code into standard
`strategy(df)` functions via Groq, with an execute-to-validate step before
acceptance (the concrete fix for the v1 gap where any output merely
containing "def strategy(" was accepted unchecked). Falls back to direct
validation (no LLM call) when GROQ_API_KEY isn't configured, since the mock
strategy pool is already in the target format.

Caches the LLM-normalized result to disk for the same reason
`scraper_node`'s raw-scrape cache exists: without it, every graph
invocation re-spends a full pass of Groq LLM calls (one per candidate
strategy, real tokens-per-minute cost, observed hitting rate limits on a
free-tier key) even when `scraper_node`'s own raw input is already cached
and therefore identical run to run.
"""
from __future__ import annotations

import json
import os

from scaata.agents.state import AgentState
from scaata.config import DATA_CACHE_DIR
from scaata.strategies.normalizer import normalize_and_validate, validate_mock_strategies

NORMALIZED_CACHE_PATH = DATA_CACHE_DIR / "normalizer_node_normalized_cache.json"


def normalizer_node(state: AgentState) -> dict:
    raw_scripts = state["raw_scripts"]

    if not os.environ.get("GROQ_API_KEY"):
        return {"normalized_strategies": validate_mock_strategies(raw_scripts)}

    if NORMALIZED_CACHE_PATH.exists():
        with open(NORMALIZED_CACHE_PATH, "r", encoding="utf-8") as f:
            return {"normalized_strategies": json.load(f)}

    normalized = normalize_and_validate(raw_scripts)
    with open(NORMALIZED_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)

    return {"normalized_strategies": normalized}
