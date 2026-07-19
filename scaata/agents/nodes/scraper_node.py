"""Strategy Scraper Agent — live GitHub search if GITHUB_PAT is configured,
otherwise falls back to a small hand-written mock pool so the graph can run
(and be tested) with no live API access.
"""
from __future__ import annotations

import os

from scaata.agents.state import AgentState
from scaata.strategies.scraper import filter_and_dedup, mock_strategies, scrape_strategies


def scraper_node(state: AgentState) -> dict:
    if os.environ.get("GITHUB_PAT"):
        raw = scrape_strategies()
        raw = filter_and_dedup(raw)
        if not raw:
            print("Live scrape returned nothing usable — falling back to mock strategies.")
            raw = mock_strategies()
    else:
        raw = mock_strategies()

    return {"raw_scripts": raw}
