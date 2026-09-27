"""Strategy Scraper Agent — live GitHub search if GITHUB_PAT is configured,
otherwise falls back to a small hand-written mock pool so the graph can run
(and be tested) with no live API access.

Caches the raw scrape (post keyword-filter/dedup, pre-LLM) to disk. Found
necessary live: without this, every single graph invocation with
GITHUB_PAT configured re-scrapes GitHub from scratch, a real, observed
~10-minute cost hit repeatedly by any caller (including
`scaata.agents.orchestrator.run_inner_loop`, which can run several times
per ticker/ablation) — the exact same cost `scaata.strategies.scraped_cache`
was built to avoid for direct callers, just not wired into the graph nodes
until now.
"""
from __future__ import annotations

import json
import os

from scaata.agents.state import AgentState
from scaata.config import DATA_CACHE_DIR
from scaata.strategies.scraper import filter_and_dedup, mock_strategies, scrape_strategies

RAW_SCRIPTS_CACHE_PATH = DATA_CACHE_DIR / "scraper_node_raw_cache.json"


def scraper_node(state: AgentState) -> dict:
    if not os.environ.get("GITHUB_PAT"):
        return {"raw_scripts": mock_strategies()}

    if RAW_SCRIPTS_CACHE_PATH.exists():
        with open(RAW_SCRIPTS_CACHE_PATH, "r", encoding="utf-8") as f:
            return {"raw_scripts": json.load(f)}

    raw = scrape_strategies()
    raw = filter_and_dedup(raw)
    if not raw:
        print("Live scrape returned nothing usable — falling back to mock strategies.")
        raw = mock_strategies()
    else:
        with open(RAW_SCRIPTS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)

    return {"raw_scripts": raw}
