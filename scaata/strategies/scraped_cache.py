"""Caches the (expensive, rate-limited) GitHub-scrape + LLM-normalize
result to disk so `get_full_strategy_pool` doesn't have to re-scrape and
re-spend LLM quota on every run. A live run costs real GitHub search calls
(a fetch per candidate file) and ~1 Groq call per candidate, routinely
takes minutes, and hit the free-tier Groq TPM rate limit repeatedly the
first time this was run live -- reproducible speed and cost, not just
convenience, is the point.
"""
from __future__ import annotations

import json
import time

from scaata.config import DATA_CACHE_DIR

SCRAPED_STRATEGIES_CACHE_PATH = DATA_CACHE_DIR / "scraped_strategies_cache.json"


def load_cached_scraped_strategies() -> list[dict] | None:
    """Returns the cached `[{"source", "clean_code"}, ...]` list, or None
    if no cache file exists yet."""
    if not SCRAPED_STRATEGIES_CACHE_PATH.exists():
        return None
    with open(SCRAPED_STRATEGIES_CACHE_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["strategies"]


def save_scraped_strategies_cache(normalized_scraped: list[dict]) -> None:
    payload = {"cached_at_unix": time.time(), "strategies": normalized_scraped}
    with open(SCRAPED_STRATEGIES_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_full_strategy_pool(force_rescrape: bool = False) -> list[dict]:
    """The real, current strategy pool: mock + library (always, free,
    instant) + GitHub-scraped-and-LLM-normalized (cached after the first
    live run; pass `force_rescrape=True` to refresh from GitHub/Groq).
    Every entry is already execute-to-validated (see
    `scaata.strategies.normalizer`) before being cached, so nothing here
    needs re-validation by the caller.
    """
    from scaata.strategies.library import library_strategies
    from scaata.strategies.normalizer import normalize_and_validate, validate_mock_strategies
    from scaata.strategies.scraper import filter_and_dedup, mock_strategies, scrape_strategies

    baseline = validate_mock_strategies(mock_strategies() + library_strategies())

    if force_rescrape:
        scraped = None
    else:
        scraped = load_cached_scraped_strategies()

    if scraped is None:
        raw = scrape_strategies()
        deduped = filter_and_dedup(raw)
        scraped = normalize_and_validate(deduped)
        save_scraped_strategies_cache(scraped)

    return baseline + scraped
