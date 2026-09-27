"""Unit tests for the scraped-strategy disk cache (avoids re-spending
GitHub/Groq quota on every run). Uses monkeypatched cache paths and a fake
scraper/normalizer so this stays a fast, offline test of the caching logic
itself, not a live-network test.
"""
import json

import scaata.strategies.scraped_cache as scraped_cache


def test_load_cached_returns_none_when_no_cache_file(tmp_path, monkeypatch):
    monkeypatch.setattr(scraped_cache, "SCRAPED_STRATEGIES_CACHE_PATH", tmp_path / "does_not_exist.json")
    assert scraped_cache.load_cached_scraped_strategies() is None


def test_save_then_load_roundtrips(tmp_path, monkeypatch):
    path = tmp_path / "cache.json"
    monkeypatch.setattr(scraped_cache, "SCRAPED_STRATEGIES_CACHE_PATH", path)

    strategies = [{"source": "https://github.com/x/y", "clean_code": "def strategy(df):\n    return df"}]
    scraped_cache.save_scraped_strategies_cache(strategies)

    assert path.exists()
    loaded = scraped_cache.load_cached_scraped_strategies()
    assert loaded == strategies


def test_get_full_strategy_pool_uses_cache_without_rescraping(tmp_path, monkeypatch):
    path = tmp_path / "cache.json"
    monkeypatch.setattr(scraped_cache, "SCRAPED_STRATEGIES_CACHE_PATH", path)
    cached = [{"source": "cached:one", "clean_code": "def strategy(df):\n    return df"}]
    scraped_cache.save_scraped_strategies_cache(cached)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("scrape_strategies should not be called when a cache hit is available")

    monkeypatch.setattr("scaata.strategies.scraper.scrape_strategies", _fail_if_called)

    pool = scraped_cache.get_full_strategy_pool()
    sources = [s["source"] for s in pool]
    assert "cached:one" in sources


def test_force_rescrape_bypasses_cache(tmp_path, monkeypatch):
    path = tmp_path / "cache.json"
    monkeypatch.setattr(scraped_cache, "SCRAPED_STRATEGIES_CACHE_PATH", path)
    scraped_cache.save_scraped_strategies_cache([{"source": "stale:one", "clean_code": "def strategy(df):\n    return df"}])

    called = {"scrape": False}

    def _fake_scrape(*args, **kwargs):
        called["scrape"] = True
        return []

    monkeypatch.setattr("scaata.strategies.scraper.scrape_strategies", _fake_scrape)
    monkeypatch.setattr("scaata.strategies.scraper.filter_and_dedup", lambda raw: raw)
    monkeypatch.setattr("scaata.strategies.normalizer.normalize_and_validate", lambda deduped: [])

    scraped_cache.get_full_strategy_pool(force_rescrape=True)
    assert called["scrape"] is True
