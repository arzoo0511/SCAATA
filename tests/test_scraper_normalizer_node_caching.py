"""Tests for scraper_node/normalizer_node's disk caching of the live
GitHub scrape + Groq LLM normalization pass. Without this, every graph
invocation with GITHUB_PAT/GROQ_API_KEY configured re-pays a real,
observed ~10-minute cost (live GitHub search + one Groq call per
candidate, hitting free-tier rate limits) even across repeated calls with
identical input -- caught live while running scaata.agents.orchestrator.
run_inner_loop for the first time with real keys configured.

conftest.py's autouse fixture forces GITHUB_PAT/GROQ_API_KEY unset for
every test by default, so these tests explicitly re-set them via
monkeypatch to exercise the cached-key code path.
"""
import json

import scaata.agents.nodes.normalizer_node as normalizer_node_module
import scaata.agents.nodes.scraper_node as scraper_node_module
from scaata.agents.nodes.normalizer_node import normalizer_node
from scaata.agents.nodes.scraper_node import scraper_node


def test_scraper_node_uses_cache_without_rescraping(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_PAT", "fake-token-for-test")
    cache_path = tmp_path / "raw_cache.json"
    monkeypatch.setattr(scraper_node_module, "RAW_SCRIPTS_CACHE_PATH", cache_path)

    cached_raw = [{"source": "cached:one", "code": "def strategy(df):\n    return df"}]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached_raw, f)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("scrape_strategies should not be called when a cache hit is available")

    monkeypatch.setattr(scraper_node_module, "scrape_strategies", _fail_if_called)

    result = scraper_node({})
    assert result["raw_scripts"] == cached_raw


def test_scraper_node_populates_cache_after_a_live_scrape(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_PAT", "fake-token-for-test")
    cache_path = tmp_path / "raw_cache.json"
    monkeypatch.setattr(scraper_node_module, "RAW_SCRIPTS_CACHE_PATH", cache_path)

    fresh_raw = [{"source": "live:one", "code": "def strategy(df):\n    import numpy as np\n    return np.zeros(len(df))"}]
    monkeypatch.setattr(scraper_node_module, "scrape_strategies", lambda: fresh_raw)
    monkeypatch.setattr(scraper_node_module, "filter_and_dedup", lambda raw: raw)

    assert not cache_path.exists()
    result = scraper_node({})
    assert result["raw_scripts"] == fresh_raw
    assert cache_path.exists()

    with open(cache_path, "r", encoding="utf-8") as f:
        assert json.load(f) == fresh_raw


def test_scraper_node_falls_back_to_mock_without_github_pat(monkeypatch):
    monkeypatch.delenv("GITHUB_PAT", raising=False)
    result = scraper_node({})
    assert len(result["raw_scripts"]) > 0
    assert all("source" in s and s["source"].startswith("mock:") for s in result["raw_scripts"])


def test_normalizer_node_uses_cache_without_recalling_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
    cache_path = tmp_path / "normalized_cache.json"
    monkeypatch.setattr(normalizer_node_module, "NORMALIZED_CACHE_PATH", cache_path)

    cached_normalized = [{"source": "cached:one", "clean_code": "def strategy(df):\n    return df"}]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached_normalized, f)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("normalize_and_validate should not be called when a cache hit is available")

    monkeypatch.setattr(normalizer_node_module, "normalize_and_validate", _fail_if_called)

    result = normalizer_node({"raw_scripts": []})
    assert result["normalized_strategies"] == cached_normalized


def test_normalizer_node_populates_cache_after_live_normalization(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
    cache_path = tmp_path / "normalized_cache.json"
    monkeypatch.setattr(normalizer_node_module, "NORMALIZED_CACHE_PATH", cache_path)

    fresh_normalized = [{"source": "live:one", "clean_code": "def strategy(df):\n    return df"}]
    monkeypatch.setattr(normalizer_node_module, "normalize_and_validate", lambda raw: fresh_normalized)

    assert not cache_path.exists()
    result = normalizer_node({"raw_scripts": [{"source": "x", "code": "..."}]})
    assert result["normalized_strategies"] == fresh_normalized
    assert cache_path.exists()


def test_normalizer_node_falls_back_to_direct_validation_without_groq_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    raw_scripts = [{"source": "mock:x", "code": "def strategy(df):\n    import numpy as np\n    return np.zeros(len(df), dtype=int)\n"}]

    result = normalizer_node({"raw_scripts": raw_scripts})
    assert len(result["normalized_strategies"]) == 1
    assert result["normalized_strategies"][0]["source"] == "mock:x"
