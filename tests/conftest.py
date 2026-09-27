"""Repo-wide test fixtures.

Forces GITHUB_PAT/GROQ_API_KEY unset for every test by default. Several
graph-integration tests (`test_agents_graph.py`, `test_critique_node_hedge.py`,
`test_devils_advocate_graph_integration.py`) invoke the full LangGraph
pipeline via `run_inner_loop`, which runs `scraper_node`/`normalizer_node` --
these silently switch from the mock strategy pool to real GitHub search +
Groq LLM calls the moment those keys are present in the environment. Every
one of those tests' docstrings/comments already assumed "mock pool, no
network access," but that was only ever true by accident (the keys happened
to be unset), not by explicit test design. Caught live: a full test-suite
run hung for ~20 minutes and burned real Groq quota on
test_graph_compiles_cycles_and_terminates the first time the keys were
actually configured for the strategy-scraping feature. A single
autouse fixture here closes the whole class of bug (any current or future
test that transitively touches the graph) rather than patching each
affected file individually.

Tests that specifically want to exercise the live-key code path already
monkeypatch these explicitly in their own test bodies (e.g.
`test_llm_baseline.py`), which still works fine layered on top of this.

Also clears STRIPE_SECRET_KEY/STRIPE_WEBHOOK_SECRET/STRIPE_PRICE_ID
(Phase 19, `scaata.product.billing`) for the same reason: a real key
sitting in a dev `.env` must never let a test suite run silently hit
Stripe's live API or, worse, treat an unverified webhook payload as real
just because a secret happened to be configured in that environment.
"""
import pytest


@pytest.fixture(autouse=True)
def _default_offline_no_live_api_calls(monkeypatch):
    monkeypatch.delenv("GITHUB_PAT", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("STRIPE_PRICE_ID", raising=False)
