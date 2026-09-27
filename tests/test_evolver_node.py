"""Tests for the optional evolver node (Phase 11): must be an exact no-op
by default (the existing graph tests already cover this implicitly, since
they all pass with the node wired in), and must actually inject evolved
strategies into the pool when explicitly enabled.
"""
import numpy as np
import pandas as pd

from scaata.agents.nodes import evolver_node as evolver_node_module
from scaata.agents.nodes.evolver_node import evolver_node
from scaata.config import FEATURE_COLUMNS


def _make_state(n=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}, index=dates)
    df["rsi"] = rng.uniform(0, 100, n)
    df["Close"] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    df["Ticker"] = "TEST"
    return {"train_df": df, "feature_columns": FEATURE_COLUMNS, "iteration": 0}


def test_evolver_node_is_a_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(evolver_node_module, "ENABLE_STRATEGY_EVOLUTION", False)
    result = evolver_node(_make_state())
    assert result == {}


def test_evolver_node_adds_strategies_when_enabled(monkeypatch):
    monkeypatch.setattr(evolver_node_module, "ENABLE_STRATEGY_EVOLUTION", True)
    monkeypatch.setattr(evolver_node_module, "EVOLUTION_POPULATION_SIZE", 8)
    monkeypatch.setattr(evolver_node_module, "EVOLUTION_N_GENERATIONS", 3)
    monkeypatch.setattr(evolver_node_module, "EVOLUTION_TOP_K_TO_POOL", 2)

    state = _make_state()
    state["normalized_strategies"] = [{"source": "mock:existing", "clean_code": "def strategy(df):\n    import numpy as np\n    return np.zeros(len(df), dtype=int)\n"}]

    result = evolver_node(state)

    assert "normalized_strategies" in result
    assert len(result["normalized_strategies"]) == 1 + 2  # existing + top-2 evolved
    assert result["normalized_strategies"][0]["source"] == "mock:existing"
    assert all(s["source"].startswith("evolved:") for s in result["normalized_strategies"][1:])
    for s in result["normalized_strategies"][1:]:
        assert "def strategy(df):" in s["clean_code"]
