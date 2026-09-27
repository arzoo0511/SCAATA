"""Structural tests for the monolithic-vs-decomposed comparison harness.

These confirm the harness runs end-to-end and produces well-formed output
for both schemes — they do NOT validate the actual research claim (whether
LLM self-assessed confidence tracks real profitability), since that
requires a live GROQ_API_KEY and is not something a offline/CI test can
meaningfully assert about mocked random weights.
"""
import numpy as np
import pandas as pd

import scaata.agents.nodes.evolver_node as evolver_node_module
from scaata.config import FEATURE_COLUMNS
from scaata.evaluation.agent_comparison import (
    actual_strategy_profitability,
    compare_monolithic_vs_decomposed,
)
from scaata.strategies.scraper import mock_strategies


def _make_synthetic_train_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}, index=dates)
    df["Close"] = close
    df["returns"] = pd.Series(close, index=dates).pct_change().fillna(0)
    df["Volume"] = rng.integers(1_000_000, 5_000_000, n)
    df["volume_ma_30"] = df["Volume"].rolling(30, min_periods=1).mean()
    df["Ticker"] = "TEST"
    return df


def test_actual_profitability_is_computed_per_strategy():
    train_df = _make_synthetic_train_df()
    signals = [{"source": "a", "signals": np.ones(len(train_df))}, {"source": "b", "signals": -np.ones(len(train_df))}]
    profits = actual_strategy_profitability(signals, train_df)
    assert set(profits.keys()) == {"a", "b"}
    # strategy "b" is the exact negation of "a", so their profitability must be exact opposites
    assert profits["a"] == -profits["b"]


def test_comparison_harness_runs_end_to_end_without_llm_access(monkeypatch):
    # This test's assertions assume a closed strategy set (exactly `raw`,
    # nothing more) -- that's specifically what Phase 11 strategy evolution
    # (now on by default, see scaata.config.ENABLE_STRATEGY_EVOLUTION) is
    # designed to violate, since `compare_monolithic_vs_decomposed` runs the
    # full graph via `run_inner_loop`. This test's actual purpose (the
    # harness runs end-to-end and produces well-formed output without LLM
    # access) is orthogonal to evolution, so disable it here rather than
    # loosen the assertions and lose the "no extra strategies snuck in"
    # check for the case this test actually cares about.
    monkeypatch.setattr(evolver_node_module, "ENABLE_STRATEGY_EVOLUTION", False)

    train_df = _make_synthetic_train_df()
    raw = mock_strategies()

    result = compare_monolithic_vs_decomposed(train_df, raw, FEATURE_COLUMNS, max_iterations=3)

    assert set(result["decomposed_weights"].keys()) <= {s["source"] for s in raw}
    assert set(result["monolithic_weights"].keys()) <= {s["source"] for s in raw}
    assert result["n_decomposed_llm_calls"] == result["n_monolithic_llm_calls"] == len(raw)
    assert 1 <= result["decomposed_iterations"] <= 3
