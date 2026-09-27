"""Integration tests for the Phase 10 Hedge wiring in `critique_node`: the
optional sentiment expert must actually appear (and be trackable) when
`train_df` has a `sentiment_score` column, novelty/regret bookkeeping must
be populated, and everything must degrade gracefully (falls back to the old
strategy-only behavior) when sentiment is absent — exercising the one
integration path (`has_sentiment_expert=True`) no pre-existing test in this
codebase touches.
"""
import numpy as np
import pandas as pd

from scaata.agents.orchestrator import run_inner_loop
from scaata.agents.nodes.critique_node import SENTIMENT_EXPERT_SOURCE
from scaata.config import FEATURE_COLUMNS


def _make_synthetic_train_df(n=300, seed=0, with_sentiment=False):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}, index=dates)
    df["Close"] = close
    df["returns"] = pd.Series(close, index=dates).pct_change().fillna(0)
    df["Volume"] = rng.integers(1_000_000, 5_000_000, n)
    df["volume_ma_30"] = df["Volume"].rolling(30, min_periods=1).mean()
    df["Ticker"] = "TEST"
    if with_sentiment:
        df["sentiment_score"] = rng.normal(0, 0.5, n)
    return df


def test_sentiment_expert_is_included_when_sentiment_score_present():
    train_df = _make_synthetic_train_df(with_sentiment=True)
    final_state = run_inner_loop(train_df, max_iterations=3)

    assert SENTIMENT_EXPERT_SOURCE in final_state["expert_names"]
    assert len(final_state["strategy_pool_weights"]) == len(final_state["expert_names"])


def test_sentiment_expert_is_absent_without_sentiment_score():
    train_df = _make_synthetic_train_df(with_sentiment=False)
    final_state = run_inner_loop(train_df, max_iterations=3)

    assert SENTIMENT_EXPERT_SOURCE not in final_state["expert_names"]
    assert len(final_state["strategy_pool_weights"]) == len(final_state["expert_names"])


def test_novelty_score_and_regret_are_populated():
    train_df = _make_synthetic_train_df(seed=3)
    final_state = run_inner_loop(train_df, max_iterations=3)

    assert "novelty_score" in final_state
    assert 0.0 <= final_state["novelty_score"] <= 1.0
    assert "cumulative_regret" in final_state
    assert final_state["cumulative_regret"]["n_rounds"] == final_state["iteration"]


def test_all_weights_still_respect_the_floor_with_sentiment_expert_present():
    from scaata.config import MIN_STRATEGY_WEIGHT

    train_df = _make_synthetic_train_df(seed=4, with_sentiment=True)
    final_state = run_inner_loop(train_df, max_iterations=4)

    weights = final_state["strategy_pool_weights"]
    assert all(w >= MIN_STRATEGY_WEIGHT - 1e-9 for w in weights)
    assert sum(weights) == 1.0 or abs(sum(weights) - 1.0) < 1e-9
