"""Integration test for the LangGraph inner loop: confirms the graph
compiles, actually cycles through the Critique -> Meta-Selector feedback
edge, and terminates within the configured bound — using the mock
strategy pool (no live GitHub/Groq calls) and a fully synthetic price
series (no network access), so this runs anywhere.
"""
import numpy as np
import pandas as pd

from scaata.agents.orchestrator import run_inner_loop
from scaata.config import FEATURE_COLUMNS


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


def test_graph_compiles_cycles_and_terminates():
    train_df = _make_synthetic_train_df()

    final_state = run_inner_loop(train_df, max_iterations=4)

    assert final_state["converged"] is True
    assert 1 <= final_state["iteration"] <= 4
    assert final_state["bc_model"] is not None
    assert final_state["meta_model"] is not None
    assert final_state["strategy_weight_matrix"].shape[0] == len(train_df)
    # The critique loop must have actually run at least once (real cycle,
    # not just a single pass-through).
    assert len(final_state["history"]) == final_state["iteration"]


def test_graph_respects_max_iterations_bound():
    train_df = _make_synthetic_train_df(seed=1)
    final_state = run_inner_loop(train_df, max_iterations=2)
    assert final_state["iteration"] <= 2


def test_strategy_pool_weights_stay_within_valid_bounds():
    from scaata.config import MIN_STRATEGY_WEIGHT

    train_df = _make_synthetic_train_df(seed=2)
    final_state = run_inner_loop(train_df, max_iterations=4)

    weights = final_state["strategy_pool_weights"]
    assert all(w >= MIN_STRATEGY_WEIGHT - 1e-9 for w in weights)
    assert all(w <= 1.0 for w in weights)


def test_run_inner_loop_does_not_mutate_callers_train_df():
    """Regression test: a real crash was traced to `meta_selector_node`
    running strategy-pool code against `state["train_df"]` by reference
    (scaata.strategies.pool.run_strategy_safely). At least one candidate
    strategy assigned its own `df['rsi'] = ...`, colliding with the
    project's real `rsi` feature column and silently overwriting it with
    NaN-during-warmup values -- corrupting the caller's own DataFrame after
    `run_inner_loop` returned, and crashing a subsequent `train_ppo` call
    with NaN logits on its very first forward pass. `run_inner_loop` must
    never let the caller's DataFrame come back changed, in columns or
    values.
    """
    train_df = _make_synthetic_train_df(seed=3)
    columns_before = list(train_df.columns)
    snapshot_before = train_df.copy(deep=True)

    run_inner_loop(train_df, max_iterations=2)

    assert list(train_df.columns) == columns_before
    pd.testing.assert_frame_equal(train_df, snapshot_before)
