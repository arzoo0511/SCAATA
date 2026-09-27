"""End-to-end check that the devil's-advocate trace actually reaches
`history` through the full LangGraph pipeline (not just the node function
in isolation) -- this is the concrete "what is its thinking, what
alternative was considered" record a caller would actually inspect.
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


def test_devils_advocate_report_reaches_history_through_the_full_graph():
    train_df = _make_synthetic_train_df()
    final_state = run_inner_loop(train_df, max_iterations=3)

    assert len(final_state["history"]) >= 1
    for entry in final_state["history"]:
        assert "devils_advocate_report" in entry

    # With the mock 3-strategy pool, there IS a runner-up -- the report
    # should be a real dict, not None, and contain the verdict trace.
    last_report = final_state["history"][-1]["devils_advocate_report"]
    assert last_report is not None
    assert "verdict" in last_report
    assert "delta" in last_report
    assert isinstance(last_report["verdict"], str) and len(last_report["verdict"]) > 0
