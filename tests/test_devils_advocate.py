"""Tests for the devil's-advocate adversarial critique (Phase 12):
- `_verdict_from_delta` must return the right quadrant for each sign
  combination of risk/return delta.
- `devils_advocate_report` must correctly identify the runner-up and
  produce a sensible diff on a simple hand-built case.
- The falsifiable, known-ground-truth check: reusing
  `tests/test_robustness.py`'s poisoning construction, when the chosen
  strategy is a known-bad reversed signal, the delta must favor the good
  runner-up and the verdict must flag the chosen path as inferior.
"""
import numpy as np
import pandas as pd

from scaata.agents.nodes.devils_advocate_node import (
    _verdict_from_delta,
    devils_advocate_report,
)


def test_verdict_dominates_when_chosen_better_on_both():
    delta = {"total_penalty": 0.1, "window_return": 0.05}
    assert "dominates" in _verdict_from_delta(delta)


def test_verdict_flags_chosen_as_weaker_when_worse_on_both():
    delta = {"total_penalty": -0.1, "window_return": -0.05}
    assert "weaker pick" in _verdict_from_delta(delta)


def test_verdict_tradeoff_lower_risk_lower_return():
    delta = {"total_penalty": 0.1, "window_return": -0.05}
    assert "avoided deeper risk penalties" in _verdict_from_delta(delta)


def test_verdict_tradeoff_higher_return_higher_risk():
    delta = {"total_penalty": -0.1, "window_return": 0.05}
    assert "higher risk penalties" in _verdict_from_delta(delta)


def test_verdict_negligible_when_both_near_zero():
    delta = {"total_penalty": 1e-9, "window_return": -1e-9}
    assert "negligible" in _verdict_from_delta(delta)


def _make_synthetic_train_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": close}, index=dates)


def test_returns_none_when_fewer_than_two_strategies():
    train_df = _make_synthetic_train_df()
    strategy_signals = [{"source": "only_one", "signals": np.ones(len(train_df), dtype=int)}]
    weight_matrix = np.ones((len(train_df), 1))

    result = devils_advocate_report(train_df, strategy_signals, weight_matrix, window=20)
    assert result is None


def test_identifies_correct_chosen_and_runner_up_indices():
    train_df = _make_synthetic_train_df()
    n = len(train_df)
    strategy_signals = [
        {"source": "a", "signals": np.ones(n, dtype=int)},
        {"source": "b", "signals": np.zeros(n, dtype=int)},
        {"source": "c", "signals": -np.ones(n, dtype=int)},
    ]
    # strategy 0 always highest weight, strategy 1 always second
    weight_matrix = np.tile([0.7, 0.2, 0.1], (n, 1))

    result = devils_advocate_report(train_df, strategy_signals, weight_matrix, window=20)

    assert result is not None
    assert result["chosen_strategy_idx"] == 0
    assert result["runner_up_strategy_idx"] == 1


def test_poisoned_chosen_strategy_is_correctly_flagged_as_inferior_to_good_runner_up():
    """The chosen (top-weighted) strategy is a deliberately sign-reversed
    version of a "good" strategy built with perfect one-step foresight
    (`good_signal[t] = sign(returns[t+1])`) over a deterministic
    alternating up/down price series -- so "good" reliably buys right
    before every up-day and exits right before every down-day (monotonic
    gains, no drawdown), while its negation reliably buys right before
    every down-day (monotonic losses). This is a stronger, fully
    deterministic ground truth than negating a real mock strategy over a
    noisy random walk (whose outperformance on any single 60-day window
    isn't guaranteed) -- the devil's-advocate delta must still correctly
    identify the runner-up ("good") as the better path.

    Note: an "always sell, never actually bought" reversed signal (the
    naive way to build a poisoned entry) would be a no-op here, not
    "wrong" -- it never opens a position to lose money on. The alternating
    construction below avoids that degenerate case by ensuring the
    poisoned path actually trades, and loses, throughout.
    """
    n = 150
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = np.where(np.arange(n) % 2 == 0, 0.02, -0.02)  # even days +2%, odd days -2%
    close = 100 * np.cumprod(1 + returns)
    train_df = pd.DataFrame({"Close": close}, index=dates)

    future_returns = np.roll(returns, -1)
    future_returns[-1] = 0.0
    good_signal = np.sign(future_returns).astype(int)  # perfect one-step foresight
    poisoned_signal = -good_signal

    strategy_signals = [
        {"source": "poisoned", "signals": poisoned_signal},
        {"source": "good", "signals": good_signal},
    ]
    # The poisoned strategy is (wrongly) the top pick; "good" is the runner-up.
    weight_matrix = np.tile([0.9, 0.1], (n, 1))

    result = devils_advocate_report(train_df, strategy_signals, weight_matrix, window=60)

    assert result["chosen_strategy_idx"] == 0  # poisoned
    assert result["runner_up_strategy_idx"] == 1  # good
    assert result["delta"]["window_return"] < 0, "the poisoned (chosen) path should show a worse window return"

    chosen_score = result["chosen_report"]["window_return"] + result["chosen_report"]["total_penalty"]
    alt_score = result["alt_report"]["window_return"] + result["alt_report"]["total_penalty"]
    assert chosen_score < alt_score, "the poisoned path's combined risk+return score should be clearly worse"
    assert "dominates" not in result["verdict"]  # the chosen path must not be reported as the winner
