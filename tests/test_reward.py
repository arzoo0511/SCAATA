"""Correctness tests for the self-critique reward terms — the exact area
the v1 notebook silently never implemented, so these are regression tests
guarding against that gap reopening.
"""
import numpy as np

from scaata.rl.reward import (
    SelfCritiqueTracker,
    drawdown_penalty,
    holding_time_penalty,
    volatility_chasing_penalty,
)


def test_drawdown_penalty_zero_when_at_peak():
    history = np.array([100, 105, 110, 115])
    assert drawdown_penalty(history, lam=1.0) == 0.0


def test_drawdown_penalty_negative_beyond_threshold():
    history = np.array([100, 110, 90])  # ~18% drawdown from peak of 110
    penalty = drawdown_penalty(history, lam=1.0, threshold=0.05)
    assert penalty < 0


def test_drawdown_penalty_zero_within_threshold():
    history = np.array([100, 102, 100])  # ~2% drawdown, below a 5% threshold
    assert drawdown_penalty(history, lam=1.0, threshold=0.05) == 0.0


def test_volatility_chasing_penalty_zero_when_not_new_entry():
    returns = np.concatenate([np.full(30, 0.001), np.full(5, 0.05)])
    assert volatility_chasing_penalty(returns, is_new_entry=False, lam=1.0) == 0.0


def test_volatility_chasing_penalty_negative_on_elevated_vol_entry():
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.001, 30)
    spike = rng.normal(0, 0.05, 5)
    returns = np.concatenate([calm, spike])
    penalty = volatility_chasing_penalty(returns, is_new_entry=True, lam=1.0, window=5, baseline_window=30)
    assert penalty < 0


def test_volatility_chasing_penalty_near_zero_when_vol_flat():
    returns = np.full(40, 0.001)
    penalty = volatility_chasing_penalty(returns, is_new_entry=True, lam=1.0, window=5, baseline_window=30)
    assert penalty == 0.0


def test_holding_time_penalty_zero_when_profitable():
    assert holding_time_penalty(unrealized_pnl_pct=0.05, holding_days=20, lam=1.0) == 0.0


def test_holding_time_penalty_zero_within_grace_period():
    assert holding_time_penalty(unrealized_pnl_pct=-0.05, holding_days=3, lam=1.0, grace_days=5) == 0.0


def test_holding_time_penalty_grows_with_days_and_loss_depth():
    p_short = holding_time_penalty(unrealized_pnl_pct=-0.05, holding_days=10, lam=1.0, grace_days=5)
    p_long = holding_time_penalty(unrealized_pnl_pct=-0.05, holding_days=20, lam=1.0, grace_days=5)
    p_deeper_loss = holding_time_penalty(unrealized_pnl_pct=-0.15, holding_days=10, lam=1.0, grace_days=5)

    assert p_short < 0
    assert p_long < p_short  # more negative: held longer
    assert p_deeper_loss < p_short  # more negative: deeper loss


def test_self_critique_tracker_total_matches_component_sum():
    tracker = SelfCritiqueTracker(lambda_dd=1.0, lambda_vol=1.0, lambda_hold=1.0)
    tracker.reset(initial_value=10_000)
    tracker.on_entry(step_idx=0)

    values = [10_000, 9_800, 9_500, 9_000, 8_800]
    breakdown = None
    for i, v in enumerate(values[1:], start=1):
        breakdown = tracker.step_penalty(
            current_value=v, step_idx=i, position=1, entry_price=100, current_price=100 * (v / 10_000),
            is_new_entry=False,
        )

    assert breakdown is not None
    component_sum = (
        breakdown["drawdown_penalty"] + breakdown["volatility_penalty"] + breakdown["holding_time_penalty"]
    )
    assert breakdown["total"] == component_sum
    # A sustained losing drawdown should produce a strictly negative total.
    assert breakdown["total"] < 0


def test_self_critique_tracker_reset_clears_history():
    tracker = SelfCritiqueTracker()
    tracker.reset(initial_value=10_000)
    tracker.on_entry(step_idx=0)
    tracker.step_penalty(9_000, step_idx=1, position=1, entry_price=100, current_price=90, is_new_entry=False)

    tracker.reset(initial_value=5_000)
    assert tracker.portfolio_history == [5_000]
    assert tracker.entry_step is None
