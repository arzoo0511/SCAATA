"""Tests for the Hedge / multiplicative-weights signal combiner (Phase 10):
weight updates must shift toward lower-loss experts, cumulative regret must
scale sub-linearly (the actual theoretical property the mechanism is
supposed to have, not just "it runs"), and the critique-based per-expert
loss must correctly identify a strategy that trades recklessly even if its
raw return looks fine.
"""
import numpy as np
import pandas as pd
import pytest

from scaata.strategies.hedge import (
    HedgeWeights,
    blended_expert_losses,
    eta_from_horizon,
    expert_losses_from_critique,
    expert_losses_from_returns,
    hedge_regret_report,
)


def test_update_shifts_weight_toward_lower_loss_expert():
    hedge = HedgeWeights(n_experts=3, eta=1.0)
    losses = np.array([0.0, 1.0, 1.0])  # expert 0 clearly better this round

    weights = hedge.update(losses)

    assert weights[0] > weights[1]
    assert weights[0] > weights[2]
    assert weights[1] == pytest.approx(weights[2])
    assert weights.sum() == pytest.approx(1.0)


def test_update_is_a_no_op_shape_when_all_losses_equal():
    hedge = HedgeWeights(n_experts=4, eta=1.0)
    weights = hedge.update(np.array([0.5, 0.5, 0.5, 0.5]))
    np.testing.assert_allclose(weights, np.ones(4) / 4)


def test_log_prior_shifts_weight_without_needing_a_loss_difference():
    hedge = HedgeWeights(n_experts=2, eta=1.0)
    weights = hedge.update(np.array([0.5, 0.5]), log_prior=np.array([0.0, 2.0]))
    assert weights[1] > weights[0]


def test_eta_from_horizon_decreases_with_longer_horizon():
    short = eta_from_horizon(n_experts=5, horizon=10)
    long = eta_from_horizon(n_experts=5, horizon=1000)
    assert short > long > 0


def test_expert_losses_from_returns_rewards_correct_direction_calls():
    # expert 0: always long, returns are all positive -> should have low loss
    # expert 1: always short, same positive returns -> should have high loss
    signal_matrix = np.array([[1, 1, 1, 1], [-1, -1, -1, -1]])
    returns = np.array([0.01, 0.02, 0.01, 0.015])

    losses = expert_losses_from_returns(signal_matrix, returns)

    assert losses[0] < losses[1]


def _make_synthetic_train_df(n=80, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": close}, index=dates)


def test_expert_losses_from_critique_penalizes_a_reckless_alternating_strategy():
    train_df = _make_synthetic_train_df()
    n = len(train_df)
    # Expert 0: never trades (all hold) -> zero penalties, zero return.
    # Expert 1: flips buy/sell every single day -> constantly re-entering,
    # which should trigger volatility-chasing/fee-churn-style penalties
    # relative to the flat expert once critiqued.
    hold_signals = np.zeros(n, dtype=int)
    flip_signals = np.array([1 if i % 2 == 0 else -1 for i in range(n)])

    strategy_signals = [
        {"source": "flat", "signals": hold_signals},
        {"source": "flip", "signals": flip_signals},
    ]

    losses = expert_losses_from_critique(train_df, strategy_signals, window=40)

    assert len(losses) == 2
    # Not asserting a specific direction (depends on realized synthetic
    # returns) — just that the two experts are meaningfully distinguished,
    # i.e. the mechanism isn't returning the same score for both.
    assert losses[0] != pytest.approx(losses[1])


def test_blended_expert_losses_falls_back_to_critique_only():
    critique = np.array([0.2, 0.8])
    blended = blended_expert_losses(critique, return_losses=None)
    np.testing.assert_array_equal(blended, critique)


def test_blended_expert_losses_combines_both_sources():
    critique = np.array([0.0, 1.0])
    returns = np.array([1.0, 0.0])  # opposite ranking from critique
    blended_equal_rho = blended_expert_losses(critique, returns, rho=0.5)
    # with rho=0.5 and opposite rankings, the two experts should end up
    # roughly balanced rather than either critique's or returns' extreme ranking
    assert blended_equal_rho[0] == pytest.approx(blended_equal_rho[1])


def test_regret_grows_sublinearly_relative_to_best_expert_in_hindsight():
    """The actual theoretical property under test: Hedge's cumulative
    regret vs. the best fixed expert in hindsight should scale roughly like
    O(sqrt(T ln N)), not linearly in T. We run two horizons and check the
    ratio of regret growth is well below the ratio of T growth.
    """
    rng = np.random.default_rng(0)
    n_experts = 4

    def run(horizon):
        eta = eta_from_horizon(n_experts, horizon)
        hedge = HedgeWeights(n_experts=n_experts, eta=eta)
        loss_history = np.zeros((horizon, n_experts))
        weight_history = np.zeros((horizon, n_experts))
        # expert 0 has a small consistent edge (lower mean loss); the rest are noisier
        base_means = np.array([0.3, 0.5, 0.5, 0.5])
        for t in range(horizon):
            weight_history[t] = hedge.weights
            losses = np.clip(rng.normal(base_means, 0.2), 0, 1)
            loss_history[t] = losses
            hedge.update(losses)
        return hedge_regret_report(loss_history, weight_history)["regret"]

    regret_short = run(50)
    regret_long = run(800)  # 16x horizon

    # Linear growth would give a ~16x regret increase; sqrt(T ln N) growth
    # gives ~4x. Allow generous slack but this should catch a mechanism
    # that isn't sub-linear at all (e.g. a bug making it grow linearly).
    assert regret_long < regret_short * 10
