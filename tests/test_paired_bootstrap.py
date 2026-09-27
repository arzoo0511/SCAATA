"""Tests for the paired block bootstrap used by the retrain gate."""
import numpy as np

from scaata.evaluation.stats import paired_block_bootstrap_prob_better


def _curve(drift, n=252, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return 100 * np.cumprod(np.r_[1.0, 1 + rng.normal(drift, vol, n - 1)])


def test_clearly_better_strategy_scores_near_one():
    good, bad = _curve(0.003, seed=1), _curve(-0.003, seed=2)
    assert paired_block_bootstrap_prob_better(good, bad) > 0.95
    assert paired_block_bootstrap_prob_better(bad, good) < 0.05


def test_identical_strategies_are_never_strictly_better():
    curve = _curve(0.001)
    assert paired_block_bootstrap_prob_better(curve, curve) == 0.0


def test_flat_curve_counts_as_cash_not_nan():
    flat = np.full(252, 100.0)
    losing = _curve(-0.003, seed=3)
    assert paired_block_bootstrap_prob_better(flat, losing) > 0.95
    assert paired_block_bootstrap_prob_better(losing, flat) < 0.05


def test_pairing_cancels_shared_market_moves():
    """Two strategies with the same huge market exposure plus a small,
    consistent edge: pairing should see the edge through the market noise."""
    rng = np.random.default_rng(4)
    market = rng.normal(0.0, 0.03, 251)
    a = 100 * np.cumprod(np.r_[1.0, 1 + market + 0.002])
    b = 100 * np.cumprod(np.r_[1.0, 1 + market])
    assert paired_block_bootstrap_prob_better(a, b) > 0.9


def test_too_short_returns_nan():
    assert np.isnan(paired_block_bootstrap_prob_better([100.0, 101.0], [100.0, 99.0]))
