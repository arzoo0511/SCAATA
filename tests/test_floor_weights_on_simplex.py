"""Regression test for the simplex-floor projection bug caught while
wiring Hedge into `critique_node` (Phase 10d): a naive "floor then divide
by the new sum" renormalization can silently push a component back below
the floor it was meant to enforce. `floor_weights_on_simplex` must not.
"""
import numpy as np
import pytest

from scaata.strategies.hedge import floor_weights_on_simplex


def test_naive_approach_would_have_violated_the_floor():
    # Sanity-check the bug this function fixes: floor+divide alone breaks.
    weights = np.array([0.98, 0.01, 0.01])
    floor = 0.05
    naive = np.maximum(weights, floor)
    naive = naive / naive.sum()
    assert naive.min() < floor  # demonstrates the bug exists in the naive approach


def test_every_component_meets_the_floor():
    weights = np.array([0.98, 0.01, 0.01])
    floor = 0.05
    result = floor_weights_on_simplex(weights, floor)
    assert (result >= floor - 1e-12).all()
    assert result.sum() == pytest.approx(1.0)


def test_preserves_ranking_of_components_above_the_floor():
    weights = np.array([0.6, 0.3, 0.1])
    floor = 0.05
    result = floor_weights_on_simplex(weights, floor)
    assert result[0] > result[1] > result[2]


def test_already_valid_distribution_is_unchanged():
    weights = np.array([0.5, 0.3, 0.2])
    floor = 0.05
    result = floor_weights_on_simplex(weights, floor)
    np.testing.assert_allclose(result, weights)


def test_all_weights_at_or_below_floor_distributes_uniformly_above_floor():
    weights = np.array([0.05, 0.05, 0.05, 0.05])  # already sums to 0.2, not 1 -- edge case: raw doesn't sum to 1
    # Function assumes inputs conceptually on the simplex, but must still
    # behave sanely (no crash, floor respected) even if given something odd.
    result = floor_weights_on_simplex(weights, 0.05)
    assert (result >= 0.05 - 1e-12).all()
    assert result.sum() == pytest.approx(1.0)


def test_floor_times_n_at_capacity_returns_uniform():
    weights = np.array([0.9, 0.05, 0.05])
    floor = 1.0 / 3  # exactly at capacity: n * floor == 1
    result = floor_weights_on_simplex(weights, floor)
    np.testing.assert_allclose(result, np.full(3, 1 / 3))
