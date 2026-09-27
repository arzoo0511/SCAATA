"""Tests for portfolio-level evaluation (Phase 15): basic combination
arithmetic must be correct, error handling must catch mismatched-length
and misconfigured-allocation inputs, and -- the actual falsifiable claim
this module exists for -- uncorrelated crashes at different times must
show a real diversification benefit while a fully synchronized crash
across all holdings must show none, demonstrating the mechanism actually
reveals the "correlated multi-asset drawdown" blind spot rather than just
asserting that it does.
"""
import numpy as np
import pandas as pd
import pytest

from scaata.evaluation.portfolio import (
    combine_into_portfolio,
    normalize_equity_curve,
    slice_equity_curves_to_common_length,
)


def test_normalize_equity_curve_preserves_return_path_at_given_allocation():
    equity = np.array([100.0, 110.0, 99.0])
    scaled = normalize_equity_curve(equity, allocation_fraction=0.5, initial_cash=1000.0)
    # starts at 0.5 * 1000 = 500, same percentage path as the original
    np.testing.assert_allclose(scaled, [500.0, 550.0, 495.0])


def test_slice_equity_curves_to_common_length_trims_to_shortest():
    dfs = {
        "A": pd.DataFrame({"x": range(10)}),
        "B": pd.DataFrame({"x": range(8)}),
        "C": pd.DataFrame({"x": range(12)}),
    }
    result = slice_equity_curves_to_common_length(dfs)
    assert all(len(df) == 8 for df in result.values())


def test_combine_into_portfolio_equal_weight_sums_scaled_curves():
    equity_a = np.array([100.0, 110.0])
    equity_b = np.array([100.0, 90.0])
    result = combine_into_portfolio({"A": equity_a, "B": equity_b}, initial_cash=1000.0)

    # equal weight: each gets 500 initial; A ends at 550, B ends at 450 -> 1000
    np.testing.assert_allclose(result["portfolio_equity_curve"], [1000.0, 1000.0])


def test_combine_into_portfolio_raises_on_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        combine_into_portfolio({"A": np.array([100.0, 110.0]), "B": np.array([100.0, 110.0, 120.0])})


def test_combine_into_portfolio_raises_on_bad_allocations():
    with pytest.raises(ValueError, match="sum to 1.0"):
        combine_into_portfolio(
            {"A": np.array([100.0, 110.0]), "B": np.array([100.0, 90.0])},
            allocations={"A": 0.3, "B": 0.3},
        )


def test_combine_into_portfolio_raises_on_empty_input():
    with pytest.raises(ValueError, match="at least one ticker"):
        combine_into_portfolio({})


def _step_down_and_recover(n, dip_start, dip_end, dip_level=0.85):
    """Flat at 100, dips to `dip_level*100` for [dip_start, dip_end), then
    recovers back to 100 for the rest of the series. Crucially recovers
    (unlike a permanent step-down) so that whether two such dips overlap
    in time actually changes the combined worst point -- a permanent,
    never-recovered drop would reach the same final trough regardless of
    timing, which would make a "different day" vs. "same day" comparison
    meaningless.
    """
    equity = np.full(n, 100.0)
    equity[dip_start:dip_end] = 100.0 * dip_level
    return equity


def test_uncorrelated_crashes_on_different_days_show_diversification_benefit():
    n = 20
    equity_a = _step_down_and_recover(n, dip_start=5, dip_end=8)
    equity_b = _step_down_and_recover(n, dip_start=15, dip_end=18)

    result = combine_into_portfolio({"A": equity_a, "B": equity_b})

    assert result["diversification_effect"] > 0, (
        "uncorrelated crashes at different times should show a real diversification benefit"
    )


def test_correlated_simultaneous_crash_shows_no_diversification_benefit():
    n = 20
    equity_a = _step_down_and_recover(n, dip_start=10, dip_end=13)
    equity_b = _step_down_and_recover(n, dip_start=10, dip_end=13)  # same window as A

    result = combine_into_portfolio({"A": equity_a, "B": equity_b})

    assert result["diversification_effect"] == pytest.approx(0.0, abs=1e-9), (
        "a fully synchronized, equal-magnitude crash across all holdings should show zero "
        "diversification benefit -- the portfolio suffers exactly what the naive average predicts"
    )


def test_diversification_effect_is_larger_when_crashes_are_uncorrelated():
    """The direct, falsifiable comparison behind "closes the correlated
    multi-asset drawdown blind spot": the benefit measured in the
    uncorrelated scenario must clearly exceed the (near-zero) benefit in
    the correlated scenario, not just be asserted to differ.
    """
    n = 20
    uncorrelated_result = combine_into_portfolio({
        "A": _step_down_and_recover(n, 5, 8),
        "B": _step_down_and_recover(n, 15, 18),
    })
    correlated_result = combine_into_portfolio({
        "A": _step_down_and_recover(n, 10, 13),
        "B": _step_down_and_recover(n, 10, 13),
    })

    assert uncorrelated_result["diversification_effect"] > correlated_result["diversification_effect"]


def test_naive_average_reported_alongside_portfolio_metric():
    equity_a = np.array([100.0, 90.0, 100.0])
    equity_b = np.array([100.0, 100.0, 80.0])
    result = combine_into_portfolio({"A": equity_a, "B": equity_b})

    assert set(result["per_ticker_max_drawdown"].keys()) == {"A", "B"}
    assert result["naive_average_max_drawdown"] == pytest.approx(
        np.mean(list(result["per_ticker_max_drawdown"].values()))
    )
