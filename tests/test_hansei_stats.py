"""The Lab's statistics: they must catch luck, admit real effects, and stay valid when read every day."""
from __future__ import annotations

import numpy as np
import pytest

from scaata.research import stats


def test_deflated_sharpe_discounts_the_best_of_many_noise_strategies():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.01, (2000, 50))                  # 50 strategies with no skill at all
    srs = [stats.sharpe_per_period(returns[:, k]) for k in range(50)]
    best = int(np.argmax(srs))
    result = stats.deflated_sharpe(srs[best], 2000, srs)
    assert result["psr_vs_benchmark"] > 0.95, "taken alone, the lucky winner looks significant"
    assert result["dsr"] < 0.9, "after counting the 50 tries, it is not"
    assert result["luck_threshold_annual"] > 0.5, "luck alone is expected to produce a sizeable Sharpe here"


def test_deflated_sharpe_keeps_a_genuinely_strong_strategy():
    rng = np.random.default_rng(4)
    returns = rng.normal(0.0, 0.01, (2500, 10))
    returns[:, 0] += 0.0012                                       # about 1.9 annual Sharpe, planted
    srs = [stats.sharpe_per_period(returns[:, k]) for k in range(10)]
    assert stats.deflated_sharpe(srs[0], 2500, srs)["dsr"] > 0.95


def test_deflated_sharpe_against_a_benchmark_asks_about_the_difference():
    srs = [0.0500, 0.0501, 0.0499]                                # near-copies of one strategy, about 0.8 a year
    against_zero = stats.deflated_sharpe(0.0500, 3000, srs)
    against_holding = stats.deflated_sharpe(0.0500, 3000, srs, benchmark=0.0450)
    assert against_zero["dsr"] > 0.99, "a Sharpe of 0.8 a year over 12 years is clearly above zero"
    assert against_holding["dsr"] < 0.7, "but barely above a benchmark that is nearly as good"
    assert against_holding["benchmark_annual"] == pytest.approx(0.0450 * np.sqrt(252))


def test_psr_basics():
    assert stats.psr(0.0, 0.0, 1000) == pytest.approx(0.5)
    assert stats.psr(0.1, 0.0, 1000) > 0.99 and stats.psr(-0.1, 0.0, 1000) < 0.01
    assert stats.psr(0.05, 0.0, 1000, skew=-2.0, kurt=9.0) < stats.psr(0.05, 0.0, 1000), \
        "fat tails and negative skew make the same Sharpe less trustworthy"
    assert stats.expected_max_sharpe(1, 1.0) == 0.0
    assert stats.expected_max_sharpe(100, 1.0) > stats.expected_max_sharpe(10, 1.0) > 0


def test_pbo_is_high_for_pure_noise_and_low_when_one_variant_is_really_better():
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 0.01, (800, 12))
    planted = noise.copy()
    planted[:, 3] += 0.002
    assert stats.pbo(noise)["pbo"] > 0.3
    assert stats.pbo(planted)["pbo"] < 0.15
    assert stats.pbo(noise[:, :1])["pbo"] is None


def test_e_process_finds_a_real_difference_and_stays_quiet_without_one():
    rng = np.random.default_rng(6)
    real = rng.normal(0.003, 0.004, 400)                          # a steady daily edge
    result = stats.e_process(real)
    assert result["up_final"] > stats.THRESHOLD and result["down_final"] <= 1.0
    assert int(np.argmax(result["up"] >= stats.THRESHOLD)) < 400
    nothing = [stats.e_process(rng.normal(0.0, 0.004, 300)) for _ in range(300)]
    crossed = np.mean([(r["up"] >= stats.THRESHOLD).any() or (r["down"] >= stats.THRESHOLD).any() for r in nothing])
    assert crossed <= 0.05, "reading it every day must not raise false alarms above the stated 5% per side"


def test_e_process_is_valid_when_you_stop_at_the_best_moment():
    """Optional stopping: pick the most flattering day to read it. The guarantee is on the whole path."""
    rng = np.random.default_rng(7)
    peaks = [stats.e_process(rng.normal(0, 0.004, 250))["up"].max() for _ in range(400)]
    assert np.mean(np.array(peaks) >= stats.THRESHOLD) <= 0.05


def test_the_evidence_budget_shrinks_with_failures_and_refills_on_a_discovery():
    wealth = stats.INITIAL_WEALTH
    levels = []
    for _ in range(6):
        alpha, wealth, discovery = stats.spend(wealth, 0.5)
        levels.append(alpha)
        assert not discovery
    assert levels == sorted(levels, reverse=True) and levels[-1] < levels[0] / 2
    alpha, after, discovery = stats.spend(wealth, alpha / 10)
    assert discovery and after == pytest.approx(wealth + stats.PAYOUT)
    assert stats.spend(wealth, None) == (stats.offered_level(wealth), wealth, False), "no p-value, nothing spent"
    rows = stats.ledger([("a", 0.5), ("b", None), ("c", 0.5)])
    assert rows[1]["wealth_after"] == rows[1]["wealth_before"] and rows[2]["level"] < rows[0]["level"]


def _book(returns_hansei, returns_hold, cash_share):
    """A paper book from daily returns and a constant share of cash."""
    equity, hold, history = 10000.0, 10000.0, []
    history.append({"date": "2026-01-01", "equity": equity, "benchmark_equity": hold, "cash": equity * cash_share})
    for i, (a, b) in enumerate(zip(returns_hansei, returns_hold), start=2):
        equity, hold = equity * (1 + a), hold * (1 + b)
        history.append({"date": f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", "equity": equity, "benchmark_equity": hold,
                        "cash": equity * cash_share})
    return {"equity_history": history}


def test_selection_meter_gives_no_credit_for_simply_being_less_invested():
    from scaata.research import audit

    rng = np.random.default_rng(8)
    market = rng.normal(0.0004, 0.01, 300)
    zero_rate = lambda year: 0.0   # noqa: E731
    half_in = audit.selection_meter(_book(0.5 * market, market, 0.5), zero_rate)      # exactly the matched holder
    assert half_in["total_selection"] == pytest.approx(0.0, abs=1e-6)
    assert half_in["e_selection"]["up_final"] < 3 and half_in["e_selection"]["down_final"] < 3
    assert audit.selection_meter({"equity_history": []}) is None


def test_selection_meter_sees_a_real_edge_in_stock_choice_and_in_losses_avoided():
    from scaata.research import audit

    rng = np.random.default_rng(9)
    market = rng.normal(0.0004, 0.01, 500)
    zero_rate = lambda year: 0.0   # noqa: E731
    edge = audit.selection_meter(_book(0.5 * market + 0.0015, market, 0.5), zero_rate)    # +0.15% a day on top
    assert edge["e_selection"]["up_final"] > stats.THRESHOLD
    dodged = np.where(market < -0.01, 0.0, 0.5 * market)                                  # sits out the worst days
    safe = audit.selection_meter(_book(dodged, market, 0.5), zero_rate)
    assert safe["e_downside"]["up_final"] > 2
