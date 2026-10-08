from datetime import date

import pytest

from scaata.agent.evaluate import advisor_scorecard, evaluate, health, performance, significance


def _book(rows):
    return {"initial_cash": 100.0, "last_session": rows[-1][0],
            "equity_history": [{"date": d, "equity": e, "benchmark_equity": b, "cash": c, "holdings": 1}
                               for d, e, b, c in rows]}


def test_all_cash_beats_a_falling_market_entirely_through_exposure():
    book = _book([("2026-01-05", 100.0, 100.0, 100.0), ("2026-01-06", 100.0, 90.0, 100.0)])
    p = performance(book)
    assert p["gap"] == pytest.approx(0.10)
    assert p["gap_from_exposure"] == pytest.approx(0.10 + 0.047 / 365, abs=1e-6)
    assert p["gap_from_selection"] == pytest.approx(-0.047 / 365, abs=1e-6)
    assert p["hold_max_drawdown"] == pytest.approx(-0.10)
    assert p["hansei_max_drawdown"] == 0.0


def test_fully_invested_gap_is_all_selection():
    book = _book([("2026-01-05", 100.0, 100.0, 0.0), ("2026-01-06", 105.0, 101.0, 0.0)])
    p = performance(book)
    assert p["gap_from_exposure"] == pytest.approx(0.0)
    assert p["gap_from_selection"] == pytest.approx(0.04)


def test_scorecard_judges_view_sign_against_the_forward_return():
    obs = [{"date": f"2026-01-{i + 1:02d}", "symbol": "A.NS", "close": 100.0 + i,
            "views": {"trend": 0.5, "volatility": 0.0, "momentum": -0.5, "news": 0.5 if i % 2 else -0.5}}
           for i in range(10)]
    card = advisor_scorecard({"observations": obs}, horizons=(5,))["5_sessions"]
    assert card["trend"]["hit_rate"] == 1.0 and card["trend"]["judged"] == 5
    assert card["momentum"]["hit_rate"] == 0.0
    assert card["volatility"]["judged"] == 0 and card["volatility"]["hit_rate"] is None


def test_health_flags_gaps_and_orders_that_bought_nothing(tmp_path):
    journal = [
        {"session": "2026-01-05", "filled": [], "decisions": [{"symbol": "A.NS", "action": "BUY"}]},
        {"session": "2026-01-07", "filled": [], "decisions": [{"symbol": "A.NS", "action": "HOLD"}]},
    ]
    (tmp_path / "2026-01-07_A.json").write_text('{"headlines": [{"scorer": "keywords"}]}', encoding="utf-8")
    h = health(_book([("2026-01-07", 100.0, 100.0, 100.0)]), journal, tmp_path)
    assert h["weekdays_without_a_session"] == ["2026-01-06"]
    assert h["orders_that_filled_nothing"] == [{"queued_on": "2026-01-05", "symbol": "A.NS"}]
    assert h["news_scored_by_keywords_latest"] is True


def test_significance_grows_with_noise_and_shrinks_with_edge():
    perf = {"sessions": 10, "daily_gap_volatility": 0.002}
    small = significance(perf, {"periods": {"Full period": {"cagr_diff": 0.01}}})
    big = significance(perf, {"periods": {"Full period": {"cagr_diff": 0.10}}})
    assert small["sessions_needed"] > big["sessions_needed"] > 0
    assert "sessions_needed" not in significance(perf, None)


def test_evaluate_runs_end_to_end(tmp_path):
    book = _book([("2026-01-05", 100.0, 100.0, 50.0), ("2026-01-06", 101.0, 100.5, 50.0)])
    memory = {"weights": {"trend": 0.25, "volatility": 0.25, "momentum": 0.25, "news": 0.25}, "observations": []}
    report = evaluate(book, [], memory, None, tmp_path, date(2026, 1, 7))
    assert set(report) >= {"performance", "advisors", "health", "significance", "trust"}


def test_volatility_ratio_compares_daily_swings():
    rows = [("2026-01-05", 100.0, 100.0, 50.0), ("2026-01-06", 101.0, 102.0, 50.0),
            ("2026-01-07", 100.0, 100.0, 50.0), ("2026-01-08", 101.0, 102.0, 50.0)]
    p = performance(_book(rows))
    assert 0 < p["volatility_ratio"] < 1
