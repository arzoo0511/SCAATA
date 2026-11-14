"""The HANSEI dashboard: it must draw from the agent's saved state, survive
missing or empty state, and never write to it."""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest

pytest.importorskip("plotly")
pytest.importorskip("streamlit")

from scaata.config import ROOT_DIR
from scaata.dashboard.ui import charts, data, mind

STATE_FILES = [ROOT_DIR / "paper_book_india.json", ROOT_DIR / "forward_test" / "hansei_journal.json",
               ROOT_DIR / "forward_test" / "hansei_memory.json", ROOT_DIR / "forward_test" / "hansei_evaluation.json"]
PAGES = ["now_page", "mind_page", "learning_page", "news_page", "evidence_page", "lab_page", "macro_page",
         "system_page"]


def _book():
    return {"initial_cash": 10000.0, "cash": 4000.0, "pending": [], "positions": {},
            "equity_history": [{"date": "2026-01-01", "equity": 10000.0, "benchmark_equity": 10000.0, "cash": 10000.0},
                               {"date": "2026-01-02", "equity": 10100.0, "benchmark_equity": 10300.0, "cash": 4000.0},
                               {"date": "2026-01-05", "equity": 9999.0, "benchmark_equity": 9785.0, "cash": 4000.0}],
            "trades": [{"date": "2026-01-02", "symbol": "ITC.NS", "action": "BUY", "qty": 3, "price": 300.0,
                        "cost": 1.2, "reason": "Buying ITC"}]}


def _journal():
    def entry(day, action, held, news_view):
        return {"session": day, "equity": 10000.0, "benchmark_equity": 10100.0, "filled": [], "lessons_learned": 0,
                "weights": {"trend": 0.3, "volatility": 0.2, "momentum": 0.25, "news": 0.25},
                "decisions": [{"symbol": "ITC.NS", "score": -0.1, "target_exposure": 0.8, "current_exposure": held,
                               "action": action, "reasons": ["Holding </script> at 80%", "trend against"],
                               "views": {"trend": -0.5, "volatility": 0.0, "momentum": 0.2, "news": news_view},
                               "value": 1500.0}],
                "news": {"ITC.NS": {"signal": {"view": news_view, "material_count": 1},
                                    "top": [{"title": "ITC wins order", "source": "Wire", "sentiment": 0.8,
                                             "material": 0.9, "count": 2,
                                             "published_utc": "2026-01-02T05:00:00+00:00"}]}}}
    return [entry("2026-01-02", "BUY", 0.0, 0.4), entry("2026-01-05", "HOLD", 0.8, 0.0)]


def test_equity_frame_gap_and_drawdown():
    frame = data.equity_frame(_book())
    assert frame["gap"].round(4).tolist() == [0.0, -0.02, 0.0214]
    assert frame["hold_dd"].iloc[-1] == pytest.approx(9785 / 10300 - 1)
    assert frame["hansei_dd"].iloc[1] == 0.0
    assert frame["invested"].iloc[-1] == pytest.approx(1 - 4000 / 9999)


def test_frames_from_the_journal():
    journal = _journal()
    decisions = data.decisions_frame(journal)
    assert decisions[["action", "held", "news"]].values.tolist() == [["BUY", 0.0, 0.4], ["HOLD", 0.8, 0.0]]
    assert data.trust_frame(journal)["trend"].tolist() == [0.3, 0.3]
    news = data.news_frame(journal)
    assert len(news) == 1, "a story seen on two days is counted once"
    assert news["session"].iloc[0] == pd.Timestamp("2026-01-02")
    assert data.allocation(_book(), journal) == [("ITC", 1500.0), ("Liquid fund", 4000.0)]


def test_empty_state_gives_empty_frames_not_errors():
    assert data.equity_frame(None).empty and data.trades_frame(None).empty
    assert data.decisions_frame([]).empty and data.trust_frame([]).empty and data.news_frame([]).empty
    assert data.lessons_frame({}).empty and data.beliefs_frame(None).empty
    assert data.allocation(None, []) == [] and data.gap_story(None) is None and data.mind_payload([]) == []
    assert data.backtest_curve(None).empty and data.blocks_frame(None).empty and data.signal_frame(None).empty
    assert data.read_json(ROOT_DIR / "no_such_file.json") is None
    for fig in (charts.race(data.equity_frame(None), data.trades_frame(None)), charts.exposure_map(data.decisions_frame([])),
                charts.trust_lines(data.trust_frame([]), data.lessons_frame({})), charts.lessons_strip(data.lessons_frame({})),
                charts.beliefs_map(data.beliefs_frame(None)), charts.news_bubbles(data.news_frame([])),
                charts.backtest(data.backtest_curve(None)), charts.drawdown_dumbbell(data.blocks_frame(None)),
                charts.return_bars(data.blocks_frame(None)), charts.knob_grid([]), charts.forest(data.signal_frame(None)),
                charts.hit_rates({})):
        assert fig.layout.height


@pytest.mark.parametrize("mode", ["value", "gap", "drawdown"])
def test_race_draws_both_legs(mode):
    fig = charts.race(data.equity_frame(_book()), data.trades_frame(_book()), mode)
    assert len(fig.data) >= 2
    if mode == "value":
        assert {"HANSEI", "Just holding", "Buy"} <= {t.name for t in fig.data}


def test_blocks_exclude_the_tune_and_holdout_aggregates():
    period = {"cagr_diff": 0.01, "sharpe_diff": 0.1, "hansei": {"max_drawdown": -0.2, "cagr": 0.1},
              "buy_and_hold": {"max_drawdown": -0.3, "cagr": 0.09}}
    blocks = data.blocks_frame({"periods": {"Tune 2007-2019": period, "2007-2009": period, "2023-": period}})
    assert blocks["block"].tolist() == ["2007-2009", "2023-"]


def test_now_cockpit_payload_matches_the_book():
    from scaata.dashboard.ui import now

    payload = data.now_payload(_book(), _journal(), None)
    assert payload["summary"]["equity"] == 9999.0 and payload["summary"]["hold"] == 9785.0
    assert payload["summary"]["gap"] == pytest.approx(0.0214)
    assert payload["series"]["dates"] == ["2026-01-01", "2026-01-02", "2026-01-05"]
    assert payload["stocks"][0]["history"] == [0.0, 0.8] and payload["bridge"] is None
    assert payload["trades"][0]["name"] == "ITC"
    html = now.cockpit_html(payload)
    assert html.count("</script>") == 1 and "__DATA__" not in html


def test_macroverse_stage_embeds_runs_and_keeps_real_and_synthetic_apart():
    from scaata.dashboard.ui import macro

    runs = [{"id": "r", "kind": "replay", "title": "A </script> crisis"}, {"id": "s", "kind": "scenario", "title": "x"}]
    html = macro.stage_html(runs, None)
    assert html.count("</script>") == 1 and "__RUNS__" not in html and "__SENS__" not in html
    assert "REAL HISTORY" in html and "SYNTHETIC" in html and "NOT A FORECAST" in html


def test_explorer_embeds_the_journal_as_it_was_and_cannot_break_out_of_its_script():
    payload = data.mind_payload(_journal())
    assert payload[0]["stocks"][0]["views"]["news"] == 0.4 and payload[1]["stocks"][0]["held"] == 0.8
    html = mind.explorer_html(payload)
    assert html.count("</script>") == 1, "journal text must not be able to close the script tag"
    assert "__DATA__" not in html and "__UP__" not in html


def _render(page: str) -> None:
    """Runs inside Streamlit's test harness: draw one page, with live prices stubbed out."""
    import pandas as pd

    from scaata.dashboard.ui import pages, theme

    def prices(symbols):
        index = pd.bdate_range("2026-01-01", periods=30)
        return pd.DataFrame({s: range(100, 130) for s in symbols}, index=index, dtype=float)

    pages.load_prices = prices
    theme.inject()
    getattr(pages, page)()


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_on_the_real_state_without_writing_to_it(page, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.delenv("KITE_CONNECT_API_KEY", raising=False)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in STATE_FILES if p.exists()}
    app = AppTest.from_function(_render, args=(page,), default_timeout=60).run()
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.markdown) > 2
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in STATE_FILES if p.exists()}
    assert before == after, "the dashboard must only read the agent's state"


def test_pages_render_before_the_agent_has_ever_run(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    import scaata.agent.daily as daily
    import scaata.agent.memory as memory
    import scaata.live.paper_book as paper_book

    monkeypatch.setattr(paper_book, "BOOK_PATH", tmp_path / "book.json")
    monkeypatch.setattr(daily, "JOURNAL_PATH", tmp_path / "journal.json")
    monkeypatch.setattr(memory, "MEMORY_PATH", tmp_path / "memory.json")
    monkeypatch.setattr(data, "EVALUATION_PATH", tmp_path / "evaluation.json")
    monkeypatch.setattr(data, "RESULTS", tmp_path)
    import scaata.research.director as director
    import scaata.research.registry as registry

    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setattr(director, "AGENDA_PATH", tmp_path / "agenda.json")
    import scaata.dashboard.ui.pages as pages

    monkeypatch.setattr(pages, "MACRO_RUNS", tmp_path / "macroverse_runs")
    for page in PAGES:
        app = AppTest.from_function(_render, args=(page,), default_timeout=60).run()
        assert not app.exception, (page, [e.value for e in app.exception])
