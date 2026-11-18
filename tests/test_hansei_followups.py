"""Event labels, Fixed-Share, the live-like replay switches, the event study
and proposals: each does what it says, and none can move a trade by accident."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from scaata.agent import backtest as bt, memory as mem, news
from scaata.agent.events import EVENTS, LABELS, event_type
from scaata.config import INDIA_TICKERS
from scaata.macroverse.replay import stock_values
from scaata.research import director, event_study, proposals


# ------------------------------------------------------------ event labels
@pytest.mark.parametrize("title,expected", [
    ("IOC Q2 profit falls as crude rises", "results"),
    ("Indian Oil cuts ATF price", "crude"),
    ("HDFC Bank Interest Rate Hike: Loan EMIs Set to Increase", "rates"),
    ("RBI approves HDFC Bank re-appointment of V. Srinivasa Rangan", "management"),
    ("RBI imposes Rs 1 crore penalty on ICICI Bank", "regulation"),
    ("ITC to demerge hotels business", "deal"),
    ("Petronet LNG celebrates foundation day", "other"),
    ("", "other"),
])
def test_event_type(title, expected):
    assert event_type(title) == expected


def test_event_words_match_whole_words_only_and_every_event_has_a_label():
    assert event_type("ITC spat with distributor") == "other", "'pat' inside 'spat' is not a results story"
    assert set(EVENTS) | {"other"} == set(LABELS)


def test_the_event_label_cannot_change_the_news_signal():
    now = datetime(2026, 1, 5, tzinfo=timezone.utc)
    plain = [{"title": "ITC Q2 profit beats estimates", "source": "A", "published_utc": "2026-01-04T05:00:00+00:00",
              "relevant": True, "sentiment": 0.8, "material": 0.9},
             {"title": "ITC faces GST notice", "source": "B", "published_utc": "2026-01-03T05:00:00+00:00",
              "relevant": True, "sentiment": -0.6, "material": 0.7}]
    labelled = [{**h, "event": "something-else-entirely"} for h in plain]
    assert news.news_signal(plain, now=now, symbol="ITC.NS") == news.news_signal(labelled, now=now, symbol="ITC.NS")
    stories = news.stories(plain, "ITC.NS")
    assert {s["event"] for s in stories} == {"results", "regulation"}, "stories carry a label for the journal"


# -------------------------------------------------------------- Fixed-Share
def _lesson_run(share: float | None) -> dict:
    """Trend is right for 60 judged days, then momentum is; returns the trust after each update."""
    index = pd.bdate_range("2020-01-01", periods=160)
    closes = pd.Series(100 * np.cumprod(1 + np.where(np.arange(160) % 2 == 0, 0.01, -0.01)), index=index)
    memory, path = mem.new_memory(), []
    for k, t in enumerate(index[:-21]):
        forward = closes.iloc[k + 20] / closes.iloc[k] - 1
        good = 1.0 if forward >= 0 else -1.0
        first = k < 60
        mem.learn(memory, {"X": closes.loc[:t]}, t.date(), **({} if share is None else {"share": share}))
        mem.remember(memory, t.date(), "X", {"trend": good * 0.8 if first else -good * 0.8,
                                             "momentum": -good * 0.8 if first else good * 0.8,
                                             "volatility": 0.0, "news": 0.0}, float(closes.iloc[k]))
        path.append(dict(memory["weights"]))
    return {"path": path, "final": memory["weights"]}


def test_fixed_share_recovers_a_fallen_advisor_sooner_and_keeps_trust_a_distribution():
    plain, shared = _lesson_run(0.0), _lesson_run(0.02)
    cross = lambda run: next(i for i, w in enumerate(run["path"]) if i > 80 and w["momentum"] > w["trend"])  # noqa: E731
    assert cross(shared) < cross(plain), "after the switch, momentum overtakes trend earlier with the share step"
    for w in shared["path"]:
        assert sum(w.values()) == pytest.approx(1.0, abs=2e-4) and min(w.values()) >= mem.MIN_WEIGHT - 1e-4


def test_the_live_learner_uses_the_filed_fixed_share_fraction():
    assert mem.SHARE == pytest.approx(director.FIXED_SHARE), "live setting and the experiment's must be the same number"
    assert _lesson_run(None)["final"] == _lesson_run(mem.SHARE)["final"]
    assert _lesson_run(0.0)["final"] != _lesson_run(mem.SHARE)["final"]


# --------------------------------------------------- live-like replay switches
@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2015-01-01", periods=800)
    closes = pd.DataFrame({s: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(index)))) for s in INDIA_TICKERS},
                          index=index)
    opens = closes.shift().fillna(closes) * (1 + rng.normal(0, 0.002, closes.shape))
    return opens, closes, bt.precompute_views(closes, "2016-06-01")


def _cash(run, opens, closes):
    return run["curve"] - stock_values(run, opens, closes).sum(axis=1).values


def test_no_borrowing_keeps_cash_at_or_above_zero(market):
    opens, closes, views = market
    base = bt.simulate(opens, closes, views, "2016-06-01", None)
    capped = bt.simulate(opens, closes, views, "2016-06-01", None, no_borrowing=True)
    assert _cash(base, opens, closes).min() < -1.0, "the default replay does dip below zero cash on this data"
    assert _cash(capped, opens, closes).min() > -1e-6


def test_whole_lots_trade_less_and_fine_lots_converge_on_the_rupee_replay(market):
    opens, closes, views = market
    capped = bt.simulate(opens, closes, views, "2016-06-01", None, no_borrowing=True)
    coarse = bt.simulate(opens, closes, views, "2016-06-01", None, no_borrowing=True,
                         lots_per_slot={s: 1.5 for s in INDIA_TICKERS})
    fine = bt.simulate(opens, closes, views, "2016-06-01", None, no_borrowing=True,
                       lots_per_slot={s: 400.0 for s in INDIA_TICKERS})
    invested = lambda run: float((1 - _cash(run, opens, closes) / run["curve"]).mean())   # noqa: E731
    assert invested(coarse) < invested(capped) - 0.1, "one-and-a-half shares a slot leaves it far less invested"
    assert len(coarse["trades"]) < len(capped["trades"])
    assert fine["curve"][-1] == pytest.approx(capped["curve"][-1], rel=0.01)


def test_the_fixed_share_setting_really_reaches_the_learner_in_a_replay(market):
    """On 16 Nov 2026 a variable-name collision made every replay ignore the setting and corrupted the
    baseline with it. A replay with the share step must differ from one without, and the default must be
    the live setting."""
    opens, closes, views = market
    plain = bt.simulate(opens, closes, views, "2016-06-01", None, fixed_share=0.0)
    shared = bt.simulate(opens, closes, views, "2016-06-01", None, fixed_share=0.05)
    default = bt.simulate(opens, closes, views, "2016-06-01", None)
    live = bt.simulate(opens, closes, views, "2016-06-01", None, fixed_share=mem.SHARE)
    assert plain["weights"] != shared["weights"], "the share step changes trust"
    assert np.array_equal(default["curve"], live["curve"]) and default["weights"] == live["weights"]
    spread = lambda w: max(w.values()) - min(w.values())   # noqa: E731
    assert spread(shared["weights"]) < spread(plain["weights"]), "it pulls trust back toward equal"
    assert all(0.04 < v < 0.9 for v in plain["weights"].values()), "trust stays a sane distribution"


# ---------------------------------------------------------------- event study
def test_event_study_finds_a_planted_effect_and_starts_the_day_after(market):
    _, closes, _ = market
    closes = closes.copy()
    rng = np.random.default_rng(0)
    days = closes.index[100:700:6]
    headlines = []
    for d in days:
        at = closes.index.get_loc(d)
        closes.iloc[at + 2:, 0] *= 1.02          # the first stock gains 2% two sessions after each headline
        headlines.append({"title": "Q2 profit beats", "published_utc": d.isoformat() + "+00:00", "relevant": True,
                          "sentiment": 0.8, "material": 0.9})
        headlines.append({"title": "share price today", "published_utc": d.isoformat() + "+00:00", "relevant": True,
                          "sentiment": 0.9, "material": 0.1})
    frame = event_study.events_frame({INDIA_TICKERS[0]: headlines}, closes)
    assert set(frame["event"]) == {"results"} and (frame["tone"] == "good").all(), "immaterial headlines are dropped"
    assert frame.groupby("week").size().max() == 1, "one event per stock, week, kind and tone"
    table = event_study.summarise(frame)
    row = next(r for r in table if r["span"] == "all")
    assert row["h5"]["mean"] > 0.01 and row["h5"]["reliable"] and row["h5"]["as_expected"]
    same_day = closes.copy()
    same_day.loc[days[10], INDIA_TICKERS[0]] *= 5      # a wild print on the headline's own day
    again = event_study.events_frame({INDIA_TICKERS[0]: headlines[20:22]}, same_day)
    before = event_study.events_frame({INDIA_TICKERS[0]: headlines[20:22]}, closes)
    assert again["excess_5"].iloc[0] == pytest.approx(before["excess_5"].iloc[0], abs=0.02), \
        "the headline day's own move is not counted"


# ------------------------------------------------------- proposals, director
def test_proposals_follow_the_registry_and_report_only_what_is_really_applied():
    def record(experiment, verdict):
        return {"id": experiment, "title": experiment, "review": {"verdict": verdict, "summary": "s"},
                "results": {"primary": {"label": "Sharpe", "values": {"tune": 0.1, "holdout": 0.2}}}}

    none = {p["id"]: p for p in proposals.build([])}
    assert all(p["status"] == "needs a test" for p in none.values())
    built = {p["id"]: p for p in proposals.build([record("wait-five-sessions", "supported"),
                                                  record("without-trend", "not_supported"),
                                                  record("learning-switched-off", "supported"),
                                                  record("no-borrowing", "not_supported")])}
    assert built["shorter-wait"]["status"] == "ready"
    assert built["drop-trend"]["status"] == "inconclusive", "'not shown to help' is not 'shown to hurt'"
    assert built["freeze-trust"]["status"] == "not supported", "if learning is shown to help, do not freeze it"
    assert built["replay-no-borrowing"]["status"] == "ready", "a correction to the measuring tool"
    applied = {k for k, p in built.items() if p["applied"]}
    assert applied == {"fixed-share"}, "only the change the owner decided on is live"


def test_the_standing_question_is_re_asked_only_when_the_snapshot_reaches_a_new_month():
    done = [{"id": "live-rule-vs-holding", "review": {"verdict": "inconclusive"},
             "results": {"data": {"last": "2026-11-13"}, "engine": {"live_learner": {"fixed_share": mem.SHARE},
                                                                    "replay": {"no_borrowing": True}}}}]
    same = director.build_agenda(None, [], done, snapshot_last="2026-11-28")
    assert same[0]["experiment"] == "live-rule-vs-holding" and same[0]["status"] == "answered"
    later = director.build_agenda(None, [], done, snapshot_last="2026-12-05")
    assert later[0]["experiment"] == "live-rule-vs-holding-2026-12" and later[0]["status"] == "proposed"
    assert "live-rule-vs-holding-2026-12" in director.specs("2026-12-05")


def test_the_standing_question_reopens_when_the_live_learner_changes():
    plain_hedge = [{"id": "live-rule-vs-holding", "review": {"verdict": "inconclusive"},
                    "results": {"data": {"last": "2026-11-13"}}}]      # filed before the learner was recorded
    agenda = director.build_agenda(None, [], plain_hedge, snapshot_last="2026-11-13")
    assert agenda[0]["experiment"] == "live-rule-vs-holding-fixed-share" and agenda[0]["status"] == "proposed"


# ------------------------------------------------------------------ Phase B
def test_rebound_state_needs_a_negative_slow_trend_and_a_positive_fast_one():
    from scaata.agent import advisors

    assert advisors.is_rebound(-0.3, 0.02) and not advisors.is_rebound(0.3, 0.02)
    assert not advisors.is_rebound(-0.3, -0.02) and not advisors.is_rebound(-0.3, 0.0)
    index = pd.bdate_range("2020-01-01", periods=320)
    fall_then_bounce = pd.Series(np.r_[np.linspace(100, 140, 260), np.linspace(140, 100, 40), np.linspace(100, 112, 20)],
                                 index=index)
    assert advisors.trend_view(fall_then_bounce) < 0 and advisors.fast_return(fall_then_bounce) > 0
    assert advisors.rebound_view(fall_then_bounce) == advisors.REBOUND_VIEW
    assert advisors.rebound_view(fall_then_bounce.iloc[:250]) == 0.0, "no rebound before the fall"
    assert advisors.rebound_view(fall_then_bounce.iloc[:30]) == 0.0, "too little history to say anything"
    assert "rebound" in advisors.SHADOW and "rebound" not in advisors.ADVISORS, "recorded and scored, never weighed"


def test_rebound_transform_only_withdraws_a_negative_trend_vote_and_never_looks_ahead(market):
    from scaata.research.engine import rebound_transform

    _, closes, views = market                                     # the fixture's precomputed views from 2016-06
    out = rebound_transform(views, closes)
    changed = 0
    for day in list(views)[:300]:
        for s in INDIA_TICKERS:
            before, after = views[day][s], out[day][s]
            assert after["volatility"] == before["volatility"] and after["momentum"] == before["momentum"]
            assert after["trend"] in (before["trend"], 0.0) and not (before["trend"] >= 0 and after["trend"] != before["trend"])
            changed += after["trend"] != before["trend"]
    assert changed > 0, "the synthetic world contains rebounds"
    day = list(views)[120]
    k = closes.index.get_loc(day)
    wrecked = closes.copy()
    wrecked.iloc[k + 1:] *= 7.0                                   # change only prices after that day
    again = rebound_transform({day: views[day]}, wrecked)
    assert again[day] == out[day], "a day's view does not depend on later prices"


def test_capture_measures_how_much_of_holdings_rebound_was_recovered():
    from scaata.research.rebound import capture

    holding = [100, 80, 70, 85, 105]                              # low of 70, then +50%
    assert capture([100, 90, 85, 95, 110], holding) == pytest.approx(((110 / 85) - 1) / 0.5)
    assert capture([100, 99, 98, 98.5, 99], [100, 99, 98, 98.5, 99]) is None, "no real rebound, nothing to recapture"
