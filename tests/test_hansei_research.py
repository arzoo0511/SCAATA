"""The Research Lab: criteria are filed before a run, results are never
rewritten, and the reviewer fails an experiment that looked at the future."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scaata.agent import backtest as bt
from scaata.config import INDIA_TICKERS
from scaata.research import director, legacy, registry
from scaata.research.engine import Engine, regimes, silence
from scaata.research.review import review

PERIODS = {"tune": ("2016-06-01", "2017-03-31"), "holdout": ("2017-04-03", None)}


def _spec(**over) -> dict:
    return {**director.baseline_spec(), **over}


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2015-01-01", periods=760)
    closes = pd.DataFrame({s: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(index)))) for s in INDIA_TICKERS},
                          index=index)
    opens = closes.shift().fillna(closes) * (1 + rng.normal(0, 0.002, closes.shape))
    return opens, closes


def _record(metrics: dict, primary: dict, prob: float | None = 0.99, **over) -> dict:
    """A finished experiment as the registry would hand it to the reviewer."""
    span = {"sessions": 2000, "trades": 100, "costs": 50.0}
    return {"id": "x", "origin": "lab", "registered_at": "2026-01-01T00:00:00+00:00",
            "ran_at": "2026-01-01T00:05:00+00:00",
            "criteria": [{"metric": "holdout.sharpe_vs_baseline", "op": ">", "value": 0.0, "label": "better"}],
            "results": {"metrics": metrics, "periods": {"tune": dict(span), "holdout": dict(span)},
                        "data": {"sha256": "a" * 64, "last": "2026-01-01"},
                        "integrity": {"checked": True, "invariant": True, "sessions": 260, "days_sampled": 16},
                        "primary": {"label": "Sharpe against the live rule", "values": primary,
                                    "prob_better": {} if prob is None else {"holdout": prob}}}, **over}


# ---------------------------------------------------------------- registry
def test_registry_is_append_only_and_criteria_cannot_be_rewritten(tmp_path):
    path = tmp_path / "experiments.jsonl"
    registry.register(_spec(), path)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_spec(hypothesis="something easier to pass"), path)
    registry.record_result("live-rule-vs-holding", {"metrics": {}}, path)
    with pytest.raises(ValueError, match="already has a result"):
        registry.record_result("live-rule-vs-holding", {"metrics": {"better": 1}}, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["registered", "result"]
    record = registry.get("live-rule-vs-holding", path)
    assert record["registered_at"] <= record["ran_at"] and registry.status(record) == "ran"


def test_an_experiment_must_be_registered_with_its_criteria_before_it_can_have_a_result(tmp_path):
    path = tmp_path / "experiments.jsonl"
    with pytest.raises(KeyError):
        registry.record_result("never-filed", {"metrics": {}}, path)
    spec = _spec()
    del spec["criteria"]
    with pytest.raises(ValueError, match="criteria"):
        registry.register(spec, path)
    with pytest.raises(ValueError, match="verdict"):
        registry.register(_spec(), path) and registry.record_review("live-rule-vs-holding", {"verdict": "great"}, path)


# ------------------------------------------------------------------ review
def test_supported_needs_criteria_and_every_check():
    good = _record({"holdout.sharpe_vs_baseline": 0.2}, {"tune": 0.1, "holdout": 0.2})
    assert review(good)["verdict"] == "supported"


def test_a_missed_criterion_is_not_supported_whatever_else_passes():
    record = _record({"holdout.sharpe_vs_baseline": -0.1}, {"tune": -0.2, "holdout": -0.1}, prob=0.01)
    verdict = review(record)
    assert verdict["verdict"] == "not_supported" and verdict["criteria"][0]["passed"] is False


def test_an_effect_in_one_span_only_or_within_noise_is_inconclusive():
    flips = _record({"holdout.sharpe_vs_baseline": 0.2}, {"tune": -0.1, "holdout": 0.2})
    assert review(flips)["verdict"] == "inconclusive"
    noisy = _record({"holdout.sharpe_vs_baseline": 0.2}, {"tune": 0.1, "holdout": 0.2}, prob=0.7)
    assert review(noisy)["verdict"] == "inconclusive"


def test_each_look_at_the_holdout_raises_the_bar():
    record = _record({"holdout.sharpe_vs_baseline": 0.2}, {"tune": 0.1, "holdout": 0.2}, prob=0.93)
    assert review(record)["verdict"] == "supported"
    earlier = [{"id": f"e{i}", "origin": "lab", "results": {}, "registered_at": "2025-12-01T00:00:00+00:00"}
               for i in range(4)]
    crowded = review(record, earlier)
    assert crowded["holdout_looks"] == 5 and crowded["verdict"] == "inconclusive"


@pytest.mark.parametrize("broken", ["late_criteria", "no_snapshot", "look_ahead", "no_costs"])
def test_an_integrity_failure_fails_review_even_with_perfect_numbers(broken):
    record = _record({"holdout.sharpe_vs_baseline": 5.0}, {"tune": 5.0, "holdout": 5.0})
    if broken == "late_criteria":
        record["registered_at"] = "2026-02-01T00:00:00+00:00"
    elif broken == "no_snapshot":
        record["results"]["data"] = {}
    elif broken == "look_ahead":
        record["results"]["integrity"].update(invariant=False, days_with_leaked_views=["2020-01-02"])
    else:
        record["results"]["periods"]["holdout"]["costs"] = 0.0
    assert review(record)["verdict"] == "failed_review"


# ------------------------------------------------------------------ engine
def test_the_real_engine_passes_the_look_ahead_test_and_a_leaky_advisor_fails_review(market, tmp_path):
    opens, closes = market
    meta = {"sha256": "b" * 64, "last": str(closes.index[-1].date())}

    def leaky(frame, start):
        """An advisor that reads tomorrow's close."""
        honest, tomorrow = bt.precompute_views(frame, start), frame.shift(-1)
        return {t: {s: {**v, "trend": float(np.sign(tomorrow.at[t, s] / frame.at[t, s] - 1))
                        if pd.notna(tomorrow.at[t, s]) else 0.0} for s, v in by.items()} for t, by in honest.items()}

    path = tmp_path / "experiments.jsonl"
    verdicts = {}
    for name, views_fn in (("honest", bt.precompute_views), ("leaky", leaky)):
        engine = Engine(opens, closes, meta, views_fn=views_fn, periods=PERIODS)
        spec = _spec(id=name)
        registry.register(spec, path)
        registry.record_result(name, engine.run(spec), path)
        verdicts[name] = review(registry.get(name, path), registry.load(path))
    assert registry.get("honest", path)["results"]["integrity"]["invariant"] is True
    assert verdicts["honest"]["verdict"] != "failed_review"
    assert verdicts["leaky"]["verdict"] == "failed_review"
    assert registry.get("leaky", path)["results"]["metrics"]["holdout.sharpe_vs_holding"] > 0, \
        "the leak makes the numbers look good; the reviewer must still fail it"


def test_baseline_run_matches_the_backtest_it_wraps(market):
    opens, closes = market
    engine = Engine(opens, closes, {"sha256": "c" * 64}, periods=PERIODS)
    out = engine.run(_spec())
    direct = bt.evaluate(bt.simulate(opens, closes, bt.precompute_views(closes, "2016-06-01"), *PERIODS["tune"]))
    assert out["periods"]["tune"]["variant"]["cagr"] == pytest.approx(direct["hansei"]["cagr"])
    assert out["metrics"]["tune.sharpe_vs_holding"] == pytest.approx(direct["sharpe_diff"])
    assert out["periods"]["tune"]["costs"] > 0 and out["data"]["sha256"] == "c" * 64


def test_silencing_an_advisor_changes_only_that_advisor():
    views = {pd.Timestamp("2020-01-01"): {"A": {"trend": 0.5, "momentum": -0.2}}}
    assert silence(views, "trend") == {pd.Timestamp("2020-01-01"): {"A": {"trend": 0.0, "momentum": -0.2}}}
    assert views[pd.Timestamp("2020-01-01")]["A"]["trend"] == 0.5


def test_regime_labels_use_only_earlier_days(market):
    _, closes = market
    full = regimes(closes)
    cut = regimes(closes.iloc[:600])
    pd.testing.assert_frame_equal(full.iloc[:600], cut)
    assert full.iloc[:126].isna().all().all(), "no label until there is six months of history"


# ---------------------------------------------------------------- director
def test_director_raises_questions_from_the_live_record_and_marks_answered_ones():
    evaluation = {"performance": {"gap": -0.05, "gap_from_exposure": -0.053, "gap_from_selection": 0.002,
                                  "average_invested": 0.46},
                  "advisors": {"20_sessions": {"trend": {"judged": 67, "hit_rate": 0.46},
                                               "volatility": {"judged": 21, "hit_rate": 0.29},
                                               "momentum": {"judged": 70, "hit_rate": 0.60}}},
                  "trust": {"trend": 0.21, "volatility": 0.23, "momentum": 0.31, "news": 0.25}, "health": {}}
    journal = [{"decisions": [{"current_exposure": 0.4, "target_exposure": 0.7}]}]
    agenda = director.build_agenda(evaluation, journal, [])
    proposed = {a["experiment"] for a in agenda if a["status"] == "proposed"}
    assert proposed == {"live-rule-vs-holding", "edge-by-market-condition", "without-trend", "learning-switched-off"}
    assert "without-volatility" not in proposed, "21 judged opinions is too few to raise a question"
    assert any(a["status"] == "needs engine work" and "whole-share" in a["question"] for a in agenda)
    done = [{"id": "without-trend", "review": {"verdict": "not_supported"}}]
    answered = next(a for a in director.build_agenda(evaluation, journal, done) if a["experiment"] == "without-trend")
    assert answered["status"] == "answered" and answered["verdict"] == "not_supported"
    assert all(spec["criteria"] and spec["limitations"] for spec in director.specs().values())


def test_legacy_experiments_are_entered_once_and_marked_as_not_reviewed(tmp_path):
    path = tmp_path / "experiments.jsonl"
    added = legacy.seed(path)
    assert added and legacy.seed(path) == []
    for record in registry.load(path):
        assert record["origin"] == "legacy" and record["review"]["legacy"] is True
        assert record["results"]["metrics"], "numbers come from the original result files"
