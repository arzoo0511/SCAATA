"""Cognitive HANSEI: a belief's status comes from its evidence, predictions are
scored honestly, and the history of beliefs is never rewritten."""
from __future__ import annotations

import json

import pytest

from scaata.cognition import beliefs
from scaata.cognition.__main__ import load_history, snapshot


def _memory(days: int, right: bool, view: float = 0.5) -> dict:
    """`days` judged days, two stocks a day, where the trend advisor is always right or always wrong."""
    obs = []
    for d in range(days):
        for symbol in ("A.NS", "B.NS"):
            outcome = 0.03 if d % 2 else -0.03
            sign = (1 if outcome > 0 else -1) * (1 if right else -1)
            obs.append({"date": f"2026-01-{d + 1:02d}", "symbol": symbol, "judged": True, "forward_return": outcome,
                        "views": {"trend": sign * view, "volatility": 0.01, "momentum": 0.0, "news": 0.0}})
    obs.append({"date": "2026-03-01", "symbol": "A.NS", "judged": False, "views": {"trend": 0.9}})
    return {"observations": obs, "weights": {"trend": 0.4, "volatility": 0.2, "momentum": 0.2, "news": 0.2},
            "lessons": []}


def test_wilson_interval_is_wide_when_there_is_little_data():
    low, high = beliefs.wilson(3, 4)
    assert low < 0.5 < high, "three right out of four is not evidence"
    low, high = beliefs.wilson(90, 100)
    assert low > 0.8 and high < 0.96
    assert beliefs.wilson(0, 0) == (0.0, 1.0)


def test_predictions_pair_views_with_outcomes_and_skip_silence():
    journal = [{"session": "2026-01-01", "decisions": [{"symbol": "A.NS", "score": -0.3}]}]
    rows = beliefs.predictions(_memory(2, right=True), journal)
    assert {r["source"] for r in rows} == {"trend", "brain"}, "views under 0.05 are no opinion; unjudged are skipped"
    assert len([r for r in rows if r["source"] == "trend"]) == 4 and all(
        r["right"] for r in rows if r["source"] == "trend")
    brain = next(r for r in rows if r["source"] == "brain")
    assert brain["right"] is True and brain["outcome"] == -0.03, "a negative score before a fall is right"


@pytest.mark.parametrize("right,status", [(True, "established"), (False, "contradicted")])
def test_a_long_consistent_record_settles_the_question(right, status):
    card = beliefs.scorecard(beliefs.predictions(_memory(20, right), []))
    assert card["trend"]["status"] == status and card["trend"]["days"] == 20 and card["trend"]["opinions"] == 40
    assert card["news"]["status"] == "untested" and card["news"]["hit_rate"] is None


def test_a_short_record_cannot_tell_however_good_it_looks():
    card = beliefs.scorecard(beliefs.predictions(_memory(4, right=True), []))
    assert card["trend"]["hit_rate"] == 1.0 and card["trend"]["status"] == "cannot tell"
    assert card["trend"]["days"] == 4, "eight opinions on four days are judged as four days"


def test_calibration_groups_weak_and_strong_views():
    memory = _memory(12, right=True)
    for i, obs in enumerate(o for o in memory["observations"] if o["judged"]):
        obs["views"]["trend"] *= 0.2 + i / 30
    groups = beliefs.calibration(beliefs.predictions(memory, []), "trend")
    assert [g["bucket"] for g in groups] == ["weakest", "middle", "strongest"]
    assert groups[0]["mean_strength"] < groups[2]["mean_strength"] and sum(g["n"] for g in groups) == 24
    assert beliefs.calibration([], "trend") == []


def test_relationship_and_finding_status_follow_the_evidence(tmp_path):
    sens = {"weeks": 900, "first": "2007-01-01", "last": "2026-01-01", "split": "2017-01-01", "table": [
        {"symbol": "IOC", "name": "Indian Oil", "factor": "crude", "beta": -0.11, "low": -0.17, "high": -0.05,
         "reliable": True, "before": -0.2, "after": -0.05},
        {"symbol": "ITC", "name": "ITC", "factor": "rupee", "beta": 0.0, "low": -0.2, "high": 0.2, "reliable": False,
         "before": None, "after": None}]}
    (tmp_path / "sens.json").write_text(json.dumps(sens), encoding="utf-8")
    spec = {"id": "x", "title": "X", "hypothesis": "X helps.", "origin": "lab"}
    events = [{"event": "registered", "id": "x", "at": "t", "spec": spec},
              {"event": "result", "id": "x", "at": "t", "results": {}},
              {"event": "review", "id": "x", "at": "t", "review": {"verdict": "not_supported", "summary": "No.",
                                                                    "criteria": [{"passed": False}]}}]
    (tmp_path / "reg.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    missing = tmp_path / "missing.json"
    state = beliefs.build({"sensitivities": tmp_path / "sens.json", "registry": tmp_path / "reg.jsonl",
                           "memory": missing, "journal": missing, "evaluation": missing, "book": missing,
                           "agenda": missing, "macro_runs": tmp_path / "none", "backtest": missing})
    by_key = {b["key"]: b for b in state["beliefs"]}
    assert by_key["IOC:crude"]["status"] == "established" and "against" in by_key["IOC:crude"]["statement"]
    assert by_key["ITC:rupee"]["status"] == "cannot tell"
    assert by_key["x"]["status"] == "contradicted" and by_key["x"]["kind"] == "finding"
    assert "0 of 1 criteria" in by_key["x"]["evidence"]
    assert by_key["no-leverage"]["status"] == "untested", "with no replay on file it cannot claim either way"
    assert all(b["status"] in beliefs.STATUSES and b["kind"] in beliefs.KINDS for b in state["beliefs"])
    assert state["working"] is None and state["predictions"]["judged"] == 0


def test_the_real_state_builds_and_every_belief_cites_its_evidence():
    state = beliefs.build()
    assert state["beliefs"] and len({b["key"] for b in state["beliefs"]}) == len(state["beliefs"])
    assert all(b["statement"] and b["source"] for b in state["beliefs"])
    assert sum(state["counts"].values()) == len(state["beliefs"])


def test_belief_history_is_append_only_and_records_changes(tmp_path):
    path = tmp_path / "history.jsonl"
    state = {"beliefs": [{"key": "a", "kind": "fact", "status": "cannot tell", "statement": "A", "evidence": "",
                          "source": "s"}]}
    assert len(snapshot(state, path, at="2026-01-01")) == 1
    assert snapshot(state, path, at="2026-01-02") == [], "nothing changed, nothing written"
    state["beliefs"][0]["status"] = "established"
    added = snapshot(state, path, at="2026-01-03")
    assert added[0]["was"] == "cannot tell" and added[0]["status"] == "established"
    history = load_history(path)
    assert [h["status"] for h in history] == ["cannot tell", "established"], "the earlier belief is still on file"
    assert history[0]["at"] == "2026-01-01"
