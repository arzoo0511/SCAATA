"""Phase E: the self-proposed experiment generator and its guards."""
import numpy as np
import pytest

from scaata.research import generator as G


def test_reality_check_flags_a_real_edge_and_not_luck():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, (2500, 12))
    ps = [G.reality_check(np.random.default_rng(s).normal(0, 0.01, (2500, 12)), draws=300)["p"] for s in range(1, 9)]
    assert 0.2 < np.mean(ps) < 0.8 and sum(p < 0.05 for p in ps) <= 2   # pure noise: p is spread out, rarely small
    planted = noise.copy()
    planted[:, 3] += 0.0015                                             # one candidate with a real daily edge
    out = G.reality_check(planted, draws=500)
    assert out["p"] < 0.05 and out["best_index"] == 3


def test_best_of_many_needs_more_than_best_of_one():
    rng = np.random.default_rng(1)
    d = rng.normal(0, 0.01, (2500, 1))
    one = G.reality_check(d, draws=500)["luck_only_median"]
    many = G.reality_check(np.repeat(d, 1, axis=1).repeat(1, axis=1) * 0 + rng.normal(0, 0.01, (2500, 30)), draws=500)["luck_only_median"]
    assert many > one


def test_candidates_state_reason_cost_and_a_single_knob():
    ids = [c["id"] for c in G.CANDIDATES]
    assert len(ids) == len(set(ids))
    for c in G.CANDIDATES:
        assert c["reason"] and c["cost"] and len(c["params"]) == 1


def test_stationary_bootstrap_stays_in_range():
    idx = G.stationary_indices(500, np.random.default_rng(0))
    assert idx.min() >= 0 and idx.max() < 500 and len(idx) == 500


class _Engine:
    """Stands in for the real engine: a flat live rule and candidates that differ only by noise."""
    meta = {"last": "2026-11-18", "sha256": "x"}
    views = {}

    def _run(self, period, views, **params):
        assert period == "tune"                                         # the holdout is never opened
        rng = np.random.default_rng(abs(hash(tuple(sorted(params.items())))) % 2**32 if params else 0)
        curve = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 900))
        return {"curve": curve, "days": list(range(900))}


def test_batch_is_logged_before_results_and_cannot_be_rerun(tmp_path, monkeypatch):
    import scaata.agent.backtest as bt
    monkeypatch.setattr(bt, "evaluate", lambda r: {"hansei": {"sharpe": float(r["curve"][-1] / 100), "cagr": 0.1,
                                                              "max_drawdown": -0.2}, "trades": 100, "costs": 1.0})
    log, res = tmp_path / "g.jsonl", tmp_path / "g.json"
    out = G.run(_Engine(), [], batch="2099Q1", log=log, result=res)
    events = G._events(log)
    assert [e["event"] for e in events] == ["register", "results"]
    assert events[0]["at"] <= events[1]["at"] and events[0]["holdout_sealed"] is True
    assert out["tried"] == len(G.CANDIDATES) and out["reality_check"]["p"] > 0.05 and out["champion"] is None
    with pytest.raises(RuntimeError):
        G.run(_Engine(), [], batch="2099Q1", log=log, result=res)
