"""Phase A's learners must reproduce the live learner exactly, and the tests built on them must behave."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from scaata.agent import memory as mem
from scaata.agent.advisors import ADVISORS
from scaata.config import INDIA_TICKERS
from scaata.research import learners as L


def _world(n_days: int = 700, seed: int = 0):
    """Prices and views where trend carries real information and news is only sometimes awake."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2012-01-02", periods=n_days)
    closes = pd.DataFrame({s: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n_days))) for s in INDIA_TICKERS},
                          index=index)
    future = closes.shift(-20) / closes - 1
    views = {}
    for i, day in enumerate(index):
        views[day] = {}
        for s in INDIA_TICKERS:
            f = future.at[day, s]
            signal = np.sign(f) if np.isfinite(f) else 0.0
            views[day][s] = {"trend": 0.4 * signal * (1 if rng.random() < 0.58 else -1),
                             "volatility": float(rng.uniform(-0.3, 0.3)), "momentum": float(rng.uniform(-0.5, 0.5)),
                             "news": float(rng.uniform(-0.8, 0.8)) if rng.random() < 0.35 else 0.0}
    return closes, views


@pytest.fixture(scope="module")
def world():
    closes, views = _world()
    return closes, views, L.build_arrays(views, closes)


def test_arrays_line_up_with_the_prices(world):
    closes, views, arrays = world
    assert arrays["views"].shape == (len(closes), 5, 4) and arrays["fwd"].shape == (len(closes), 5)
    assert np.isnan(arrays["fwd"][-20:]).all() and np.isfinite(arrays["fwd"][:-20]).all()
    k, s = 100, 2
    hurdle = L.liquid_yield(closes.index[k].year) * 20 / 252
    assert arrays["fwd"][k, s] == pytest.approx(closes.iloc[k + 20, s] / closes.iloc[k, s] - 1 - hurdle, abs=1e-4)
    assert arrays["views"][k, s, 0] == views[closes.index[k]][INDIA_TICKERS[s]]["trend"]


@pytest.mark.parametrize("share", [0.0, mem.SHARE])
def test_the_lean_learner_reproduces_the_live_learner(world, share, monkeypatch):
    """Same weights as `memory.learn` on every day, to the four decimals the live learner stores."""
    closes, views, arrays = world
    monkeypatch.setattr(mem, "SHARE", share)
    memory = mem.new_memory()
    live = []
    for day in closes.index:
        mem.learn(memory, {s: closes[s].loc[:day] for s in INDIA_TICKERS}, day.date())
        live.append([memory["weights"][a] for a in ADVISORS])
        for s in INDIA_TICKERS:
            mem.remember(memory, day.date(), s, views[day][s], float(closes.at[day, s]))
    monkeypatch.setattr(L, "SHARE", share)
    lean = L.run_learner(arrays, "live")["weights"]
    assert np.abs(lean - np.array(live)).max() < 2e-4, "the learner used for Phase A must be the live one"
    assert np.abs(lean[-1] - np.array(live)[-1]).max() < 2e-4


def test_every_learner_keeps_trust_a_distribution_inside_the_floor(world):
    _, _, arrays = world
    for kind in L.LEARNERS:
        w = L.run_learner(arrays, kind)["weights"]
        assert np.allclose(w.sum(axis=1), 1.0, atol=2e-3), kind
        assert w.min() >= mem.MIN_WEIGHT - 1e-3 or kind in ("equal",), kind
    assert np.allclose(L.run_learner(arrays, "equal")["weights"], 0.25)
    shrunk = L.run_learner(arrays, "shrunk")
    assert shrunk["shrink"][0] == 0.0 and shrunk["shrink"][-1] > shrunk["shrink"][50]


def test_a_learner_beats_equal_trust_when_one_advisor_really_is_better(world):
    _, _, arrays = world
    rows = {r["kind"]: r for r in L.compare_learners(arrays, ["live", "hedge"])}
    assert rows["hedge"]["payoff"]["all"]["improvement"] > 0, "trend is informative here, so learning should help"
    assert rows["hedge"]["final_weights"][0] == max(rows["hedge"]["final_weights"])
    assert rows["hedge"]["all"]["p_corrected"] == pytest.approx(min(1.0, rows["hedge"]["all"]["p"] * 2))


def test_isotonic_fit_is_monotone_and_recovers_a_step():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 2000)
    y = (rng.random(2000) < np.where(x > 0.5, 0.7, 0.3)).astype(float)
    fit = L.isotonic(x, y)
    assert (np.diff(fit[0]) >= -1e-12).all()
    pred = L.apply_isotonic(fit, np.array([0.1, 0.9]))
    assert pred[0] == pytest.approx(0.3, abs=0.06) and pred[1] == pytest.approx(0.7, abs=0.06)


def test_poison_test_demotes_the_adversary_and_reports_what_it_could_not_do(world):
    _, _, arrays = world
    result = L.poison_test(arrays, seeds=6)
    assert result["passes"]["adversary_demoted"] and result["rates"]["adversary_demoted"] == 1.0
    assert result["mean_trust"]["adversary"] < result["mean_trust"]["random"]
    assert set(result["rates"]) == {"adversary_demoted", "random_not_promoted", "informative_top2",
                                    "weak_adversary_below_equal"}
    assert all(0.0 <= v <= 1.0 for v in result["rates"].values()), "pass rates are reported, not assumed"


def test_skill_ledger_and_calibration_run_and_see_the_planted_skill(world, monkeypatch):
    _, _, arrays = world
    monkeypatch.setattr(L, "SPANS", {"tune": ("2012-01-01", "2013-06-30"), "holdout": ("2013-07-01", None)})
    ledger = {(r["source"], r["span"]): r for r in L.skill_ledger(arrays)}
    assert ledger[("trend", "all")]["skill"] > ledger[("volatility", "all")]["skill"]
    assert ledger[("trend", "all")]["hit_rate"] > 0.55
    calibration = L.calibration(arrays)           # the equal blend has a continuous range of views
    assert len(calibration["reliability"]["tune"]) >= 5
    assert set(calibration["strong_vs_weak"]) == {"tune", "holdout"} and "improvement" in calibration["recalibration"]
    assert L.calibration({**arrays, "days": arrays["days"] - pd.DateOffset(years=30)}, "trend")["recalibration"] is None,         "with no data in a span it says so instead of inventing a number"


def test_shapley_values_add_up_and_treat_players_fairly():
    from scaata.research.contribution import shapley

    players = ("a", "b", "c")
    value = {frozenset(): 1.0}
    for k in range(1, 4):
        import itertools
        for subset in itertools.combinations(players, k):
            value[frozenset(subset)] = 1.0 + (2.0 if "a" in subset else 0.0) + (1.0 if len(subset) >= 2 and "b" in subset
                                                                                  and "c" in subset else 0.0)
    phi = shapley(value, players)
    assert sum(phi.values()) == pytest.approx(value[frozenset(players)] - value[frozenset()]), "efficiency"
    assert phi["a"] == pytest.approx(2.0), "a player whose effect never depends on the others gets exactly it"
    assert phi["b"] == pytest.approx(phi["c"]) == pytest.approx(0.5), "symmetric players are paid the same"
    nothing = {frozenset(s): 5.0 for k in range(4) for s in itertools.combinations(players, k)}
    assert all(v == pytest.approx(0.0) for v in shapley(nothing, players).values()), "a null player is worth zero"
