"""Macroverse: a replay is the real engine on real prices with no view of the
future; a scenario is labelled synthetic and built the way it says it is."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scaata.agent import backtest as bt
from scaata.config import INDIA_TICKERS
from scaata.macroverse import replay, shock

TRUE_BETA = {"market": 1.2, "crude": -0.3, "rupee": 0.0}


@pytest.fixture(scope="module")
def world():
    """1,400 sessions where every stock is 1.2 x market - 0.3 x crude + noise, and the rupee does nothing."""
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2014-01-01", periods=1400)
    moves = pd.DataFrame(rng.normal(0, [0.010, 0.020, 0.004], (len(index), 3)), index=index,
                         columns=["market", "crude", "rupee"])
    factors = 100 * np.exp(moves.cumsum())
    driven = moves["market"] * TRUE_BETA["market"] + moves["crude"] * TRUE_BETA["crude"]
    closes = pd.DataFrame({s: 100 * np.exp((driven + rng.normal(0.0002, 0.008, len(index))).cumsum())
                           for s in INDIA_TICKERS}, index=index)
    opens = closes.shift().fillna(closes) * (1 + rng.normal(0, 0.002, closes.shape))
    return opens, closes, factors


EPISODE = {"id": "test", "title": "Test episode", "summary": "", "start": "2016-06-01", "end": "2016-12-30",
           "events": [("2016-07-02", "a Saturday event"), ("2030-01-01", "an event after the episode")]}


def test_replay_is_the_real_engine_and_rebases_both_at_100(world):
    opens, closes, factors = world
    out = replay.run_episode(EPISODE, opens, closes, factors)
    assert out["kind"] == "replay" and out["hansei"][0] == 100 and out["holding"][0] == 100
    sessions = closes.index[(closes.index >= "2016-06-01") & (closes.index <= "2016-12-30")]
    warm = closes.index[closes.index.get_loc(sessions[0]) - replay.WARMUP_SESSIONS]
    cut_o, cut_c = opens.loc[:sessions[-1]], closes.loc[:sessions[-1]]
    direct = bt.simulate(cut_o, cut_c, bt.precompute_views(cut_c, str(warm.date())), str(warm.date()),
                         str(sessions[-1].date()))
    inside = pd.DatetimeIndex(direct["days"]) >= sessions[0]
    assert out["hansei"][-1] == pytest.approx(direct["curve"][inside][-1] / direct["curve"][inside][0] * 100, abs=1e-3)
    assert out["stats"]["hansei"]["return"] == pytest.approx(out["hansei"][-1] / 100 - 1, abs=1e-4)
    assert out["sessions"] == len(sessions) == len(out["dates"]) == len(out["invested"])
    assert set(out["factors"]) == {"market", "crude", "rupee"} and out["factors"]["crude"][0] == 100


def test_replay_never_sees_prices_after_the_episode(world):
    opens, closes, factors = world
    full = replay.run_episode(EPISODE, opens, closes, factors)
    wrecked_o, wrecked_c = opens.copy(), closes.copy()
    after = closes.index > pd.Timestamp(EPISODE["end"])
    wrecked_o.loc[after] *= 5.0
    wrecked_c.loc[after] *= 5.0
    again = replay.run_episode(EPISODE, wrecked_o, wrecked_c, factors)
    assert again["hansei"] == full["hansei"] and again["trades"] == full["trades"]


def test_replay_books_balance_and_events_land_on_sessions(world):
    opens, closes, factors = world
    out = replay.run_episode(EPISODE, opens, closes, factors)
    # The backtest engine lets cash go slightly negative when holdings outgrow their slots;
    # the replay must report that honestly rather than hide it.
    peak = max(out["invested"])
    assert min(out["invested"]) >= -1e-6 and peak < 1.25
    assert out["stats"]["peak_invested"] == pytest.approx(peak, abs=1e-3)
    assert (peak > 1.005) == any("cash dip below zero" in note for note in out["notes"])
    assert all(e >= -1e-6 for s in out["stocks"] for e in s["exposure"])
    assert [e["date"] for e in out["events"]] == ["2016-07-04"], "weekend event moves to Monday; later one dropped"
    assert all(0 <= t["index"] < out["sessions"] for t in out["trades"])
    assert len(out["trades"]) == out["stats"]["trades"] and (out["stats"]["costs"] > 0) == bool(out["trades"])


def test_drawdown_and_recovery():
    curve = np.array([100.0, 110.0, 88.0, 99.0, 110.0, 121.0])
    stats = replay.summarise(curve)
    assert stats["max_drawdown"] == pytest.approx(-0.2) and stats["trough_index"] == 2
    assert stats["recovery_sessions"] == 2 and stats["return"] == pytest.approx(0.21)
    assert replay.summarise(np.array([100.0, 80.0, 90.0]))["recovery_sessions"] is None


def test_the_shock_is_exactly_what_it_says(world):
    path = shock.factor_path({"crude": 0.60, "market": -0.20})
    total = np.exp(path.sum(axis=0)) - 1
    assert total == pytest.approx([-0.20, 0.60, 0.0])
    assert not path[shock.RAMP:].any(), "nothing moves after the shock has arrived"


def test_sensitivities_recover_what_drives_the_stocks_and_admit_what_they_cannot_see(world):
    _, closes, factors = world
    sens = shock.sensitivities(closes, factors)
    rows = {(r["symbol"], r["factor"]): r for r in sens["table"]}
    for symbol in ("HDFCBANK", "ITC"):
        assert rows[(symbol, "market")]["beta"] == pytest.approx(TRUE_BETA["market"], abs=0.15)
        assert rows[(symbol, "crude")]["beta"] == pytest.approx(TRUE_BETA["crude"], abs=0.1)
        assert rows[(symbol, "market")]["reliable"] and rows[(symbol, "crude")]["reliable"]
        assert rows[(symbol, "market")]["low"] < TRUE_BETA["market"] < rows[(symbol, "market")]["high"]
    undetectable = [rows[(s.replace(".NS", ""), "rupee")]["reliable"] for s in INDIA_TICKERS]
    assert sum(undetectable) <= 1, "a factor that does nothing should not be reported as reliable"
    noise = shock.residuals(closes, factors, sens["point"])
    assert np.abs(noise.mean(axis=0)).max() < 1e-12, "resampled noise carries no drift of its own"


def test_a_scenario_is_labelled_synthetic_reproducible_and_moves_the_right_way(world):
    opens, closes, factors = world
    sens = shock.sensitivities(closes, factors)
    crash = {"id": "crash", "title": "Crash", "summary": "", "shock": {"market": -0.30}}
    calm = {"id": "calm", "title": "Calm", "summary": "", "shock": {}}
    a = shock.run_scenario(crash, opens, closes, factors, paths=2, seed=5, sens=sens)
    b = shock.run_scenario(crash, opens, closes, factors, paths=2, seed=5, sens=sens)
    c = shock.run_scenario(calm, opens, closes, factors, paths=2, seed=5, sens=sens)
    assert a["kind"] == "scenario" and "Synthetic" in a["notes"][0] and a["anchor"] == str(closes.index[-1].date())
    assert a["hansei"] == b["hansei"] and a["stats"] == b["stats"], "same seed, same result"
    assert len(a["days"]) == shock.HORIZON + 1 == len(a["hansei"]["p50"]) and a["holding"]["p50"][0] == 100
    assert a["stats"]["holding"]["median_return"] < c["stats"]["holding"]["median_return"] - 0.15
    assert all(s["expected_move"] == pytest.approx(np.exp(np.log(0.7) * TRUE_BETA["market"]) - 1, abs=0.06)
               for s in a["stocks"])
    assert all(s["expected_move"] == 0 for s in c["stocks"])
