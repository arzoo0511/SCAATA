import numpy as np
import pandas as pd
import pytest

from scaata.agent import advisors
from scaata.agent.signal_study import all_views, forward_excess, score, study


def _closes(days=600, symbols=("A.NS", "B.NS")):
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2015-01-01", periods=days)
    return pd.DataFrame({s: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, days))) for s in symbols}, index=index)


@pytest.mark.parametrize("k", [260, 333, 450, 599])
def test_vectorized_views_match_the_live_advisors(k):
    closes = _closes()
    vec = all_views(closes)
    for s in closes:
        live = advisors.views(closes[s].iloc[:k + 1], None)
        for name in ("trend", "volatility", "momentum"):
            assert vec[name][s].iloc[k] == pytest.approx(live[name], abs=6e-4)


def test_forward_excess_subtracts_the_cash_rate():
    index = pd.bdate_range("2024-01-01", periods=30)
    flat = pd.DataFrame({"A.NS": np.full(30, 100.0)}, index=index)
    out = forward_excess(flat, 5)
    assert out["A.NS"].iloc[0] < 0 and np.isnan(out["A.NS"].iloc[-1])


def test_a_perfect_signal_has_a_positive_spread_and_a_useless_one_does_not():
    closes = _closes(800, tuple(f"S{i}.NS" for i in range(8)))
    excess = forward_excess(closes, 20)
    rng = np.random.default_rng(0)
    perfect = score(np.sign(excess).replace(0, 1.0) * 0.5, excess, rng)
    noise = score(pd.DataFrame(rng.choice([-0.5, 0.5], excess.shape), excess.index, excess.columns), excess, rng)
    assert perfect["spread"] > 0 and perfect["prob_spread_positive"] == 1.0
    assert noise["spread_ci95"][0] < 0 < noise["spread_ci95"][1]


def test_study_reports_every_split_horizon_and_advisor():
    out = study(_closes(700), splits=(("all", "2015-01-01", None),), horizons=(5,))
    assert set(out["all"]["5_sessions"]) == {"trend", "volatility", "momentum"}


def test_the_volatility_brake_is_scored_warned_versus_the_rest():
    closes = _closes(800, tuple(f"S{i}.NS" for i in range(6)))
    out = study(closes, splits=(("all", "2015-01-01", None),), horizons=(20,))["all"]["20_sessions"]
    brake = out["volatility"]
    assert brake["opinions"] > 0 and "disliked_future_volatility" in brake
