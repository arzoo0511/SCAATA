"""Phase D: interest-rate sensitivities, on synthetic prices (no network)."""
import numpy as np
import pandas as pd

from scaata.cognition import beliefs
from scaata.macroverse import rates


def _world(n=1500, beta=-0.08, seed=2):
    """One stock whose weekly return falls 8% per +1 point of yield; the gilt price moves opposite to yield."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    dy = rng.normal(0, 0.02, n)                         # daily yield change, in points
    yld = 6 + np.cumsum(dy)
    gilt = 100 * np.exp(-(rates.DURATION / 100) * (yld - yld[0]))
    stock = 100 * np.exp(np.cumsum(beta * dy + rng.normal(0, 0.004, n)))
    closes = pd.DataFrame({"AAA.NS": stock, "BBB.NS": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))}, index=idx)
    market = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.008, n))), index=idx)
    return closes, pd.DataFrame({"us10y": yld, "gilt": gilt}, index=idx), market


def test_gilt_price_turns_back_into_the_yield_change():
    closes, rt, _ = _world()
    got = rates.yield_changes(rt, closes.index, "india", 1)
    true = rt["us10y"].diff().dropna()
    assert np.corrcoef(got.values, true.loc[got.index].values)[0, 1] > 0.99
    assert abs(got.std() / true.std() - 1) < 0.05


def test_estimate_recovers_the_planted_link_and_a_pure_noise_stock_is_unclear():
    closes, rt, market = _world()
    s = rates.estimate(closes, rt, market, "india")
    a, b = s["table"]
    assert a["reliable"] and -0.11 < a["beta"] < -0.05 and a["low"] < a["beta"] < a["high"]
    assert not b["reliable"]


def test_rate_study_becomes_beliefs_with_status():
    closes, rt, market = _world()
    s = rates.estimate(closes, rt, market, "india")
    study = {"studies": {"india": {k: v for k, v in s.items() if k not in ("point", "draws")}}}
    out = beliefs._rates(study)
    assert len(out) == 2 and out[0]["kind"] == "relationship" and out[0]["status"] in ("established", "leaning")
    assert out[0]["detail"]["factor"] == "rates_india" and "against" in out[0]["statement"]
    assert out[1]["status"] == "cannot tell"
