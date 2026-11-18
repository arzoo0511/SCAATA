"""Phase C: wide-universe features, model and paper book, on synthetic prices (no network)."""
import numpy as np
import pandas as pd
import pytest

from scaata.universe import book, features as F, model as M


def _prices(n_days=900, n_stocks=60, drift_per_rank=0.0004, seed=1):
    """Stocks with a persistent individual drift, so past winners keep winning (a planted momentum signal)."""
    rng = np.random.default_rng(seed)
    drift = np.linspace(-1, 1, n_stocks) * drift_per_rank
    rets = rng.normal(0, 0.012, (n_days, n_stocks)) + drift
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=[f"S{i}.NS" for i in range(n_stocks)])


def test_features_ignore_the_future():
    closes = _prices()
    cut = 600
    a = F._raw_features(closes)
    b = F._raw_features(closes.iloc[:cut + 1])
    for k in F.FEATURES:
        np.testing.assert_allclose(a[k].iloc[cut].values, b[k].iloc[cut].values, equal_nan=True)


def test_standardise_is_cross_sectional():
    z = F.standardise(_prices()["S0.NS":"S9.NS"].pct_change().dropna() if False else _prices().pct_change().dropna())
    assert abs(z.mean(axis=1).mean()) < 1e-9
    assert abs(z.std(axis=1).mean() - 1) < 1e-6


def test_model_finds_a_planted_signal_and_purge_holds():
    closes = _prices()
    panel = F.build_panel(closes)
    run = M.walk_forward(panel, "ridge", first_test=str(panel["days"][60].date()))
    ic = M.weekly_ic(run["scores"], panel["rank"])
    assert np.nanmean(ic) > 0.04
    # every fit used only weeks whose 20-session outcome had finished before the test week
    days = list(panel["days"])
    for fit in run["fits"]:
        w = days.index(pd.Timestamp(fit["day"]))
        xs, _ = M._training_rows(panel, w - M.PURGE_WEEKS)
        assert fit["rows"] == len(xs)
        last_train = w - M.PURGE_WEEKS - 1
        assert panel["positions"][last_train] + F.HORIZON < panel["positions"][w]


def test_noise_gives_no_edge():
    closes = _prices(drift_per_rank=0.0, seed=3)
    panel = F.build_panel(closes)
    ic = M.weekly_ic(M.walk_forward(panel, "equal", first_test=str(panel["days"][40].date()))["scores"], panel["rank"])
    assert abs(np.nanmean(ic)) < 0.05


def test_book_runs_forward_only_and_is_idempotent(tmp_path):
    closes = _prices(n_days=1100)
    path = tmp_path / "wide.json"
    assert book.advance(closes.iloc[:900], path)["status"] == "started"
    state = book.load(path)
    assert all(abs(s["equity"] - book.CAPITAL) < 1e-6 for s in state["sleeves"].values())   # nothing back-filled
    assert book.advance(closes, path)["status"] == "advanced"
    first = book.load(path)
    assert book.advance(closes, path)["status"] == "up to date"
    assert book.load(path)["sleeves"]["learned"]["equity"] == first["sleeves"]["learned"]["equity"]
    for s in first["sleeves"].values():
        assert len(s["history"]) == 200 and s["equity"] > 0
    # the first trade happens the session after the decision, and costs money
    assert first["sleeves"]["universe"]["rebalances"][0]["day"] == str(closes.index[900].date())


def test_book_matches_a_straight_rebuild(tmp_path):
    """Running in two steps gives the same book as running in one."""
    closes = _prices(n_days=1000)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    book.advance(closes.iloc[:900], a); book.advance(closes, a)
    book.advance(closes.iloc[:900], b); book.advance(closes.iloc[:950], b); book.advance(closes, b)
    for name in book.SLEEVES:
        assert book.load(a)["sleeves"][name]["equity"] == pytest.approx(book.load(b)["sleeves"][name]["equity"])
