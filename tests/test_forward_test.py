"""Tests for the Phase 8 live paper-trading forward test. Uses a fake
policy (not a real trained RecurrentPPO, to keep this fast) to verify the
decision/log-append plumbing: the log is source-tagged, persists across
calls, and the mock path never contacts a real broker.
"""
import numpy as np
import pandas as pd
import pytest

from scaata.config import FEATURE_COLUMNS
from scaata.live import forward_test
from scaata.rl.env import BUY, HOLD


def _synthetic_bars(ticker="TESTFWD", n=90):
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n), "Ticker": ticker,
    }, index=dates)
    df.index.name = "Date"
    return df


class _FakePolicy:
    """Stands in for a trained RecurrentPPO so this test doesn't pay for a
    real 100k-timestep training run just to check the logging plumbing."""

    def __init__(self, fixed_action=BUY):
        self.fixed_action = fixed_action

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return self.fixed_action, ("fake-lstm-state",)


@pytest.fixture(autouse=True)
def _redirect_forward_test_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(forward_test, "FORWARD_TEST_DIR", tmp_path)


@pytest.fixture(autouse=True)
def _no_broker_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)


def test_run_daily_decision_runs_end_to_end_in_mock_mode(monkeypatch):
    monkeypatch.setattr(forward_test, "get_latest_daily_bars", lambda *a, **k: (_synthetic_bars(), "mock"))

    row = forward_test.run_daily_decision("TESTFWD", _FakePolicy(fixed_action=BUY), FEATURE_COLUMNS, qty=1.0)

    assert row["overall_source"] == "mock"
    assert row["action"] == "buy"
    assert row["ticker"] == "TESTFWD"


def test_run_daily_decision_appends_to_a_persistent_log(monkeypatch):
    monkeypatch.setattr(forward_test, "get_latest_daily_bars", lambda *a, **k: (_synthetic_bars(), "mock"))

    forward_test.run_daily_decision("TESTFWD", _FakePolicy(fixed_action=HOLD), FEATURE_COLUMNS, qty=1.0)
    forward_test.run_daily_decision("TESTFWD", _FakePolicy(fixed_action=BUY), FEATURE_COLUMNS, qty=1.0)

    log = forward_test.read_decision_log("TESTFWD")
    assert len(log) == 2
    assert list(log["action"]) == ["hold", "buy"]


def test_lstm_state_persists_across_calls(monkeypatch):
    monkeypatch.setattr(forward_test, "get_latest_daily_bars", lambda *a, **k: (_synthetic_bars(), "mock"))

    assert not forward_test._state_path("TESTFWD").exists()
    forward_test.run_daily_decision("TESTFWD", _FakePolicy(), FEATURE_COLUMNS, qty=1.0)
    assert forward_test._state_path("TESTFWD").exists()

    state, episode_start = forward_test._load_lstm_state("TESTFWD")
    assert state == ("fake-lstm-state",)
    assert episode_start[0] == False


def test_repeated_live_calls_same_day_do_not_place_a_second_order(monkeypatch):
    """The bug this guards against: running the notebook twice in one day
    submitted two independent real sell orders, silently doubling the
    position instead of representing one day's decision."""
    monkeypatch.setattr(forward_test, "get_latest_daily_bars", lambda *a, **k: (_synthetic_bars(), "live"))
    monkeypatch.setattr(forward_test, "get_account_snapshot", lambda *a, **k: ({"equity": 100000.0}, "live"))

    submit_calls = []

    def _fake_submit(*a, **k):
        submit_calls.append((a, k))
        return {"order_id": f"order-{len(submit_calls)}", "status": "accepted"}, "live"

    monkeypatch.setattr(forward_test, "submit_paper_order", _fake_submit)

    first = forward_test.run_daily_decision("TESTFWD", _FakePolicy(fixed_action=BUY), FEATURE_COLUMNS, qty=1.0)
    second = forward_test.run_daily_decision("TESTFWD", _FakePolicy(fixed_action=BUY), FEATURE_COLUMNS, qty=1.0)

    assert len(submit_calls) == 1, "a second same-day call must not submit a second real order"
    assert second == first
    assert len(forward_test.read_decision_log("TESTFWD")) == 1


def test_mock_mode_is_never_rate_limited_to_once_a_day(monkeypatch):
    monkeypatch.setattr(forward_test, "get_latest_daily_bars", lambda *a, **k: (_synthetic_bars(), "mock"))

    forward_test.run_daily_decision("TESTFWD", _FakePolicy(fixed_action=BUY), FEATURE_COLUMNS, qty=1.0)
    forward_test.run_daily_decision("TESTFWD", _FakePolicy(fixed_action=BUY), FEATURE_COLUMNS, qty=1.0)
    forward_test.run_daily_decision("TESTFWD", _FakePolicy(fixed_action=BUY), FEATURE_COLUMNS, qty=1.0)

    assert len(forward_test.read_decision_log("TESTFWD")) == 3
