"""Tests for the notify-only Alpaca daily signal (Phase 16b) — the
critical property being tested throughout is that this module never
submits an order (no `submit_paper_order` import even exists here), only
ever returns/logs a recommendation. Also covers position-awareness: a
SELL signal while flat, or a BUY signal while already holding, isn't
actionable -- RobustTradingEnv's own training only ever treats BUY-while-
flat and SELL-while-holding as real transitions, so the live signal must
say the same thing, not imply a trade the policy never actually learned.
"""
import math

import numpy as np
import pandas as pd

import scaata.live.daily_signal as daily_signal_module
from scaata.config import FEATURE_COLUMNS, SIGNAL_DISCLAIMER, SIGNAL_PRICE_GAP_BUFFER
from scaata.live.daily_signal import (
    compute_daily_signal,
    compute_impersonal_signal,
    format_signal_message,
    impersonal_view,
    preview_daily_signal,
    read_signal_log,
)


def _expected_qty(cash, capital_fraction, current_price):
    """Sizing is deliberately against a gap-cushioned price, not the raw
    stale close -- see SIGNAL_PRICE_GAP_BUFFER."""
    return math.floor(cash * capital_fraction / (current_price * (1 + SIGNAL_PRICE_GAP_BUFFER)))


def _make_fake_bars(n=90, price=150.0, ticker="TEST"):
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.full(n, price) + np.linspace(0, 5, n)
    return pd.DataFrame({
        "Open": close * 0.999, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": np.full(n, 2_000_000), "Ticker": ticker,
    }, index=dates)


class _FixedActionModel:
    def __init__(self, action: int):
        self.action = action

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return self.action, {"dummy": "state"}


def _stats_fitted_on(bars, feature_columns=FEATURE_COLUMNS):
    from scaata.features.normalize import fit_normalizer
    from scaata.features.technical import add_features

    mean, std = fit_normalizer(add_features(bars.reset_index()), feature_columns)
    return mean, std, {"quality": "exact"}


def _patch_broker(
    monkeypatch, tmp_path, bars, account,
    data_source="mock", account_source="mock",
    position_qty=0.0, position_source="mock",
    norm_stats="fit",
):
    """`norm_stats="fit"` (default) supplies training stats fitted on `bars`
    so tests exercise a normally-deployed policy; pass None to simulate a
    policy with no saved stats, or an explicit (mean, std, provenance)."""
    if isinstance(norm_stats, str) and norm_stats == "fit":
        norm_stats = _stats_fitted_on(bars)
    monkeypatch.setattr(daily_signal_module, "get_latest_daily_bars", lambda ticker, lookback_days=90: (bars, data_source))
    monkeypatch.setattr(daily_signal_module, "get_account_snapshot", lambda: (account, account_source))
    monkeypatch.setattr(daily_signal_module, "get_position_qty", lambda ticker: (position_qty, position_source))
    monkeypatch.setattr(daily_signal_module, "_load_norm_stats", lambda ticker, feature_columns: norm_stats)
    monkeypatch.setattr(daily_signal_module, "SIGNAL_DIR", tmp_path)


class _CapturingModel:
    def __init__(self, action: int):
        self.action = action
        self.observations = []

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        self.observations.append(np.array(obs))
        return self.action, {"dummy": "state"}


def test_policy_sees_features_scaled_with_its_training_stats_not_the_live_window(tmp_path, monkeypatch):
    """Regression test for the live skew: the live path used to re-fit
    z-scores on the ~41 rows it had just fetched. The observation must be
    the latest row scaled with the stats the policy was trained on."""
    from scaata.features.technical import add_features
    from scaata.rl.env import HOLD

    bars = _make_fake_bars(price=100.0)
    training_bars = _make_fake_bars(price=400.0)  # a different distribution than the live window
    mean, std, provenance = _stats_fitted_on(training_bars)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, norm_stats=(mean, std, provenance))
    model = _CapturingModel(HOLD)

    row = compute_daily_signal("TEST", model, FEATURE_COLUMNS)

    latest = add_features(bars.reset_index()).iloc[-1][FEATURE_COLUMNS].astype(float)
    expected = ((latest - mean) / std).values.astype(np.float32)
    np.testing.assert_allclose(model.observations[0], expected, rtol=1e-5)
    assert row["norm_source"] == "exact"


def test_missing_training_stats_withholds_the_signal(tmp_path, monkeypatch):
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, norm_stats=None)
    model = _CapturingModel(BUY)

    row = compute_daily_signal("TEST", model, FEATURE_COLUMNS)

    assert model.observations == []  # the policy is never run on mis-scaled inputs
    assert row["norm_source"] == "missing"
    assert row["action"] == "HOLD"
    assert row["actionable"] is False
    assert row["fallback_action"] is None
    assert not (tmp_path / "lstm_state_TEST.pkl").exists()
    assert "withheld" in format_signal_message(row)


def test_preview_with_missing_training_stats_is_not_actionable(tmp_path, monkeypatch):
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, norm_stats=None)

    preview = preview_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS)

    assert preview["actionable"] is False


def test_buy_signal_while_flat_is_actionable_with_qty_and_stop_loss(tmp_path, monkeypatch):
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    row = compute_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS)

    assert row["action"] == "BUY"
    assert row["actionable"] is True
    assert row["suggested_qty"] == _expected_qty(10_000.0, 1.0, row["current_price"])
    assert row["stop_loss_price"] < row["current_price"]


def test_buy_signal_while_already_holding_is_not_actionable(tmp_path, monkeypatch):
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 0.0, "buying_power": 0.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=25.0)

    row = compute_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS)

    assert row["action"] == "BUY"
    assert row["actionable"] is False
    assert row["suggested_qty"] is None
    assert row["stop_loss_price"] is None


def test_hold_signal_has_no_qty_or_stop_loss(tmp_path, monkeypatch):
    from scaata.rl.env import HOLD

    bars = _make_fake_bars()
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account)

    row = compute_daily_signal("TEST", _FixedActionModel(HOLD), FEATURE_COLUMNS)

    assert row["action"] == "HOLD"
    assert row["suggested_qty"] is None
    assert row["stop_loss_price"] is None


def test_sell_signal_while_holding_is_actionable_and_suggests_closing_full_position(tmp_path, monkeypatch):
    from scaata.rl.env import SELL

    bars = _make_fake_bars()
    account = {"equity": 10_000.0, "cash": 0.0, "buying_power": 0.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=30.0)

    row = compute_daily_signal("TEST", _FixedActionModel(SELL), FEATURE_COLUMNS)

    assert row["action"] == "SELL"
    assert row["actionable"] is True
    assert row["suggested_qty"] == 30.0


def test_sell_signal_while_flat_is_not_actionable(tmp_path, monkeypatch):
    from scaata.rl.env import SELL

    bars = _make_fake_bars()
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    row = compute_daily_signal("TEST", _FixedActionModel(SELL), FEATURE_COLUMNS)

    assert row["action"] == "SELL"
    assert row["actionable"] is False
    assert row["suggested_qty"] is None


def test_compute_daily_signal_never_touches_submit_paper_order():
    """No order-submission function is even importable from this module --
    a structural guarantee, not just a behavioral one, that this path can
    never place a trade."""
    assert not hasattr(daily_signal_module, "submit_paper_order")


def test_signal_is_logged_to_a_permanent_csv(tmp_path, monkeypatch):
    from scaata.rl.env import HOLD

    bars = _make_fake_bars()
    account = {"equity": 5_000.0, "cash": 5_000.0, "buying_power": 5_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account)

    compute_daily_signal("TEST", _FixedActionModel(HOLD), FEATURE_COLUMNS)
    log = read_signal_log("TEST")

    assert len(log) == 1
    assert log.iloc[0]["ticker"] == "TEST"
    assert log.iloc[0]["action"] == "HOLD"


def test_signal_log_accumulates_across_multiple_calls(tmp_path, monkeypatch):
    from scaata.rl.env import HOLD

    bars = _make_fake_bars()
    account = {"equity": 5_000.0, "cash": 5_000.0, "buying_power": 5_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account)

    compute_daily_signal("TEST", _FixedActionModel(HOLD), FEATURE_COLUMNS)
    compute_daily_signal("TEST", _FixedActionModel(HOLD), FEATURE_COLUMNS)
    log = read_signal_log("TEST")

    assert len(log) == 2


def test_overall_source_is_mock_when_any_source_is_mock(tmp_path, monkeypatch):
    from scaata.rl.env import HOLD

    bars = _make_fake_bars()
    account = {"equity": 5_000.0, "cash": 5_000.0, "buying_power": 5_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, data_source="live", account_source="live", position_source="mock")

    row = compute_daily_signal("TEST", _FixedActionModel(HOLD), FEATURE_COLUMNS)
    assert row["overall_source"] == "mock"


def test_overall_source_is_live_when_all_sources_are_live(tmp_path, monkeypatch):
    from scaata.rl.env import HOLD

    bars = _make_fake_bars()
    account = {"equity": 5_000.0, "cash": 5_000.0, "buying_power": 5_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, data_source="live", account_source="live", position_source="live")

    row = compute_daily_signal("TEST", _FixedActionModel(HOLD), FEATURE_COLUMNS)
    assert row["overall_source"] == "live"


def test_format_signal_message_flags_mock_data():
    row = {
        "ticker": "TEST", "action": "HOLD", "current_price": 100.0, "position_qty": 0.0,
        "actionable": False, "suggested_qty": None, "stop_loss_price": None,
        "account_cash": 1000.0, "account_equity": 1000.0, "overall_source": "mock",
    }
    message = format_signal_message(row)
    assert "MOCK DATA" in message


def test_format_signal_message_includes_qty_and_stop_loss_for_actionable_buy():
    row = {
        "ticker": "AAPL", "action": "BUY", "current_price": 200.0, "position_qty": 0.0,
        "actionable": True, "suggested_qty": 50, "stop_loss_price": 196.0,
        "account_cash": 0.0, "account_equity": 10000.0, "overall_source": "live",
    }
    message = format_signal_message(row)
    assert "50 shares" in message
    assert "196.00" in message


def test_schema_change_archives_old_log_instead_of_corrupting_it(tmp_path, monkeypatch):
    """Regression test for a real bug caught live: changing the row schema
    (adding position_qty/actionable/position_source) while an old-schema
    log file already existed produced a column-count-mismatched CSV that
    crashed on the next read. A schema change must archive the old file,
    not silently append a wider row to it."""
    from scaata.rl.env import HOLD

    bars = _make_fake_bars()
    account = {"equity": 5_000.0, "cash": 5_000.0, "buying_power": 5_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account)

    old_log_path = tmp_path / "signal_log_TEST.csv"
    old_log_path.write_text("timestamp_utc,ticker,action,current_price\n2020-01-01,TEST,HOLD,100.0\n")

    compute_daily_signal("TEST", _FixedActionModel(HOLD), FEATURE_COLUMNS)

    # The old-schema file was archived, not appended to with mismatched columns.
    archived = list(tmp_path.glob("signal_log_TEST.schema_changed_*.csv"))
    assert len(archived) == 1
    assert "100.0" in archived[0].read_text()

    # The new file is readable and has exactly one (new-schema) row.
    log = read_signal_log("TEST")
    assert len(log) == 1
    assert log.iloc[0]["ticker"] == "TEST"


def test_fallback_suggested_when_hold_and_flat(tmp_path, monkeypatch):
    """The core new behavior: when the policy has nothing actionable to say
    (HOLD) and the ticker is flat, surface the buy-and-hold-implied action
    instead of leaving the user with no information -- this session's own
    research found buy-and-hold beats a collapsed policy on tickers where
    it never trades."""
    from scaata.rl.env import HOLD

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    row = compute_daily_signal("TEST", _FixedActionModel(HOLD), FEATURE_COLUMNS)

    assert row["fallback_action"] == "BUY"
    assert row["fallback_qty"] == _expected_qty(10_000.0, 1.0, row["current_price"])


def test_no_fallback_when_hold_but_already_holding(tmp_path, monkeypatch):
    from scaata.rl.env import HOLD

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 0.0, "buying_power": 0.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=25.0)

    row = compute_daily_signal("TEST", _FixedActionModel(HOLD), FEATURE_COLUMNS)

    assert row["fallback_action"] is None
    assert row["fallback_qty"] is None


def test_no_fallback_when_dsr_signal_is_actionable_buy(tmp_path, monkeypatch):
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    row = compute_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS)

    assert row["actionable"] is True
    assert row["fallback_action"] is None


def test_no_fallback_when_unactionable_sell_but_already_holding_something_else(tmp_path, monkeypatch):
    """An unactionable SELL only happens while flat (position_qty<=0 by
    definition of is_flat), so this exercises the SELL branch of the
    not-actionable case specifically, confirming the fallback still fires
    there too (not just for HOLD)."""
    from scaata.rl.env import SELL

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    row = compute_daily_signal("TEST", _FixedActionModel(SELL), FEATURE_COLUMNS)

    assert row["actionable"] is False
    assert row["fallback_action"] == "BUY"


def test_format_signal_message_includes_fallback_when_present():
    row = {
        "ticker": "MSFT", "action": "HOLD", "current_price": 100.0, "position_qty": 0.0,
        "actionable": False, "suggested_qty": None, "stop_loss_price": None,
        "fallback_action": "BUY", "fallback_qty": 100,
        "account_cash": 10000.0, "account_equity": 10000.0, "overall_source": "live",
    }
    message = format_signal_message(row)
    assert "Fallback" in message
    assert "buy-and-hold" in message.lower()
    assert "100 shares" in message


def test_format_signal_message_omits_fallback_when_none():
    row = {
        "ticker": "AAPL", "action": "BUY", "current_price": 200.0, "position_qty": 0.0,
        "actionable": True, "suggested_qty": 50, "stop_loss_price": 196.0,
        "fallback_action": None, "fallback_qty": None,
        "account_cash": 0.0, "account_equity": 10000.0, "overall_source": "live",
    }
    message = format_signal_message(row)
    assert "Fallback" not in message


# --- capital_fraction / portfolio-coherent sizing (regression tests for a
# real bug caught live: every actionable BUY was independently sized at
# ~100% of cash, so a day with several actionable BUYs at once suggested
# needing far more capital than the account actually has) ---

def test_capital_fraction_scales_suggested_qty(tmp_path, monkeypatch):
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    full = compute_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS, capital_fraction=1.0)
    split = compute_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS, capital_fraction=0.25)

    assert split["suggested_qty"] == _expected_qty(10_000.0, 0.25, split["current_price"])
    assert split["suggested_qty"] < full["suggested_qty"]
    assert split["capital_fraction"] == 0.25


def test_capital_fraction_scales_fallback_qty_too(tmp_path, monkeypatch):
    from scaata.rl.env import HOLD

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    row = compute_daily_signal("TEST", _FixedActionModel(HOLD), FEATURE_COLUMNS, capital_fraction=0.5)

    assert row["fallback_qty"] == _expected_qty(10_000.0, 0.5, row["current_price"])


def test_default_capital_fraction_claims_the_whole_allocation(tmp_path, monkeypatch):
    """A caller that doesn't pass capital_fraction (e.g. the dashboard's
    single-ticker 'Refresh now' button, which has no visibility into other
    tickers' signals) claims the full cash allocation -- still reduced by
    SIGNAL_PRICE_GAP_BUFFER, which applies to every sizing path."""
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    row = compute_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS)

    assert row["capital_fraction"] == 1.0
    assert row["suggested_qty"] == _expected_qty(10_000.0, 1.0, row["current_price"])


def test_preview_daily_signal_does_not_save_state_or_log(tmp_path, monkeypatch):
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    preview = preview_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS)

    assert preview == {"ticker": "TEST", "action": "BUY", "actionable": True}
    assert not (tmp_path / "lstm_state_TEST.pkl").exists()
    assert read_signal_log("TEST").empty  # nothing was logged


def test_preview_daily_signal_matches_actionability_of_real_call(tmp_path, monkeypatch):
    """Preview and the real call must agree on actionability -- if they
    ever disagreed, the actionable-BUY count used for sizing would be
    wrong relative to what actually gets logged/suggested."""
    from scaata.rl.env import SELL

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 0.0, "buying_power": 0.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=30.0)

    preview = preview_daily_signal("TEST", _FixedActionModel(SELL), FEATURE_COLUMNS)
    real = compute_daily_signal("TEST", _FixedActionModel(SELL), FEATURE_COLUMNS)

    assert preview["actionable"] == real["actionable"] == True
    assert preview["action"] == real["action"] == "SELL"


def test_format_signal_message_shows_full_cash_wording_at_default_fraction():
    row = {
        "ticker": "AAPL", "action": "BUY", "current_price": 200.0, "position_qty": 0.0,
        "actionable": True, "suggested_qty": 50, "capital_fraction": 1.0, "stop_loss_price": 196.0,
        "account_cash": 0.0, "account_equity": 10000.0, "overall_source": "live",
    }
    message = format_signal_message(row)
    assert "your available cash" in message
    # The quoted price is the prior close, not the fill price -- the
    # message must not let a reader assume otherwise.
    assert "buffer" in message and "not your fill price" in message


def test_format_signal_message_shows_split_wording_when_fraction_below_one():
    row = {
        "ticker": "AAPL", "action": "BUY", "current_price": 200.0, "position_qty": 0.0,
        "actionable": True, "suggested_qty": 12, "capital_fraction": 0.25, "stop_loss_price": 196.0,
        "account_cash": 0.0, "account_equity": 10000.0, "overall_source": "live",
    }
    message = format_signal_message(row)
    assert "25%" in message
    assert "split across today's other actionable BUYs" in message


def test_format_signal_message_says_no_action_needed_for_unactionable_sell():
    row = {
        "ticker": "AAPL", "action": "SELL", "current_price": 200.0, "position_qty": 0.0,
        "actionable": False, "suggested_qty": None, "stop_loss_price": None,
        "account_cash": 10000.0, "account_equity": 10000.0, "overall_source": "live",
    }
    message = format_signal_message(row)
    assert "no open position" in message.lower()


def test_suggested_order_survives_an_overnight_gap_up(tmp_path, monkeypatch):
    """Regression test for a real live case: GOOGL was sized at 104 shares
    x $319.73 = $33,251 against a $33,333 allocation (1/3 of $100k), with
    zero headroom. `current_price` is always the PREVIOUS session's close
    (the signal job runs pre-market), so a routine overnight gap up meant
    the suggested order no longer fit its allocation -- and with several
    simultaneous BUYs gapping together, the basket could exceed the
    account. Sizing must leave room for an ordinary gap.
    """
    from scaata.rl.env import BUY

    stale_close = 319.73
    bars = _make_fake_bars(price=stale_close)
    account = {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 100_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    row = compute_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS, capital_fraction=1 / 3)

    allocation = 100_000.0 / 3
    gapped_price = row["current_price"] * 1.02  # a modest 2% overnight gap up
    assert row["suggested_qty"] * gapped_price <= allocation, (
        "order must still fit its cash allocation after an ordinary gap up"
    )


def test_gap_buffer_never_suggests_more_than_unbuffered_sizing(tmp_path, monkeypatch):
    """The buffer must only ever err toward FEWER shares -- suggesting more
    than the raw price allows would defeat its whole purpose."""
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    row = compute_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS)

    unbuffered = math.floor(10_000.0 / row["current_price"])
    assert row["suggested_qty"] <= unbuffered
    assert row["suggested_qty"] > 0  # but still a usable position, not zeroed out


def test_negative_cash_never_suggests_a_negative_share_count(tmp_path, monkeypatch):
    """Regression test for a real live case: an earlier over-sized order
    pushed the account to -$1,954 cash, and every BUY signal then rendered
    as "BUY -5 shares" -- math.floor on a negative numerator rounds AWAY
    from zero, so the deeper into margin the account went, the larger the
    nonsense negative order became.
    """
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=197.59)
    account = {"equity": 100_000.0, "cash": -1_954.48, "buying_power": 0.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    row = compute_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS, capital_fraction=0.5)

    assert row["suggested_qty"] is None or row["suggested_qty"] >= 0
    assert row["fallback_qty"] is None or row["fallback_qty"] >= 0
    assert row["actionable"] is False  # a buy you cannot fund is not actionable
    assert row["unaffordable"] is True


def test_unaffordable_buy_is_explained_as_a_cash_problem_not_a_position_one(tmp_path, monkeypatch):
    """The message must not claim "already holding a position" when the
    real reason is no cash -- that would be simply false, and the user
    would have no idea the policy actually wanted to buy."""
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=197.59)
    account = {"equity": 100_000.0, "cash": -1_954.48, "buying_power": 0.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    row = compute_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS)
    message = format_signal_message(row)

    assert "no available cash" in message
    assert "already holding" not in message
    assert "-" not in message.split("shares")[0].split(":")[-1]  # no negative qty anywhere


def test_zero_cash_produces_no_fallback_buy(tmp_path, monkeypatch):
    """With no cash there is no buy-and-hold fallback to suggest either."""
    from scaata.rl.env import HOLD

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 50_000.0, "cash": 0.0, "buying_power": 0.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    row = compute_daily_signal("TEST", _FixedActionModel(HOLD), FEATURE_COLUMNS)

    assert row["fallback_action"] is None
    assert row["fallback_qty"] is None


def test_preview_does_not_count_an_unfundable_buy_as_actionable(tmp_path, monkeypatch):
    """The actionable count drives how capital is split. Counting a BUY the
    account can't fund inflates that count and under-sizes the BUYs that
    ARE fundable -- observed live with negative cash, where three "actionable"
    BUYs split capital 33% each and then all three were demoted."""
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=197.59)
    account = {"equity": 100_000.0, "cash": -1_954.48, "buying_power": 0.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    preview = preview_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS)

    assert preview["action"] == "BUY"
    assert preview["actionable"] is False


def test_preview_still_counts_a_fundable_buy_as_actionable(tmp_path, monkeypatch):
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    preview = preview_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS)

    assert preview["actionable"] is True


def test_preview_and_real_call_agree_on_unfundable_buys(tmp_path, monkeypatch):
    """Preview must not disagree with the real call about actionability --
    that mismatch is exactly what produced the inflated split."""
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=197.59)
    account = {"equity": 100_000.0, "cash": -1_954.48, "buying_power": 0.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    preview = preview_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS)
    real = compute_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS)

    assert preview["actionable"] == real["actionable"]


# --- impersonal_view / compute_impersonal_signal (Phase 19: the
# client-product signal, decoupled from any one account's cash/position) ---

def test_impersonal_view_has_no_account_or_sizing_fields(tmp_path, monkeypatch):
    """The whole point of this function: a paying subscriber must never be
    able to see this account's real cash, position, or a share count sized
    against money that isn't theirs."""
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    row = compute_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS)
    view = impersonal_view(row)

    for leaked_field in ("account_cash", "account_equity", "position_qty", "suggested_qty", "capital_fraction"):
        assert leaked_field not in view


def test_impersonal_view_keeps_the_actual_signal(tmp_path, monkeypatch):
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    row = compute_daily_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS)
    view = impersonal_view(row)

    assert view["ticker"] == "TEST"
    assert view["action"] == "BUY"
    assert view["current_price"] == row["current_price"]
    assert view["as_of_utc"] == row["timestamp_utc"]


def test_impersonal_view_includes_the_compliance_disclaimer(tmp_path, monkeypatch):
    from scaata.rl.env import HOLD

    bars = _make_fake_bars()
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account)

    row = compute_daily_signal("TEST", _FixedActionModel(HOLD), FEATURE_COLUMNS)
    view = impersonal_view(row)

    assert view["disclaimer"] == SIGNAL_DISCLAIMER


def test_compute_impersonal_signal_never_saves_state_or_logs(tmp_path, monkeypatch):
    """Same no-side-effects contract as preview_daily_signal -- calling
    this must never advance the recurrent LSTM state a canonical
    compute_daily_signal call would rely on, and must never write to the
    signal log (only the one canonical daily job does that)."""
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    view = compute_impersonal_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS)

    assert view["ticker"] == "TEST"
    assert view["action"] == "BUY"
    assert not (tmp_path / "lstm_state_TEST.pkl").exists()
    assert read_signal_log("TEST").empty


def test_compute_impersonal_signal_has_no_account_fields(tmp_path, monkeypatch):
    from scaata.rl.env import BUY

    bars = _make_fake_bars(price=100.0)
    account = {"equity": 10_000.0, "cash": 10_000.0, "buying_power": 10_000.0}
    _patch_broker(monkeypatch, tmp_path, bars, account, position_qty=0.0)

    view = compute_impersonal_signal("TEST", _FixedActionModel(BUY), FEATURE_COLUMNS)

    for leaked_field in ("account_cash", "account_equity", "position_qty", "suggested_qty"):
        assert leaked_field not in view
