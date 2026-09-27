"""Tests for the standalone (Task-Scheduler-launched) daily signal runner:
must skip untrained tickers, one ticker's failure must not stop the rest
of the run, and -- the real bug this file's newer tests guard against --
capital must be split fairly across every actionable BUY found today, not
independently sized at 100% of cash per ticker.
"""
import scaata.live.run_all_daily_signals as runner_module


def test_main_skips_untrained_tickers(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner_module, "FORWARD_TEST_DIR", tmp_path)
    monkeypatch.setattr(runner_module, "ALL_TICKERS", ["NOPE1", "NOPE2"])

    runner_module.main()

    out = capsys.readouterr().out
    assert "NOPE1: no trained model yet, skipping." in out
    assert "NOPE2: no trained model yet, skipping." in out


def test_main_continues_after_one_ticker_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner_module, "FORWARD_TEST_DIR", tmp_path)
    monkeypatch.setattr(runner_module, "ALL_TICKERS", ["BAD", "GOOD"])
    (tmp_path / "policy_BAD.zip").write_text("not a real model")
    (tmp_path / "policy_GOOD.zip").write_text("not a real model either")

    calls = []

    class _FakeRecurrentPPO:
        @staticmethod
        def load(path):
            if "BAD" in path:
                raise RuntimeError("corrupt model file")
            return "fake-good-model"

    def _fake_preview(ticker, model):
        return {"ticker": ticker, "action": "HOLD", "actionable": False}

    def _fake_compute(ticker, model, capital_fraction=1.0):
        calls.append((ticker, capital_fraction))
        return {"ticker": ticker, "action": "HOLD"}

    monkeypatch.setattr(runner_module, "RecurrentPPO", _FakeRecurrentPPO)
    monkeypatch.setattr(runner_module, "preview_daily_signal", _fake_preview)
    monkeypatch.setattr(runner_module, "compute_daily_signal", _fake_compute)
    monkeypatch.setattr(runner_module, "format_signal_message", lambda row: f"{row['ticker']}: {row['action']}")

    runner_module.main()

    # GOOD still got processed even though BAD raised first (BAD never
    # made it into `models` at all, since it failed at load time).
    assert [c[0] for c in calls] == ["GOOD"]
    err = capsys.readouterr().err
    assert "BAD: FAILED" in err


def _fake_preview_factory(actionable_buy_tickers):
    def _preview(ticker, model):
        if ticker in actionable_buy_tickers:
            return {"ticker": ticker, "action": "BUY", "actionable": True}
        return {"ticker": ticker, "action": "HOLD", "actionable": False}
    return _preview


def test_main_splits_capital_across_multiple_actionable_buys(tmp_path, monkeypatch, capsys):
    """Regression test for a real bug caught live: GOOGL, AMZN, NVDA, and
    META were all actionable BUYs the same day, and each was independently
    sized at ~100% of available cash -- acting on more than one would need
    far more capital than the account actually has. With 3 actionable BUYs
    (out of 4 tickers), each real compute call must get a 1/3 capital
    fraction, not 1.0."""
    monkeypatch.setattr(runner_module, "FORWARD_TEST_DIR", tmp_path)
    tickers = ["BUY1", "BUY2", "BUY3", "HOLDER"]
    monkeypatch.setattr(runner_module, "ALL_TICKERS", tickers)
    for t in tickers:
        (tmp_path / f"policy_{t}.zip").write_text("not a real model")

    class _FakeRecurrentPPO:
        @staticmethod
        def load(path):
            return "fake-model"

    calls = []

    monkeypatch.setattr(runner_module, "RecurrentPPO", _FakeRecurrentPPO)
    monkeypatch.setattr(runner_module, "preview_daily_signal", _fake_preview_factory({"BUY1", "BUY2", "BUY3"}))
    monkeypatch.setattr(
        runner_module, "compute_daily_signal",
        lambda ticker, model, capital_fraction=1.0: calls.append((ticker, capital_fraction)) or {"ticker": ticker, "action": "X"},
    )
    monkeypatch.setattr(runner_module, "format_signal_message", lambda row: row["ticker"])

    runner_module.main()

    fractions = dict(calls)
    assert fractions["BUY1"] == fractions["BUY2"] == fractions["BUY3"] == 1 / 3
    # the non-actionable ticker gets the same fraction too (harmless -- it
    # has no suggested_qty to scale), rather than a separate code path.
    assert fractions["HOLDER"] == 1 / 3
    out = capsys.readouterr().out
    assert "3 actionable BUYs today" in out


def test_main_uses_full_capital_when_only_one_actionable_buy(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "FORWARD_TEST_DIR", tmp_path)
    tickers = ["ONLYBUY", "HOLDER"]
    monkeypatch.setattr(runner_module, "ALL_TICKERS", tickers)
    for t in tickers:
        (tmp_path / f"policy_{t}.zip").write_text("not a real model")

    class _FakeRecurrentPPO:
        @staticmethod
        def load(path):
            return "fake-model"

    calls = []
    monkeypatch.setattr(runner_module, "RecurrentPPO", _FakeRecurrentPPO)
    monkeypatch.setattr(runner_module, "preview_daily_signal", _fake_preview_factory({"ONLYBUY"}))
    monkeypatch.setattr(
        runner_module, "compute_daily_signal",
        lambda ticker, model, capital_fraction=1.0: calls.append((ticker, capital_fraction)) or {"ticker": ticker, "action": "X"},
    )
    monkeypatch.setattr(runner_module, "format_signal_message", lambda row: row["ticker"])

    runner_module.main()

    assert dict(calls)["ONLYBUY"] == 1.0


def test_main_uses_full_capital_when_zero_actionable_buys(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "FORWARD_TEST_DIR", tmp_path)
    tickers = ["HOLDER1", "HOLDER2"]
    monkeypatch.setattr(runner_module, "ALL_TICKERS", tickers)
    for t in tickers:
        (tmp_path / f"policy_{t}.zip").write_text("not a real model")

    class _FakeRecurrentPPO:
        @staticmethod
        def load(path):
            return "fake-model"

    calls = []
    monkeypatch.setattr(runner_module, "RecurrentPPO", _FakeRecurrentPPO)
    monkeypatch.setattr(runner_module, "preview_daily_signal", _fake_preview_factory(set()))
    monkeypatch.setattr(
        runner_module, "compute_daily_signal",
        lambda ticker, model, capital_fraction=1.0: calls.append((ticker, capital_fraction)) or {"ticker": ticker, "action": "X"},
    )
    monkeypatch.setattr(runner_module, "format_signal_message", lambda row: row["ticker"])

    runner_module.main()

    # no division by zero, and no artificial 0-sized suggestion either --
    # falls back to the original all-in convention when nothing else is competing.
    assert all(frac == 1.0 for _, frac in calls)


def test_main_continues_when_one_tickers_preview_fails(tmp_path, monkeypatch, capsys):
    """A preview failure for one ticker must not crash the whole run -- it
    just doesn't count toward the day's actionable-BUY total; that
    ticker's real pass below will independently hit (and report) the same
    underlying error."""
    monkeypatch.setattr(runner_module, "FORWARD_TEST_DIR", tmp_path)
    tickers = ["PREVIEWFAILS", "FINE"]
    monkeypatch.setattr(runner_module, "ALL_TICKERS", tickers)
    for t in tickers:
        (tmp_path / f"policy_{t}.zip").write_text("not a real model")

    class _FakeRecurrentPPO:
        @staticmethod
        def load(path):
            return "fake-model"

    def _flaky_preview(ticker, model):
        if ticker == "PREVIEWFAILS":
            raise RuntimeError("transient data error")
        return {"ticker": ticker, "action": "HOLD", "actionable": False}

    calls = []
    monkeypatch.setattr(runner_module, "RecurrentPPO", _FakeRecurrentPPO)
    monkeypatch.setattr(runner_module, "preview_daily_signal", _flaky_preview)
    monkeypatch.setattr(
        runner_module, "compute_daily_signal",
        lambda ticker, model, capital_fraction=1.0: calls.append(ticker) or {"ticker": ticker, "action": "X"},
    )
    monkeypatch.setattr(runner_module, "format_signal_message", lambda row: row["ticker"])

    runner_module.main()

    # both tickers still get a real pass -- the preview failure was isolated.
    assert set(calls) == {"PREVIEWFAILS", "FINE"}
    err = capsys.readouterr().err
    assert "PREVIEWFAILS: FAILED during preview" in err
