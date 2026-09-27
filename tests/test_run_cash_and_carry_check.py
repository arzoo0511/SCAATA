"""Tests for the standalone (Task-Scheduler-launched) cash-and-carry
checker: must do nothing but log when GMAIL_SENDER_ADDRESS isn't
configured (no crash), and must report send status per detected
opportunity.
"""
import scaata.notify.run_cash_and_carry_check as runner_module


def test_main_exits_quietly_without_recipient_configured(monkeypatch, capsys):
    monkeypatch.delenv("GMAIL_SENDER_ADDRESS", raising=False)

    runner_module.main()

    err = capsys.readouterr().err
    assert "GMAIL_SENDER_ADDRESS not configured" in err


def test_main_reports_no_opportunity_when_none_detected(monkeypatch, capsys):
    monkeypatch.setenv("GMAIL_SENDER_ADDRESS", "me@example.com")
    monkeypatch.setattr(runner_module, "send_equity_opportunity_alerts", lambda recipient: [])

    runner_module.main()

    out = capsys.readouterr().out
    assert "No cash-and-carry opportunity" in out


def test_main_reports_sent_status_per_opportunity(monkeypatch, capsys):
    monkeypatch.setenv("GMAIL_SENDER_ADDRESS", "me@example.com")
    results = [
        {"subject": "AAPL opportunity", "send_source": "live"},
        {"subject": "MSFT opportunity", "send_source": "mock"},
    ]
    monkeypatch.setattr(runner_module, "send_equity_opportunity_alerts", lambda recipient: results)

    runner_module.main()

    out = capsys.readouterr().out
    assert "SENT: AAPL opportunity" in out
    assert "MOCK" in out and "MSFT opportunity" in out
