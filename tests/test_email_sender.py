"""Tests for the real SMTP email sender -- mock mode when no credentials
are configured (never touches the network), and the live path verified
against a fake SMTP server (never a real network call in tests).
"""
import scaata.notify.email_sender as email_sender_module
from scaata.notify.email_sender import send_email


def test_send_email_is_mock_without_credentials(monkeypatch):
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    monkeypatch.delenv("GMAIL_SENDER_ADDRESS", raising=False)

    result, source = send_email("someone@example.com", "Subject", "Body")

    assert source == "mock"
    assert result["to"] == "someone@example.com"


class _FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.started_tls = False
        self.login_calls = []
        self.sent = []
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        self.started_tls = True

    def login(self, user, password):
        self.login_calls.append((user, password))

    def sendmail(self, from_addr, to_addrs, msg):
        self.sent.append((from_addr, to_addrs, msg))


def test_send_email_live_path_calls_smtp_correctly(monkeypatch):
    monkeypatch.setenv("GMAIL_SENDER_ADDRESS", "me@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcd efgh ijkl mnop")  # Google's space-separated display format
    _FakeSMTP.instances = []
    monkeypatch.setattr(email_sender_module.smtplib, "SMTP", _FakeSMTP)

    result, source = send_email("recipient@example.com", "Test Subject", "Test Body")

    assert source == "live"
    assert result == {"to": "recipient@example.com", "subject": "Test Subject"}

    smtp = _FakeSMTP.instances[0]
    assert smtp.host == "smtp.gmail.com"
    assert smtp.started_tls is True
    # spaces must be stripped from the app password before use
    assert smtp.login_calls == [("me@gmail.com", "abcdefghijklmnop")]
    assert len(smtp.sent) == 1
    from_addr, to_addrs, msg = smtp.sent[0]
    assert from_addr == "me@gmail.com"
    assert to_addrs == ["recipient@example.com"]
    assert "Test Subject" in msg
    assert "Test Body" in msg


def test_send_email_never_sends_when_only_one_credential_present(monkeypatch):
    monkeypatch.setenv("GMAIL_SENDER_ADDRESS", "me@gmail.com")
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    _FakeSMTP.instances = []
    monkeypatch.setattr(email_sender_module.smtplib, "SMTP", _FakeSMTP)

    result, source = send_email("recipient@example.com", "Subject", "Body")

    assert source == "mock"
    assert _FakeSMTP.instances == []
