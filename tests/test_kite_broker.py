"""Tests for the read-only Kite Connect wrapper. No network: a fake HTTP
object stands in for `requests`, and the session file lives in tmp_path."""
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from scaata.live import kite_broker as kb


class _Resp:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def json(self):
        return self._body


class _FakeHttp:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url, headers, None))
        return self.routes[url]

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append(("POST", url, headers, data))
        return self.routes[url]


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    monkeypatch.setenv("KITE_CONNECT_API_KEY", "key123")
    monkeypatch.setenv("KITE_CONNECT_API_SECRET", "secret456")


def _ok(data):
    return _Resp({"status": "success", "data": data})


def _write_session(path, created=None, token="tok"):
    created = created or datetime.now(timezone.utc)
    path.write_text(json.dumps({"access_token": token, "user_id": "AB1234", "user_name": "Test",
                                "created_at_utc": created.isoformat()}))


def test_checksum_is_sha256_of_key_token_secret():
    assert kb.checksum("k", "r", "s") == hashlib.sha256(b"krs").hexdigest()


def test_credentials_accept_either_name(monkeypatch):
    monkeypatch.delenv("KITE_CONNECT_API_KEY")
    monkeypatch.delenv("KITE_CONNECT_API_SECRET")
    monkeypatch.setenv("KITE_API_KEY", " short ")
    monkeypatch.setenv("KITE_API_SECRET", "s")
    assert kb.credentials() == ("short", "s")


def test_login_url_carries_the_api_key():
    assert kb.login_url("key123") == "https://kite.zerodha.com/connect/login?v=3&api_key=key123"


def test_token_expires_at_the_next_0600_ist():
    ist = kb.IST
    assert kb.session_expiry(datetime(2026, 9, 22, 9, 0, tzinfo=ist)) == datetime(2026, 9, 23, 6, 0, tzinfo=ist)
    assert kb.session_expiry(datetime(2026, 9, 22, 2, 0, tzinfo=ist)) == datetime(2026, 9, 22, 6, 0, tzinfo=ist)


def test_create_session_saves_the_token_but_does_not_return_it(tmp_path):
    http = _FakeHttp({f"{kb.API_ROOT}/session/token": _ok({"access_token": "tok", "user_id": "AB1234", "user_name": "Test"})})
    path = tmp_path / "s.json"

    user = kb.create_session("req789", http=http, path=path)

    assert user == {"user_id": "AB1234", "user_name": "Test"}
    assert json.loads(path.read_text())["access_token"] == "tok"
    sent = http.calls[0][3]
    assert sent["checksum"] == kb.checksum("key123", "req789", "secret456")
    assert "secret456" not in json.dumps(sent)  # the secret itself is never sent


def test_expired_or_missing_session_means_unavailable_not_empty(tmp_path):
    path = tmp_path / "s.json"
    http = _FakeHttp({})
    assert kb.get_holdings(http=http, path=path) == ([], "unavailable")

    _write_session(path, created=datetime.now(timezone.utc) - timedelta(days=2))
    assert kb.get_holdings(http=http, path=path) == ([], "unavailable")
    assert http.calls == []  # never calls Kite with a dead token


def test_holdings_are_parsed_and_authorised(tmp_path):
    path = tmp_path / "s.json"
    _write_session(path)
    http = _FakeHttp({f"{kb.API_ROOT}/portfolio/holdings": _ok([
        {"tradingsymbol": "ITC", "exchange": "NSE", "quantity": 10, "t1_quantity": 5,
         "average_price": 400.5, "last_price": 410.0, "pnl": 142.5},
    ])})

    holdings, source = kb.get_holdings(http=http, path=path)

    assert source == "live"
    assert holdings == [{"symbol": "ITC", "exchange": "NSE", "qty": 15.0, "avg_price": 400.5,
                         "last_price": 410.0, "pnl": 142.5}]
    assert http.calls[0][2]["Authorization"] == "token key123:tok"


def test_positions_drop_closed_rows(tmp_path):
    path = tmp_path / "s.json"
    _write_session(path)
    http = _FakeHttp({f"{kb.API_ROOT}/portfolio/positions": _ok({"net": [
        {"tradingsymbol": "IOC", "exchange": "NSE", "product": "CNC", "quantity": 20, "average_price": 140.0, "pnl": 12.0},
        {"tradingsymbol": "ITC", "exchange": "NSE", "product": "MIS", "quantity": 0, "average_price": 0.0, "pnl": -3.0},
    ]})})

    positions, _ = kb.get_positions(http=http, path=path)

    assert [p["symbol"] for p in positions] == ["IOC"]


def test_kite_errors_are_raised_not_swallowed(tmp_path):
    path = tmp_path / "s.json"
    _write_session(path)
    http = _FakeHttp({f"{kb.API_ROOT}/user/margins/equity": _Resp(
        {"status": "error", "error_type": "TokenException", "message": "Incorrect api_key or access_token."}, 403)})

    with pytest.raises(kb.KiteError, match="TokenException"):
        kb.get_funds(http=http, path=path)


def test_broker_has_no_order_placement():
    """Read-only until a strategy shows an edge -- a structural guarantee."""
    assert not any(name for name in dir(kb) if "order" in name.lower())
