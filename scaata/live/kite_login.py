"""Daily Kite login for HANSEI.

Kite access tokens expire at 06:00 IST, so run this once each morning:

    python -m scaata.live.kite_login

It opens the Kite login page in your browser. You log in there yourself;
Kite then redirects to http://127.0.0.1:5000/ with a one-time
request_token, which this script catches, exchanges for the day's access
token, and saves to `.kite_session.json`. It then prints your account name,
funds and holdings so you can see the connection works.

If your app's redirect URL is https:// the browser can't reach this local
page, but the address bar will still show `request_token=...`. Copy that
value and run:

    python -m scaata.live.kite_login --token <request_token>
"""
from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from scaata.live.kite_broker import KiteError, create_session, credentials, get_funds, get_holdings, login_url

HOST, PORT = "127.0.0.1", 5000
WAIT_SECONDS = 300


def _wait_for_request_token(wait_seconds: int = WAIT_SECONDS) -> str | None:
    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            query = parse_qs(urlparse(self.path).query)
            if query.get("status", [""])[0] == "success" and "request_token" in query:
                captured["token"] = query["request_token"][0]
                message = "HANSEI: logged in to Kite. You can close this tab."
            else:
                message = "HANSEI: waiting for the Kite login redirect."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(message.encode())

        def log_message(self, *args):
            pass

    deadline = time.time() + wait_seconds
    with HTTPServer((HOST, PORT), Handler) as server:
        while "token" not in captured and time.time() < deadline:
            server.timeout = max(1.0, deadline - time.time())
            server.handle_request()  # stray requests (e.g. favicon) just loop
    return captured.get("token")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Kite login for HANSEI")
    parser.add_argument("--token", help="request_token copied from the redirect URL (only if automatic capture fails)")
    args = parser.parse_args()

    api_key, api_secret = credentials()
    if not api_key or not api_secret:
        sys.exit("KITE_CONNECT_API_KEY / KITE_CONNECT_API_SECRET are missing from .env")

    token = args.token
    if not token:
        print(f"Opening the Kite login page. Log in there; waiting up to {WAIT_SECONDS // 60} min "
              f"for the redirect to http://{HOST}:{PORT}/ ...", flush=True)
        webbrowser.open(login_url(api_key))
        token = _wait_for_request_token()
        if not token:
            sys.exit("No request_token received. If the redirect page did not load, copy request_token "
                     "from the address bar and run: python -m scaata.live.kite_login --token <request_token>")

    try:
        user = create_session(token)
    except KiteError as e:
        sys.exit(f"Kite rejected the login: {e}")
    print(f"Logged in as {user['user_name']} ({user['user_id']}). Session valid until 06:00 IST.")

    funds, _ = get_funds()
    holdings, _ = get_holdings()
    if funds:
        print(f"Funds: net Rs {funds['net']:,.0f}, cash Rs {funds['cash']:,.0f}")
    print(f"Holdings: {len(holdings)}")
    for h in holdings:
        print(f"  {h['symbol']:<12} {h['qty']:>8g} @ Rs {h['avg_price']:,.2f}  "
              f"(last Rs {h['last_price']:,.2f}, P&L Rs {h['pnl']:,.0f})")


if __name__ == "__main__":
    main()
