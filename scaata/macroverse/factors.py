"""The macro factors Macroverse can measure daily, frozen in a snapshot.

Three factors, chosen because a long daily history exists for each:

    market  Nifty 50
    crude   Brent crude, in US dollars
    rupee   USD/INR (up = the rupee is weaker)

Interest rates live in a separate snapshot (see the rates section below). Inflation, GDP and the rest are not here: there is no
reliable free daily series, or the data is monthly and cannot be tied to
five stocks' daily prices. Scenarios on those are listed as unavailable
rather than guessed.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from scaata.config import ROOT_DIR

DATA_DIR = ROOT_DIR / "macroverse" / "data"
SNAPSHOT = DATA_DIR / "factors.csv.gz"
SNAPSHOT_META = DATA_DIR / "factors.meta.json"
SOURCES = {"market": "^NSEI", "crude": "BZ=F", "rupee": "INR=X"}
LABELS = {"market": "Nifty 50", "crude": "Brent crude ($)", "rupee": "USD/INR"}
HISTORY_START = "2006-01-01"


def refresh() -> dict:
    """Downloads the factor closes and replaces the snapshot."""
    import yfinance as yf

    columns = {}
    for name, symbol in SOURCES.items():
        series = yf.download(symbol, start=HISTORY_START, progress=False, auto_adjust=True)["Close"].squeeze().dropna()
        series = series[series > 0]
        series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
        columns[name] = series[~series.index.duplicated()]
    frame = pd.DataFrame(columns).sort_index()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.round(4).to_csv(SNAPSHOT, compression={"method": "gzip", "mtime": 0})
    meta = {"source": "Yahoo Finance via yfinance: " + ", ".join(f"{k}={v}" for k, v in SOURCES.items()),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "first": {k: str(v.index[0].date()) for k, v in columns.items()},
            "last": str(frame.index[-1].date()), "rows": len(frame),
            "sha256": hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()}
    SNAPSHOT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def meta() -> dict | None:
    return json.loads(SNAPSHOT_META.read_text(encoding="utf-8")) if SNAPSHOT_META.exists() else None


def load() -> tuple[pd.DataFrame, dict]:
    """(factor closes, meta). Fails loudly if the snapshot is missing or was altered."""
    info = meta()
    if info is None or not SNAPSHOT.exists():
        raise FileNotFoundError("No factor snapshot. Run: python -m scaata.macroverse refresh")
    if hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError("The factor snapshot does not match its recorded hash.")
    return pd.read_csv(SNAPSHOT, index_col=0, parse_dates=True), info


def aligned(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Factor closes on NSE sessions, carrying the last known value over days a factor did not trade."""
    frame, _ = load()
    return frame.reindex(frame.index.union(sessions)).ffill().reindex(sessions)


# --- interest rates (Phase D) -------------------------------------------------------------------------------
# No free daily Indian 10-year yield history exists. Two honest stand-ins, kept in a separate snapshot so the
# three-factor snapshot above (and every result built on it) is untouched:
#   us10y  US 10-year Treasury yield, in percent, daily since 2006 (^TNX). A global rates factor.
#   gilt   price of the Nippon India ETF Gilt 5Y (GILT5YBEES), daily since April 2021. A bond's price falls when
#          its yield rises, so a change in the yield is about -(price change) / duration; see `rates.py`.
RATES_SNAPSHOT = DATA_DIR / "rates.csv.gz"
RATES_META = DATA_DIR / "rates.meta.json"
RATES_SOURCES = {"us10y": "^TNX", "gilt": "GILT5YBEES.NS"}


def refresh_rates() -> dict:
    import yfinance as yf

    columns = {}
    for name, symbol in RATES_SOURCES.items():
        series = yf.download(symbol, start=HISTORY_START, progress=False, auto_adjust=True)["Close"].squeeze().dropna()
        series = series[series > 0]
        series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
        columns[name] = series[~series.index.duplicated()]
    frame = pd.DataFrame(columns).sort_index()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.round(4).to_csv(RATES_SNAPSHOT, compression={"method": "gzip", "mtime": 0})
    meta_ = {"source": "Yahoo Finance via yfinance: " + ", ".join(f"{k}={v}" for k, v in RATES_SOURCES.items()),
             "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "first": {k: str(v.index[0].date()) for k, v in columns.items()}, "last": str(frame.index[-1].date()),
             "rows": len(frame), "sha256": hashlib.sha256(RATES_SNAPSHOT.read_bytes()).hexdigest()}
    RATES_META.write_text(json.dumps(meta_, indent=2), encoding="utf-8")
    return meta_


def load_rates() -> tuple[pd.DataFrame, dict]:
    if not RATES_META.exists() or not RATES_SNAPSHOT.exists():
        raise FileNotFoundError("No rates snapshot. Run: python -m scaata.macroverse refresh-rates")
    info = json.loads(RATES_META.read_text(encoding="utf-8"))
    if hashlib.sha256(RATES_SNAPSHOT.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError("The rates snapshot does not match its recorded hash.")
    return pd.read_csv(RATES_SNAPSHOT, index_col=0, parse_dates=True), info
