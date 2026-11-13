"""A frozen price snapshot, so every experiment runs on the same data.

Yahoo revises history and rate-limits, so experiments that each download
their own prices are not comparable or reproducible. The Lab downloads once,
writes the snapshot with its source, retrieval time and hash, and every
experiment records which snapshot it used.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from scaata.config import INDIA_TICKERS, ROOT_DIR

DATA_DIR = ROOT_DIR / "research" / "data"
SNAPSHOT = DATA_DIR / "prices.csv.gz"
SNAPSHOT_META = DATA_DIR / "prices.meta.json"
HISTORY_START = "2006-01-01"


def refresh(end: date | None = None) -> dict:
    """Downloads prices through `end` (default today) and replaces the snapshot."""
    from scaata.agent.backtest import load_prices

    end = end or date.today()
    opens, closes = load_prices(end, HISTORY_START)
    frame = pd.concat({"open": opens, "close": closes}, axis=1)
    frame.columns = [f"{kind}:{symbol}" for kind, symbol in frame.columns]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.round(4).to_csv(SNAPSHOT, compression={"method": "gzip", "mtime": 0})
    meta = {"source": "Yahoo Finance via yfinance (adjusted; bad opens repaired, zero-volume bars dropped)",
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "first": str(closes.index[0].date()), "last": str(closes.index[-1].date()), "sessions": len(closes),
            "symbols": list(INDIA_TICKERS), "sha256": hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()}
    SNAPSHOT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def meta(path: Path | None = None) -> dict | None:
    path = path or SNAPSHOT_META
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(opens, closes, meta) from the snapshot. Fails loudly if it is missing or was altered."""
    info = meta()
    if info is None or not SNAPSHOT.exists():
        raise FileNotFoundError("No price snapshot. Run: python -m scaata.research refresh")
    if hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError("The price snapshot does not match its recorded hash.")
    frame = pd.read_csv(SNAPSHOT, index_col=0, parse_dates=True)
    opens = frame[[c for c in frame if c.startswith("open:")]].rename(columns=lambda c: c.split(":", 1)[1])
    closes = frame[[c for c in frame if c.startswith("close:")]].rename(columns=lambda c: c.split(":", 1)[1])
    return opens, closes, info
