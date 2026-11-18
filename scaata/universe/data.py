"""The universe's prices: download, clean, freeze, load.

Daily adjusted closes from Yahoo Finance for about 200 NSE stocks since 2006, saved as one compressed CSV with
its source, retrieval time and hash, like the five-stock research snapshot. Redistribution of NSE's own data
is not attempted: this is Yahoo's adjusted series, for private research.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scaata.agent.signal_study import UNIVERSE as CORE
from scaata.config import ROOT_DIR

DATA_DIR = ROOT_DIR / "research" / "data"
SNAPSHOT = DATA_DIR / "universe.csv.gz"
SNAPSHOT_META = DATA_DIR / "universe.meta.json"
HISTORY_START = "2006-01-01"
CHUNK = 40
MIN_HISTORY = 504          # sessions a stock needs before it can be scored or traded

MID_CAPS = """
ABBOTINDIA ABCAPITAL ACC AJANTPHARM ALKEM APLAPOLLO APOLLOTYRE ASTRAL AUBANK BALKRISIND BANDHANBNK BANKINDIA
BATAINDIA BDL BHEL BIOCON BSE CASTROLIND CESC COFORGE CONCOR CROMPTON CUB DALBHARAT DEEPAKNTR DIXON EMAMILTD
ESCORTS EXIDEIND FEDERALBNK FORTIS GILLETTE GLENMARK GODREJPROP GRANULES GUJGASLTD HDFCAMC HONAUT IDEA
IDFCFIRSTB IEX IGL INDIAMART INDUSTOWER IPCALAB JKCEMENT JSL JUBLFOOD KAJARIACER KEI KPITTECH LALPATHLAB
LAURUSLABS LICHSGFIN LTIM LTTS MANAPPURAM MAXHEALTH MCX METROPOLIS MFSL MPHASIS MRF NATIONALUM NAVINFLUOR NHPC
NMDC OBEROIRLTY OFSS OIL PATANJALI PERSISTENT PHOENIXLTD PIIND POLICYBZR POLYCAB PRESTIGE RAMCOCEM RBLBANK SAIL
SBICARD SOLARINDS SONACOMS STARHEALTH SUMICHEM SUNDARMFIN SUNTV SUPREMEIND SYNGENE TATACOMM TATAELXSI TATAINVEST
THERMAX TIINDIA TORNTPOWER TRIDENT UBL UNIONBANK UPL VOLTAS WHIRLPOOL YESBANK ZEEL ETERNAL
""".split()
UNIVERSE = sorted(set(CORE) | {f"{s}.NS" for s in MID_CAPS})


def _clean(closes: pd.DataFrame) -> pd.DataFrame:
    """Drops non-positive prints and one-day spikes that reverse the next day (bad ticks, not moves)."""
    closes = closes.where(closes > 0)
    ret = closes.pct_change(fill_method=None)
    spike = (ret.abs() > 0.5) & (ret.shift(-1) * ret < 0) & ((closes.shift(-1) / closes.shift(1) - 1).abs() < 0.2)
    return closes.mask(spike)


def download(symbols: list[str] | None = None, start: str = HISTORY_START, end: str | None = None) -> pd.DataFrame:
    """Adjusted closes for each symbol Yahoo can serve; the rest are dropped (the caller can see which)."""
    import yfinance as yf

    symbols = symbols or UNIVERSE
    end = end or (date.today() + pd.Timedelta(days=1)).isoformat()
    frames = []
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i + CHUNK]
        for attempt in range(2):
            try:
                data = yf.download(chunk, start=start, end=end, progress=False, auto_adjust=True, threads=True)["Close"]
                data = data.to_frame(chunk[0]) if isinstance(data, pd.Series) else data
                frames.append(data)
                break
            except Exception:                    # a rate limit is routine; retry once, then move on
                continue
    if not frames:
        return pd.DataFrame()
    closes = pd.concat(frames, axis=1)
    closes = closes.loc[:, ~closes.columns.duplicated()]
    for symbol in sorted(set(symbols) - set(closes.dropna(how="all", axis=1).columns)):
        try:                                      # batch calls silently drop some names; ask for each once more alone
            one = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)["Close"].squeeze()
            if isinstance(one, pd.Series) and one.notna().sum() > 0:
                closes[symbol] = one
        except Exception:
            continue
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
    closes = closes.dropna(how="all", axis=1).dropna(how="all").sort_index()
    return _clean(closes)


def freeze(closes: pd.DataFrame, path: Path | None = None) -> dict:
    """Writes the snapshot and its record (source, time, coverage, hash)."""
    path = path or SNAPSHOT
    path.parent.mkdir(parents=True, exist_ok=True)
    closes.round(2).to_csv(path, compression={"method": "gzip", "mtime": 0})
    first = closes.apply(lambda s: s.first_valid_index())
    meta = {"source": "Yahoo Finance via yfinance (adjusted closes; non-positive prints and one-day reversing spikes removed)",
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "first": str(closes.index[0].date()), "last": str(closes.index[-1].date()), "sessions": len(closes),
            "stocks": int(closes.shape[1]), "requested": len(UNIVERSE),
            "missing": sorted(set(UNIVERSE) - set(closes.columns)),
            "with_history": int((closes.notna().sum() >= MIN_HISTORY).sum()),
            "later_listings": {s: str(d.date()) for s, d in first.items() if d > pd.Timestamp("2008-01-01")},
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bias": "Today's large and mid caps only: stocks that left the index or were delisted are missing."}
    path.with_name(SNAPSHOT_META.name if path == SNAPSHOT else path.stem + ".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def meta(path: Path | None = None) -> dict | None:
    path = path or SNAPSHOT_META
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load(path: Path | None = None, meta_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """(closes, meta). Fails loudly if the snapshot is missing or does not match its recorded hash."""
    path, meta_path = path or SNAPSHOT, meta_path or SNAPSHOT_META
    info = meta(meta_path)
    if info is None or not path.exists():
        raise FileNotFoundError("No universe snapshot. Run: python -m scaata.universe refresh")
    if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError("The universe snapshot does not match its recorded hash.")
    return pd.read_csv(path, index_col=0, parse_dates=True), info
