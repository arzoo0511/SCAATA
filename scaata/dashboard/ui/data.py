"""Shapes what the daily agent wrote into frames the charts can draw.

Pure functions over the already-loaded JSON (paper book, journal, memory,
evaluation, backtest reports). Nothing here fetches, decides or writes, so
the dashboard cannot disturb the daily run. Missing files give empty frames,
never an exception: a fresh clone must still render.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scaata.config import ROOT_DIR

NAMES = {"HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank", "PETRONET.NS": "Petronet LNG",
         "IOC.NS": "Indian Oil", "ITC.NS": "ITC", "^NSEI": "Nifty 50"}
ADVISORS = ("trend", "volatility", "momentum", "news")
RESULTS = ROOT_DIR / "results"
EVALUATION_PATH = ROOT_DIR / "forward_test" / "hansei_evaluation.json"
REPO_URL = "https://github.com/arzoo0511/SCAATA"
APP_URL = "https://hansei.streamlit.app/"


def short(symbol: str) -> str:
    return NAMES.get(symbol, symbol.replace(".NS", ""))


def ticker(symbol: str) -> str:
    return symbol.replace(".NS", "")


def read_json(path: Path):
    """The file's contents, or None if it is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def equity_frame(book: dict | None) -> pd.DataFrame:
    """One row per session: both equities, the gap, and each one's drawdown."""
    history = (book or {}).get("equity_history") or []
    if not history:
        return pd.DataFrame(columns=["date", "hansei", "hold", "cash", "invested", "gap",
                                     "hansei_dd", "hold_dd"])
    frame = pd.DataFrame(history).rename(columns={"equity": "hansei", "benchmark_equity": "hold"})
    frame["date"] = pd.to_datetime(frame["date"])
    start = float(book["initial_cash"])
    frame["invested"] = 1 - frame["cash"] / frame["hansei"]
    frame["gap"] = (frame["hansei"] - frame["hold"]) / start
    for leg in ("hansei", "hold"):
        frame[f"{leg}_dd"] = frame[leg] / frame[leg].cummax() - 1
    return frame[["date", "hansei", "hold", "cash", "invested", "gap", "hansei_dd", "hold_dd"]]


def trades_frame(book: dict | None) -> pd.DataFrame:
    trades = (book or {}).get("trades") or []
    if not trades:
        return pd.DataFrame(columns=["date", "symbol", "action", "qty", "price", "cost", "reason"])
    frame = pd.DataFrame(trades)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def decisions_frame(journal: list[dict]) -> pd.DataFrame:
    """One row per (session, stock): what it held, wanted, did, and each advisor's view."""
    rows = []
    for entry in journal or []:
        for d in entry.get("decisions", []):
            news = entry.get("news", {}).get(d["symbol"], {}).get("signal", {})
            rows.append({"session": entry["session"], "symbol": d["symbol"], "action": d["action"],
                         "score": d["score"], "target": d["target_exposure"], "held": d["current_exposure"],
                         "reason": (d.get("reasons") or [""])[0], "material_news": news.get("material_count", 0),
                         **{a: d.get("views", {}).get(a, 0.0) for a in ADVISORS}})
    frame = pd.DataFrame(rows, columns=["session", "symbol", "action", "score", "target", "held", "reason",
                                        "material_news", *ADVISORS])
    frame["session"] = pd.to_datetime(frame["session"])
    return frame


def trust_frame(journal: list[dict]) -> pd.DataFrame:
    """Trust in each advisor after every session (the journal keeps what it was at the time)."""
    rows = [{"session": e["session"], **{a: e.get("weights", {}).get(a) for a in ADVISORS}} for e in journal or []]
    frame = pd.DataFrame(rows, columns=["session", *ADVISORS])
    frame["session"] = pd.to_datetime(frame["session"])
    return frame


def lessons_frame(memory: dict | None) -> pd.DataFrame:
    rows = []
    for lesson in (memory or {}).get("lessons", []):
        moves = {a: lesson["weights_after"][a] - lesson["weights_before"][a] for a in ADVISORS
                 if a in lesson.get("weights_after", {}) and a in lesson.get("weights_before", {})}
        stocks = lesson.get("stocks", {})
        rows.append({"learned_on": lesson["learned_on"], "about_day": lesson["about_day"], "best": lesson["best"],
                     "worst": lesson["worst"], "mean_return": sum(stocks.values()) / len(stocks) if stocks else 0.0,
                     **{f"d_{a}": moves.get(a, 0.0) for a in ADVISORS}})
    frame = pd.DataFrame(rows, columns=["learned_on", "about_day", "best", "worst", "mean_return",
                                        *[f"d_{a}" for a in ADVISORS]])
    for column in ("learned_on", "about_day"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def beliefs_frame(memory: dict | None) -> pd.DataFrame:
    """Every belief on file: the day, the stock, and what happened next once it could be judged."""
    rows = [{"date": o["date"], "symbol": o["symbol"], "judged": bool(o.get("judged")),
             "forward_return": o.get("forward_return")} for o in (memory or {}).get("observations", [])]
    frame = pd.DataFrame(rows, columns=["date", "symbol", "judged", "forward_return"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def news_frame(journal: list[dict]) -> pd.DataFrame:
    """Every material story the agent read, once (a story seen on several days keeps its first sighting)."""
    rows, seen = [], set()
    for entry in journal or []:
        for symbol, digest in entry.get("news", {}).items():
            for story in digest.get("top", []):
                key = (symbol, story.get("title"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"session": entry["session"], "symbol": symbol, "title": story.get("title", ""),
                             "source": story.get("source") or "", "sentiment": float(story.get("sentiment", 0.0)),
                             "material": float(story.get("material", 0.0)), "count": int(story.get("count", 1)),
                             "published": story.get("published_utc")})
    frame = pd.DataFrame(rows, columns=["session", "symbol", "title", "source", "sentiment", "material", "count",
                                        "published"])
    frame["session"] = pd.to_datetime(frame["session"])
    frame["published"] = pd.to_datetime(frame["published"], utc=True, errors="coerce")
    return frame


def allocation(book: dict | None, journal: list[dict]) -> list[tuple[str, float]]:
    """Where the paper money sits after the latest session: (ticker, rupees), stocks then liquid fund."""
    if not book or not journal:
        return []
    parts = [(ticker(d["symbol"]), float(d.get("value", 0.0))) for d in journal[-1].get("decisions", [])
             if d.get("value", 0.0) > 0]
    history = book.get("equity_history") or []
    cash = float(history[-1]["cash"]) if history else float(book.get("cash", 0.0))
    return parts + [("Liquid fund", cash)]


def gap_story(evaluation: dict | None) -> dict | None:
    """The live gap vs holding, split into its two causes."""
    perf = (evaluation or {}).get("performance")
    if not perf:
        return None
    return {"hold": perf["hold_return"], "exposure": perf["gap_from_exposure"],
            "selection": perf["gap_from_selection"], "hansei": perf["hansei_return"], "gap": perf["gap"],
            "average_invested": perf["average_invested"], "sessions": perf["sessions"],
            "hansei_dd": perf["hansei_max_drawdown"], "hold_dd": perf["hold_max_drawdown"],
            "sessions_ahead": perf.get("sessions_ahead")}


def mind_payload(journal: list[dict]) -> list[dict]:
    """The journal, trimmed to what the Mind explorer draws. One item per session."""
    out = []
    for entry in journal or []:
        stocks = []
        for d in entry.get("decisions", []):
            digest = entry.get("news", {}).get(d["symbol"], {})
            stocks.append({
                "symbol": ticker(d["symbol"]), "name": short(d["symbol"]), "action": d["action"],
                "score": d["score"], "target": d["target_exposure"], "held": d["current_exposure"],
                "reasons": d.get("reasons", []), "views": {a: d.get("views", {}).get(a, 0.0) for a in ADVISORS},
                "news": [{"title": s.get("title", ""), "source": s.get("source") or "",
                          "sentiment": s.get("sentiment", 0.0)} for s in digest.get("top", [])[:3]]})
        out.append({"session": entry["session"], "equity": entry.get("equity"),
                    "hold": entry.get("benchmark_equity"),
                    "weights": {a: entry.get("weights", {}).get(a, 0.25) for a in ADVISORS},
                    "lessons": entry.get("lessons_learned", 0),
                    "filled": [{"symbol": ticker(t["symbol"]), "action": t["action"], "qty": t["qty"],
                                "price": t["price"]} for t in entry.get("filled", [])],
                    "stocks": stocks})
    return out


def backtest_curve(report: dict | None) -> pd.DataFrame:
    curve = (report or {}).get("curve")
    if not curve:
        return pd.DataFrame(columns=["date", "hansei", "hold", "hansei_dd", "hold_dd"])
    frame = pd.DataFrame({"date": pd.to_datetime(curve["dates"]), "hansei": curve["hansei"],
                          "hold": curve["buy_and_hold"]})
    for leg in ("hansei", "hold"):
        frame[f"{leg}_dd"] = frame[leg] / frame[leg].cummax() - 1
    return frame


def blocks_frame(long_report: dict | None) -> pd.DataFrame:
    """The long backtest's separate multi-year blocks (not the tune/holdout aggregates)."""
    rows = [{"block": name, "cagr_diff": r["cagr_diff"], "sharpe_diff": r["sharpe_diff"],
             "hansei_dd": r["hansei"]["max_drawdown"], "hold_dd": r["buy_and_hold"]["max_drawdown"],
             "hansei_cagr": r["hansei"]["cagr"], "hold_cagr": r["buy_and_hold"]["cagr"]}
            for name, r in ((long_report or {}).get("periods") or {}).items() if name[:1].isdigit()]
    return pd.DataFrame(rows, columns=["block", "cagr_diff", "sharpe_diff", "hansei_dd", "hold_dd",
                                       "hansei_cagr", "hold_cagr"])


def signal_frame(study: dict | None, horizon: str = "20_sessions") -> pd.DataFrame:
    """Each advisor's return spread (liked minus disliked stocks) with its 95% interval, per split."""
    rows = []
    for split, horizons in ((study or {}).get("large_caps") or {}).items():
        for advisor, r in (horizons.get(horizon) or {}).items():
            low, high = r.get("spread_ci95") or (None, None)
            rows.append({"split": split, "advisor": advisor, "spread": r.get("spread"), "low": low, "high": high,
                         "hit_rate": r.get("hit_rate"), "opinions": r.get("opinions")})
    return pd.DataFrame(rows, columns=["split", "advisor", "spread", "low", "high", "hit_rate", "opinions"])


def now_payload(book: dict, journal: list[dict], evaluation: dict | None) -> dict:
    """Everything the Now cockpit draws, straight from the saved book, journal and evaluation."""
    equity, story = equity_frame(book), gap_story(evaluation)
    start, last = float(book["initial_cash"]), equity.iloc[-1]
    decisions = decisions_frame(journal)
    stocks = []
    for d in (journal[-1].get("decisions", []) if journal else []):
        history = decisions.loc[decisions["symbol"] == d["symbol"], "held"].round(3).tolist()
        stocks.append({"symbol": ticker(d["symbol"]), "name": short(d["symbol"]), "action": d["action"],
                       "held": d["current_exposure"], "target": d["target_exposure"],
                       "value": float(d.get("value", 0.0)), "history": history,
                       "reason": "; ".join(d.get("reasons", [])[:3])})
    if story:
        lede = (f"It has kept on average <b>{story['average_invested']:.0%}</b> of the money in stocks; the rest sat "
                f"in a liquid fund while the five stocks {'rose' if story['hold'] >= 0 else 'fell'}. Its worst fall "
                f"so far is <b>{story['hansei_dd']:.1%}</b> against holding&rsquo;s <b>{story['hold_dd']:.1%}</b>.")
    else:
        lede = "The comparison is the same ₹10,000 split equally across the five stocks and never touched."
    return {
        "summary": {"equity": float(last["hansei"]), "hold": float(last["hold"]), "ret": float(last["hansei"]) / start - 1,
                    "hold_ret": float(last["hold"]) / start - 1, "gap": float(last["gap"]), "sessions": len(equity),
                    "dd": float(equity["hansei_dd"].min()), "hold_dd": float(equity["hold_dd"].min())},
        "lede": lede,
        "series": {"dates": [d.strftime("%Y-%m-%d") for d in equity["date"]],
                   **{k: equity[k].round(5).tolist() for k in ("hansei", "hold", "gap", "hansei_dd", "hold_dd",
                                                                "invested")}},
        "trades": [{"date": t["date"], "action": t["action"], "name": short(t["symbol"]), "qty": t["qty"],
                    "price": float(t["price"])} for t in book.get("trades", [])],
        "allocation": allocation(book, journal),
        "bridge": story,
        "stocks": stocks,
        "pending": [{"name": short(o["symbol"]), "target": float(o.get("target_value", 0.0)),
                     "reason": (o.get("reason") or "").split(";")[0]} for o in book.get("pending", [])],
    }
