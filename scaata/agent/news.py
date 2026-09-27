"""Company news for the Indian book: fetch, filter, score.

Headlines come from Google News (India edition) and Yahoo Finance. Most of
what those return is noise for an investor -- "ICICIBANK share price today",
weekly technical outlooks, stories about a different bank -- so every
headline is judged on three things, not just tone:

- **relevant**: is it actually about this company?
- **sentiment**: good or bad news for the stock, -1..1
- **material**: would it move a long-term holder's view? Results, guidance,
  regulator action, deals, management changes, rating changes score high;
  price recaps and chart commentary score near zero.

Scoring uses an LLM through Groq when a key is configured, and falls back to
an offline keyword scorer so a provider outage never stops the daily run.
Every day's headlines and scores are saved, so the dashboard can show what
the agent read, and later decisions can be checked against it.
"""
from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

from scaata.config import DATA_CACHE_DIR

load_dotenv()

NEWS_DIR = DATA_CACHE_DIR / "news_india"
NEWS_DIR.mkdir(parents=True, exist_ok=True)
LOOKBACK_DAYS = 3
MAX_HEADLINES = 40
LLM_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")

# How each company is written about -- what a headline must mention to count.
COMPANY_NAMES = {
    "HDFCBANK.NS": ["HDFC Bank"],
    "ICICIBANK.NS": ["ICICI Bank"],
    "PETRONET.NS": ["Petronet LNG", "Petronet"],
    "IOC.NS": ["Indian Oil", "IOCL", "IndianOil"],
    "ITC.NS": ["ITC Ltd", "ITC"],
}

# Offline fallback vocabulary (small, finance-specific, deliberately conservative).
POSITIVE = ["beats", "beat estimates", "surge", "jumps", "soars", "record", "upgrade", "raises", "rises",
            "profit up", "strong", "wins", "bags", "order", "approval", "dividend", "buyback", "expansion",
            "outperform", "rally", "gains", "growth", "boost", "robust"]
NEGATIVE = ["misses", "miss estimates", "plunge", "slumps", "falls", "drops", "downgrade", "cuts", "loss",
            "probe", "penalty", "fine", "fraud", "default", "resigns", "weak", "slowdown", "ban", "raid",
            "lawsuit", "concern", "decline", "tumbles", "crash", "warning", "underperform"]
MATERIAL = ["result", "q1", "q2", "q3", "q4", "quarter", "profit", "revenue", "earnings", "guidance", "merger",
            "acquisition", "acquire", "stake", "rbi", "sebi", "penalty", "fine", "downgrade", "upgrade", "rating",
            "ceo", "md ", "resign", "dividend", "buyback", "order", "deal", "contract", "tax", "probe", "fraud",
            "margin", "npa", "asset quality", "capex", "expansion"]
NOISE = ["share price today", "stock price today", "price live", "live updates", "outlook for the week",
         "technical analysis", "stocks to buy", "stocks to watch", "top gainers", "top losers", "intraday"]


# ------------------------------------------------------------------ fetch
def _google_news(query: str, days: int, http=requests) -> list[dict]:
    url = (f"https://news.google.com/rss/search?q={quote(query)}%20when:{days}d"
           "&hl=en-IN&gl=IN&ceid=IN:en")
    response = http.get(url, timeout=20)
    items = []
    for node in ET.fromstring(response.content).findall(".//item"):
        title = (node.findtext("title") or "").strip()
        source = (node.findtext("source") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].strip()
        try:
            published = parsedate_to_datetime(node.findtext("pubDate")).astimezone(timezone.utc)
        except Exception:
            continue
        items.append({"title": title, "source": source or "Google News", "published_utc": published.isoformat(),
                      "link": node.findtext("link")})
    return items


def _yahoo_news(symbol: str) -> list[dict]:
    import yfinance as yf

    items = []
    for raw in yf.Ticker(symbol).news or []:
        content = raw.get("content", raw)
        title = (content.get("title") or "").strip()
        stamp = content.get("pubDate") or content.get("providerPublishTime")
        if not title or not stamp:
            continue
        published = (datetime.fromtimestamp(stamp, timezone.utc) if isinstance(stamp, (int, float))
                     else datetime.fromisoformat(str(stamp).replace("Z", "+00:00")))
        provider = content.get("provider", {})
        items.append({"title": title, "source": provider.get("displayName", "Yahoo Finance") if isinstance(provider, dict) else "Yahoo Finance",
                      "published_utc": published.isoformat(), "link": (content.get("canonicalUrl") or {}).get("url")})
    return items


def fetch_headlines(symbol: str, days: int = LOOKBACK_DAYS, http=requests) -> list[dict]:
    """Recent headlines for one stock, newest first, de-duplicated, capped."""
    names = COMPANY_NAMES.get(symbol, [symbol.replace(".NS", "")])
    headlines = []
    try:
        headlines += _google_news(f'"{names[0]}"', days, http)
    except Exception as e:
        print(f"{symbol}: Google News unavailable ({type(e).__name__})")
    try:
        headlines += _yahoo_news(symbol)
    except Exception as e:
        print(f"{symbol}: Yahoo news unavailable ({type(e).__name__})")

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    seen, unique = set(), []
    for h in sorted(headlines, key=lambda h: h["published_utc"], reverse=True):
        key = re.sub(r"[^a-z0-9]", "", h["title"].lower())[:80]
        if key in seen or datetime.fromisoformat(h["published_utc"]) < cutoff:
            continue
        seen.add(key)
        unique.append(h)
    return unique[:MAX_HEADLINES]


# ------------------------------------------------------------------ score
def keyword_score(title: str, symbol: str) -> dict:
    """Offline fallback: crude, conservative, and never the only line of
    defence -- it exists so the daily run keeps going without an LLM."""
    text = f" {title.lower()} "
    names = [n.lower() for n in COMPANY_NAMES.get(symbol, [symbol.replace('.NS', '')])]
    relevant = any(n in text for n in names)
    noise = any(n in text for n in NOISE)
    pos = sum(w in text for w in POSITIVE)
    neg = sum(w in text for w in NEGATIVE)
    sentiment = 0.0 if pos == neg else max(-1.0, min(1.0, 0.5 * (pos - neg)))
    material = 0.0 if (noise or not relevant) else (0.7 if any(w in text for w in MATERIAL) else 0.25)
    return {"relevant": relevant, "sentiment": sentiment, "material": material, "scorer": "keywords"}


def _llm_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    from groq import Groq
    return Groq(api_key=key, timeout=45, max_retries=1)


def llm_scores(headlines: list[dict], company: str, client=None) -> list[dict] | None:
    """One batched call per stock. Returns None (caller falls back) on any
    failure -- a bad response must never become a fake signal."""
    client = client if client is not None else _llm_client()
    if client is None or not headlines:
        return None
    lines = "\n".join(f"{i}: {h['title']}" for i, h in enumerate(headlines))
    prompt = (
        f"You are an equity analyst covering {company} (listed on NSE, India). For each headline decide:\n"
        f"- relevant: is it actually about {company} (not a different company, not a generic market story)?\n"
        "- sentiment: -1 (clearly bad for the stock) .. 1 (clearly good); 0 if neutral\n"
        "- material: 0..1, how much it should change a long-term investor's view. Results, guidance, "
        "regulatory action, deals, management changes, rating changes are material. Price recaps, "
        "'share price today', technical outlooks, and 'stocks to buy' lists are not (near 0).\n"
        'Return JSON only: {"items":[{"i":int,"relevant":bool,"sentiment":number,"material":number}]}\n\n'
        f"{lines}"
    )
    for model in LLM_MODELS:
        try:
            extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
            response = client.chat.completions.create(
                model=model, temperature=0, max_tokens=4000, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}], **extra)
            parsed = json.loads(response.choices[0].message.content)["items"]
            by_index = {int(item["i"]): item for item in parsed}
            scored = []
            for i in range(len(headlines)):
                item = by_index.get(i)
                if item is None:
                    return None
                scored.append({
                    "relevant": bool(item["relevant"]),
                    "sentiment": max(-1.0, min(1.0, float(item["sentiment"]))),
                    "material": max(0.0, min(1.0, float(item["material"]))),
                    "scorer": model,
                })
            return scored
        except Exception as e:
            print(f"{company}: {model} scoring failed ({type(e).__name__}); trying next")
    return None


STOPWORDS = set("""the and for with from its this that after over amid into are has have will may what why
how says said new stock stocks share shares crore lakh today india indian ltd limited company firm firms per
than more about gets get sees set amid could would their they you your all out its via""".split())
SAME_STORY = 0.5               # share of the smaller headline's key words two headlines must have in common
COVERAGE_BOOST = 0.2           # each extra outlet carrying a story adds 20% weight, up to 3 extra


def _key_words(title: str, symbol: str | None) -> set[str]:
    company = {w for name in COMPANY_NAMES.get(symbol, []) for w in name.lower().split()}
    return {w for w in re.findall(r"[a-z0-9]+", title.lower())
            if len(w) >= 3 and w not in STOPWORDS and w not in company}


def stories(scored: list[dict], symbol: str | None = None) -> list[dict]:
    """Groups relevant, material headlines into stories. News is syndicated:
    one ICRA report or one board approval shows up as six near-identical
    headlines, and counting each separately would let whichever story was
    copied most drown out the rest. Two headlines are the same story when
    at least half the key words of the shorter one are shared; grouping is
    transitive. Returns one entry per story: the most material headline as
    its title, the materiality-weighted sentiment, its freshest timestamp,
    and how many headlines carried it."""
    items = [h for h in scored if h.get("relevant") and h.get("material", 0) > 0]
    words = [_key_words(h.get("title", ""), symbol) for h in items]
    groups: list[list[int]] = []
    for i, w in enumerate(words):
        matched = [g for g in groups if any(len(w & words[j]) >= SAME_STORY * max(min(len(w), len(words[j])), 1)
                                            and w & words[j] for j in g)]
        merged = [i] + [j for g in matched for j in g]
        groups = [g for g in groups if g not in matched] + [merged]
    out = []
    for g in groups:
        members = [items[j] for j in g]
        lead = max(members, key=lambda h: (h["material"], h["published_utc"]))
        weight = sum(h["material"] for h in members)
        out.append({"title": lead.get("title", ""), "source": lead.get("source"),
                    "sentiment": round(sum(h["material"] * h["sentiment"] for h in members) / weight, 3),
                    "material": max(h["material"] for h in members),
                    "published_utc": max(h["published_utc"] for h in members), "count": len(members)})
    return sorted(out, key=lambda s: (s["material"], s["published_utc"]), reverse=True)


def news_signal(scored: list[dict], now: datetime | None = None, half_life_days: float = 1.5,
                symbol: str | None = None) -> dict:
    """Collapses scored headlines into one number per stock.

    Headlines are first grouped into stories, so a story counts once however
    many outlets copied it (wider coverage adds a little weight). Each story
    is weighted by materiality and by how recent it is (half-life 1.5 days),
    so one fresh results story outweighs ten price recaps. `strength` says
    how much material news there was at all: a strong signal from one weak
    story should not move money.
    """
    now = now or datetime.now(timezone.utc)
    weighted, weight_total, material_count = 0.0, 0.0, 0
    for story in stories(scored, symbol):
        age_days = max((now - datetime.fromisoformat(story["published_utc"])).total_seconds() / 86400, 0.0)
        coverage = 1 + COVERAGE_BOOST * min(story["count"] - 1, 3)
        w = story["material"] * 0.5 ** (age_days / half_life_days) * coverage
        weighted += w * story["sentiment"]
        weight_total += w
        material_count += story["material"] >= 0.5
    score = weighted / weight_total if weight_total > 0 else 0.0
    strength = min(weight_total / 2.0, 1.0)   # ~2 fresh, fully material stories = full strength
    return {"score": round(score, 3), "strength": round(strength, 3), "material_count": material_count,
            "view": round(score * strength, 3)}


def gather(symbol: str, on: date | None = None, http=requests, client=None, refresh: bool = False) -> dict:
    """Fetch, score and summarise one stock's news for `on` (default today),
    cached per day so the scoring call happens once."""
    on = on or date.today()
    path = NEWS_DIR / f"{on.isoformat()}_{symbol.replace('.NS', '')}.json"
    if path.exists() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        cached["signal"] = news_signal(cached["headlines"], symbol=symbol)   # scoring rules may have changed
        return cached

    company = COMPANY_NAMES.get(symbol, [symbol])[0]
    headlines = fetch_headlines(symbol, http=http)
    scores = llm_scores(headlines, company, client) or [keyword_score(h["title"], symbol) for h in headlines]
    scored = [{**h, **s} for h, s in zip(headlines, scores)]
    result = {"symbol": symbol, "date": on.isoformat(), "headlines": scored, "signal": news_signal(scored, symbol=symbol),
              "scorer": scored[0]["scorer"] if scored else None,
              "fetched_utc": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
