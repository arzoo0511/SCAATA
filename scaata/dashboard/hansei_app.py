"""HANSEI dashboard -- Indian market (NSE).

Local Streamlit page for HANSEI, the paper-trading agent that manages
₹10,000 of fake money on the five NSE stocks the user owns: HDFC Bank, ICICI
Bank, Petronet LNG, IOC and ITC.

The agent does not live here. It runs by itself after every NSE close (the
`HANSEI-DailyTrader` scheduled task, `python -m scaata.agent.daily`), whether
or not this page is open. This page only reads what it wrote: the paper book,
its journal (decisions, reasons, the news it read), its memory (how far it
trusts each advisor), and the backtest in `results/hansei_backtest.json`.

The real Zerodha account is shown read-only. No button on this page places
an order or sends a message.

Brand: midnight #0F414A, maroon #7F0303, alabaster #EFE8DF, tan #D8BA98,
light blue #96C0CE (see `.streamlit/config.toml` and docs/logo/).

Run with:  streamlit run scaata/dashboard/hansei_app.py
"""
from __future__ import annotations

import base64
import io
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from scaata.config import INDIA_BENCHMARK, INDIA_TICKERS, ROOT_DIR
from scaata.live.kite_broker import KiteError, get_funds, get_holdings, get_profile, load_access_token
from scaata.agent.daily import IST, load_journal
from scaata.agent.memory import HORIZON, load_memory
from scaata.live.paper_book import load_book, summary as book_summary

MID, MAROON, ALAB, TAN, LBLUE = "#0F414A", "#7F0303", "#EFE8DF", "#D8BA98", "#96C0CE"
LOGO = ROOT_DIR / "docs" / "logo" / "hansei_wordmark.png"
# Paths only -- deliberately NOT imported from scaata.evaluation.phase1_rerun,
# which drags in torch/sb3 and made this page take ~a minute to first render.
RESULTS_ROOT = ROOT_DIR / "results" / "phase1_rerun_india"
BACKTEST_PATH = ROOT_DIR / "results" / "hansei_backtest.json"
RUN_TIMES = ((16, 15), (21, 0))     # the scheduled task's weekday triggers, IST
ACTION_STYLE = {"BUY": ("mid", "Buy"), "SELL": ("maroon", "Sell"), "REBALANCE": ("tan", "Rebalance"),
                "HOLD": ("ghost", "Hold")}
ADVISOR_LABELS = {"trend": "Trend", "volatility": "Volatility", "momentum": "Momentum", "news": "News"}
ARMS = {"vanilla": "Original setup", "env_fixes": "With environment fixes"}
NAMES = {
    "HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank", "PETRONET.NS": "Petronet LNG",
    "IOC.NS": "Indian Oil", "ITC.NS": "ITC", INDIA_BENCHMARK: "Nifty 50",
}

st.set_page_config(page_title="HANSEI", page_icon="📉", layout="wide")

st.markdown(f"""
<style>
  .stApp {{ background: {ALAB}; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }}
  h1, h2, h3 {{ color: {MID}; font-weight: 800; letter-spacing: -0.01em; }}
  .hs-rule {{ display: flex; align-items: center; gap: .85rem; margin: 1.9rem 0 1.1rem; }}
  .hs-rule .t {{ text-transform: uppercase; letter-spacing: .18em; font-size: .78rem;
                 font-weight: 800; color: {MID}; white-space: nowrap; }}
  .hs-rule .r {{ flex: 1; height: 3px; background: {TAN}; border-radius: 2px; }}
  .hs-pill {{ display: inline-block; padding: .3rem .8rem; border-radius: 999px; font-size: .7rem;
              font-weight: 800; letter-spacing: .12em; text-transform: uppercase; margin-right: .4rem; }}
  .hs-pill.maroon {{ background: {MAROON}; color: {ALAB}; }}
  .hs-pill.mid {{ background: {MID}; color: {ALAB}; }}
  .hs-pill.ghost {{ background: transparent; color: {MID}; border: 2px solid {MID}; }}
  .hs-pill.tan {{ background: {TAN}; color: {MID}; }}
  .hs-bar {{ position: relative; height: 8px; background: #E9E1D5; border-radius: 4px; margin: 3px 0 7px; }}
  .hs-bar .mid {{ position: absolute; left: 50%; top: -2px; width: 2px; height: 12px; background: #B9A992; }}
  .hs-bar .fill {{ position: absolute; top: 0; height: 8px; border-radius: 4px; }}
  .hs-row {{ display: flex; justify-content: space-between; font-size: .76rem; color: #6B5B4B; }}
  .hs-head {{ padding: 9px 0; border-bottom: 1px solid #E1D8CA; font-size: .88rem; color: {MID}; }}
  .hs-head:last-child {{ border-bottom: none; }}
  .hs-chip {{ display: inline-block; min-width: 3.1rem; text-align: center; padding: .08rem .45rem;
              border-radius: 6px; font-size: .72rem; font-weight: 800; margin-right: .5rem;
              font-variant-numeric: tabular-nums; }}
  .hs-table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
  .hs-table th {{ text-align: right; font-size: .67rem; letter-spacing: .1em; text-transform: uppercase;
                  color: #7C6A58; padding: 6px 8px; border-bottom: 2px solid {TAN}; }}
  .hs-table td {{ text-align: right; padding: 8px; color: {MID}; border-bottom: 1px solid #E1D8CA; }}
  .hs-table th:first-child, .hs-table td:first-child {{ text-align: left; }}
  .hs-card {{ background: #F7F4EE; border: 1px solid #E1D8CA; border-radius: 16px;
              padding: 16px 18px; height: 100%; }}
  .hs-lab {{ text-transform: uppercase; letter-spacing: .13em; font-size: .67rem;
             color: #7C6A58; font-weight: 800; }}
  .hs-num {{ font-size: 1.85rem; font-weight: 800; color: {MID}; line-height: 1.15;
             font-variant-numeric: tabular-nums; }}
  .hs-sub {{ font-size: .82rem; color: #6B5B4B; font-variant-numeric: tabular-nums; }}
  .hs-up {{ color: {MID}; font-weight: 800; }}
  .hs-down {{ color: {MAROON}; font-weight: 800; }}
  .hs-note {{ background: #F7F4EE; border-left: 6px solid {MAROON}; border-radius: 0 12px 12px 0;
              padding: 14px 18px; color: {MID}; }}
  .hs-foot {{ color: #7C6A58; font-size: .78rem; }}
  div[data-testid="stDataFrame"] {{ border: 1px solid #E1D8CA; border-radius: 14px; overflow: hidden; }}
  div[data-testid="stProgress"] > div > div > div > div {{ background-color: {MAROON}; }}
</style>
""", unsafe_allow_html=True)


def rule(title: str) -> None:
    st.markdown(f'<div class="hs-rule"><span class="t">{title}</span><span class="r"></span></div>',
                unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(symbols: tuple[str, ...]) -> pd.DataFrame:
    """Daily closes for the last year. Cached 15 min so a page refresh doesn't
    re-hit Yahoo (which rate-limits hard)."""
    import yfinance as yf

    data = yf.download(list(symbols), period="1y", progress=False, auto_adjust=True)["Close"]
    return data.to_frame() if isinstance(data, pd.Series) else data


@st.cache_data(ttl=900, show_spinner=False)
def sparkline(values: tuple[float, ...], rising: bool) -> str:
    """A brand-coloured sparkline as an inline data URI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colour = MID if rising else MAROON
    fig = plt.figure(figsize=(3.1, 0.72), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.plot(values, color=colour, linewidth=2.0, solid_capstyle="round")
    ax.fill_between(range(len(values)), values, min(values), color=colour, alpha=0.10)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


@st.cache_data(ttl=300, show_spinner=False)
def equity_chart(rows: tuple[tuple[str, float, float], ...]) -> str:
    """Agent vs buy-and-hold equity, in brand colours, as a data URI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = [r[0] for r in rows]
    fig = plt.figure(figsize=(9, 2.6), dpi=110)
    ax = fig.add_axes([0.06, 0.16, 0.92, 0.78])
    ax.plot(dates, [r[1] for r in rows], color=MID, linewidth=2.4, label="HANSEI")
    ax.plot(dates, [r[2] for r in rows], color=TAN, linewidth=2.4, label="Buy & hold")
    ax.set_facecolor("none"); fig.patch.set_alpha(0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#D9CDBC")
    ax.tick_params(colors="#7C6A58", labelsize=8)
    if len(dates) > 8:
        ax.set_xticks(dates[:: max(1, len(dates) // 6)])
    ax.grid(axis="y", color="#E3D9CA", linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, labelcolor=MID, loc="upper left")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def pct(series: pd.Series, days: int) -> float | None:
    s = series.dropna()
    return None if len(s) <= days else float(s.iloc[-1] / s.iloc[-1 - days] - 1)


def move(value: float | None) -> str:
    if value is None:
        return '<span class="hs-sub">n/a</span>'
    cls, arrow = ("hs-up", "▲") if value >= 0 else ("hs-down", "▼")
    return f'<span class="{cls}">{arrow} {abs(value):.2%}</span>'


def run_state() -> list[dict]:
    """Progress of the walk-forward evaluation, per arm."""
    rows = []
    for arm in ARMS:
        directory = RESULTS_ROOT / arm
        jobs = sorted(directory.glob("fold*_seed*.pkl"))
        manifest, summary_path = directory / "manifest.json", directory / "summary.json"
        expected = None
        if manifest.exists():
            config = json.loads(manifest.read_text(encoding="utf-8"))["config"]
            expected = len(config["folds"]) * len(config["seeds"])
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
        rows.append({"arm": arm, "done": len(jobs), "expected": expected, "summary": summary})
    return rows


def next_run(now: datetime) -> datetime:
    """The scheduled task's next weekday trigger after `now`, in IST."""
    local = now.astimezone(IST)
    for offset in range(8):
        day = local.date() + timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        for hour, minute in RUN_TIMES:
            when = datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST)
            if when > local:
                return when
    return local


def view_bar(label: str, value: float, right: str | None = None) -> str:
    """A centred bar: right of the line is supportive, left is against."""
    value = 0.0 if abs(value) < 0.005 else value      # no "-0.00"
    width = min(abs(value), 1.0) * 50
    left = 50 if value >= 0 else 50 - width
    colour = MID if value >= 0 else MAROON
    return (f'<div class="hs-row"><span>{label}</span><span>{right or f"{value:+.2f}"}</span></div>'
            f'<div class="hs-bar"><span class="fill" style="left:{left}%;width:{width}%;background:{colour}"></span>'
            f'<span class="mid"></span></div>')


def chip(sentiment: float) -> str:
    colour, text = ((MID, ALAB) if sentiment > 0.15 else (MAROON, ALAB) if sentiment < -0.15 else (TAN, MID))
    return f'<span class="hs-chip" style="background:{colour};color:{text}">{sentiment:+.1f}</span>'


def card(label: str, value: str, sub: str = "", colour: str = MID) -> str:
    return (f'<div class="hs-card"><div class="hs-lab">{label}</div>'
            f'<div class="hs-num" style="color:{colour}">{value}</div><div class="hs-sub">{sub}</div></div>')


def outlets(story: dict) -> str:
    extra = story.get("count", 1) - 1
    return (story.get("source") or "") + (f" and {extra} more" if extra > 0 else "")


def short(symbol: str) -> str:
    return NAMES.get(symbol, symbol.replace(".NS", ""))


book = load_book()
journal = load_journal()
memory = load_memory()
latest = journal[-1] if journal else None
now = datetime.now(timezone.utc)

# ---------------------------------------------------------------- header
head_left, head_right = st.columns([3, 2], vertical_alignment="center")
with head_left:
    if LOGO.exists():
        st.image(str(LOGO), width=380)
    else:
        st.title("HANSEI")
with head_right:
    st.markdown(
        '<div style="text-align:right">'
        '<span class="hs-pill mid">Paper agent live</span>'
        '<span class="hs-pill ghost">Fake money</span>'
        f'<div class="hs-sub" style="margin-top:.5rem">NSE · {now.astimezone(IST).strftime("%d %b %Y · %H:%M IST")}'
        f'</div></div>', unsafe_allow_html=True)

last_session = book.get("last_session") if book else None
if last_session:
    ran = datetime.fromisoformat(book["last_run_utc"]).astimezone(IST).strftime("%d %b, %H:%M IST")
    status = (f"Last decided after the <b>{date.fromisoformat(last_session):%d %b}</b> session (ran {ran}). "
              f"Next run <b>{next_run(now):%a %d %b, %H:%M IST}</b>.")
    stale = (now.astimezone(IST).date() - date.fromisoformat(last_session)).days > 4
else:
    status, stale = "It has not run yet.", False
overdue = ("<br><b>It looks overdue.</b> If the market has been open since, check "
           "<code>logs/hansei_daily.log</code>." if stale else "")
st.markdown(
    '<div class="hs-note"><b>HANSEI is trading ₹10,000 of fake money on your five stocks.</b> It runs on its own '
    'after every NSE close, so this page does not need to be open. It trades only when it has a reason to, and '
    f"fills at the next morning&rsquo;s open. {status}{overdue}</div>", unsafe_allow_html=True)

# ------------------------------------------------------------ paper book
rule("Paper book · holding vs HANSEI")
if book is None:
    st.markdown(card("Not started", "—", "The first scheduled run opens the book."), unsafe_allow_html=True)
else:
    s = book_summary(book)
    edge_colour = MID if s["vs_benchmark"] >= 0 else MAROON
    cards = [("HANSEI", f"₹{s['equity']:,.0f}", f"{s['return']:+.2%} since start", MID),
             ("Just holding", f"₹{s['benchmark_equity']:,.0f}",
              f"{s['benchmark_return']:+.2%} · same ₹10,000, equal weight", MID),
             ("HANSEI vs holding", f"{s['vs_benchmark']:+.2%}", f"over {s['days']} session(s)", edge_colour),
             ("Activity", f"{s['trades']} trades", f"{s['holdings']} held · ₹{s['cash']:,.0f} in liquid fund", MID)]
    for col, (label, value, sub, colour) in zip(st.columns(4), cards):
        col.markdown(card(label, value, sub, colour), unsafe_allow_html=True)

    history = book["equity_history"]
    if len(history) > 1:
        rows = tuple((r["date"], r["equity"], r["benchmark_equity"]) for r in history)
        st.markdown(f'<img src="data:image/png;base64,{equity_chart(rows)}" '
                    'style="width:100%;margin-top:1rem"/>', unsafe_allow_html=True)

    if book["pending"]:
        queued = " · ".join(
            f"<b>{short(o['symbol'])}</b> → ₹{o['target_value']:,.0f}" if o["action"] == "TARGET"
            else f"<b>{o['action']} {short(o['symbol'])}</b>" for o in book["pending"])
        st.markdown(f'<div class="hs-sub" style="margin-top:.7rem">Queued for the next open: {queued}</div>',
                    unsafe_allow_html=True)
    if book["positions"]:
        frame = pd.DataFrame([{"Stock": short(sym), "Qty": p["qty"], "Avg": p["avg_price"]}
                              for sym, p in book["positions"].items()])
        st.dataframe(frame, use_container_width=True, hide_index=True,
                     column_config={"Avg": st.column_config.NumberColumn("Avg buy", format="₹%.2f")})
    if book["trades"]:
        with st.expander(f"Every trade so far ({len(book['trades'])})"):
            trades = pd.DataFrame(book["trades"])[["date", "symbol", "action", "qty", "price", "cost", "reason"]]
            trades["symbol"] = trades["symbol"].map(short)
            st.dataframe(trades.iloc[::-1], use_container_width=True, hide_index=True,
                         column_config={"price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                                        "cost": st.column_config.NumberColumn("Cost", format="₹%.2f")})
    st.markdown('<div class="hs-foot" style="margin-top:.6rem">Simulated: real NSE prices and costs (STT, stamp '
                'duty, exchange and SEBI fees, GST, spread), whole shares, no broker. Idle cash earns a liquid-fund '
                "rate, in both legs. Orders fill at the next session&rsquo;s open, never the close that produced "
                "them.</div>", unsafe_allow_html=True)

# ------------------------------------------------------------- decisions
if latest:
    rule(f"Its thinking · after {date.fromisoformat(latest['session']):%d %b}")
    weights = latest["weights"]
    for row_start in range(0, len(latest["decisions"]), 3):
        for col, d in zip(st.columns(3), latest["decisions"][row_start:row_start + 3]):
            style, word = ACTION_STYLE.get(d["action"], ("ghost", d["action"].title()))
            news = latest["news"].get(d["symbol"], {}).get("signal", {})
            bars = "".join(view_bar(f"{ADVISOR_LABELS[k]} · trusted {weights.get(k, 0):.0%}", v)
                           for k, v in d["views"].items())
            why = d["reasons"][0].split(": ", 1)[-1]
            col.markdown(
                f'<div class="hs-card" style="margin-bottom:1rem">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<div><div class="hs-lab">{d["symbol"].replace(".NS", "")}</div>'
                f'<div style="color:{MID};font-weight:800;font-size:1.05rem">{short(d["symbol"])}</div></div>'
                f'<span class="hs-pill {style}" style="margin:0">{word}</span></div>'
                f'<div class="hs-sub" style="margin:.55rem 0 .7rem;min-height:2.4em">{why}</div>'
                f'{bars}'
                f'<div class="hs-row" style="margin-top:.4rem"><span>Overall</span>'
                f'<span><b>{d["score"]:+.2f}</b> → wants {d["target_exposure"]:.0%} of its share</span></div>'
                f'<div class="hs-foot" style="margin-top:.35rem">News: {news.get("material_count", 0)} material '
                f'stor{"y" if news.get("material_count", 0) == 1 else "ies"}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="hs-foot">Each advisor gives a view from −1 (get out) to +1 (own it). The brain weighs '
                'them by how far it has learned to trust each one. By default it holds; it steps away only when '
                'the weighted evidence is clearly against a stock, and it waits 10 sessions between trades in '
                'one stock unless strong news says otherwise.</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------- news
    rule("What the news says")
    news_cols = st.columns(2)
    for i, (symbol, digest) in enumerate(latest["news"].items()):
        signal = digest["signal"]
        heads = "".join(
            f'<div class="hs-head">{chip(h["sentiment"])}{h["title"]}'
            f'<span class="hs-foot"> · {outlets(h)}</span></div>' for h in digest["top"]
        ) or '<div class="hs-sub">Nothing material. Quiet news days leave the news advisor silent.</div>'
        tone = "positive" if signal["score"] > 0.15 else "negative" if signal["score"] < -0.15 else "neutral"
        news_cols[i % 2].markdown(
            f'<div class="hs-card" style="margin-bottom:1rem"><div style="display:flex;justify-content:space-between">'
            f'<div class="hs-lab">{short(symbol)}</div>'
            f'<div class="hs-sub">{tone} · view {signal.get("view", 0):+.2f}</div></div>{heads}</div>',
            unsafe_allow_html=True)
    st.markdown('<div class="hs-foot">Headlines from Google News India and Yahoo Finance over the last three days. '
                'An LLM reads each one and judges whether it is about the company, whether it matters, and '
                'whether it is good or bad; price-update and listicle noise is ignored. The same story copied by '
                'many outlets counts once. Fresh, material news counts most (half-life 1.5 days).</div>', unsafe_allow_html=True)

# -------------------------------------------------------------- learning
rule("What it has learned")
learn_left, learn_right = st.columns([2, 3])
with learn_left:
    bars = "".join(view_bar(ADVISOR_LABELS[k], (w - 0.25) * 4, f"{w:.0%}") for k, w in memory["weights"].items())
    st.markdown(f'<div class="hs-card"><div class="hs-lab">Trust in each advisor</div>'
                f'<div style="height:.6rem"></div>{bars}'
                f'<div class="hs-foot">Starts equal, 25% each. Right of centre: trusted more than at the start.'
                f'</div></div>', unsafe_allow_html=True)
with learn_right:
    lessons = memory.get("lessons", [])
    first_day = min((o["date"] for o in memory.get("observations", [])), default=None)
    if lessons:
        body = "".join(
            f'<div class="hs-head">About {date.fromisoformat(l["about_day"]):%d %b}: '
            f'<b>{ADVISOR_LABELS[l["best"]]}</b> was most right, <b>{ADVISOR_LABELS[l["worst"]]}</b> least. '
            f'<span class="hs-foot">' + ", ".join(f"{short(s)} {r:+.1%}" for s, r in l["stocks"].items())
            + "</span></div>" for l in lessons[-6:][::-1])
    else:
        when = (f"about {HORIZON} sessions after {date.fromisoformat(first_day):%d %b}" if first_day
                else f"{HORIZON} sessions after it starts")
        body = (f'<div class="hs-sub">No lessons yet. Every day it writes down what each advisor thought; '
                f"{HORIZON} sessions later it checks what the stocks actually did and shifts trust toward whoever "
                f"was right. The first check comes {when}.</div>")
    st.markdown(f'<div class="hs-card"><div class="hs-lab">Recent lessons</div>'
                f'<div style="height:.4rem"></div>{body}</div>', unsafe_allow_html=True)

# -------------------------------------------------------------- backtest
if BACKTEST_PATH.exists():
    report = json.loads(BACKTEST_PATH.read_text(encoding="utf-8"))
    rule("Would it have worked? · 2020 to now")
    table = "".join(
        f'<tr><td>{label}</td><td>{r["hansei"]["cagr"]:.1%}</td><td>{r["buy_and_hold"]["cagr"]:.1%}</td>'
        f'<td>{r["hansei"]["sharpe"]:.2f}</td><td>{r["buy_and_hold"]["sharpe"]:.2f}</td>'
        f'<td>{r["hansei"]["max_drawdown"]:.0%}</td><td>{r["buy_and_hold"]["max_drawdown"]:.0%}</td>'
        f'<td>{r["trades"]}</td><td>{r["prob_sharpe_better"]:.0%}</td></tr>'
        for label, r in report["periods"].items())
    st.markdown(
        '<div class="hs-card"><table class="hs-table"><tr><th>Period</th><th>HANSEI / yr</th><th>Holding / yr</th>'
        '<th>HANSEI Sharpe</th><th>Holding Sharpe</th><th>HANSEI worst fall</th><th>Holding worst fall</th>'
        f'<th>Trades</th><th>Chance it is real</th></tr>{table}</table></div>', unsafe_allow_html=True)
    curve = report["curve"]
    rows = tuple(zip(curve["dates"], curve["hansei"], curve["buy_and_hold"]))
    st.markdown(f'<img src="data:image/png;base64,{equity_chart(rows)}" style="width:100%;margin-top:1rem"/>',
                unsafe_allow_html=True)
    grid = report["grid"]
    beat = sum(g["sharpe_diff"] > 0 for g in grid)
    no_learning = next((r for r in report["learning"] if r["learning_rate"] == 0), None)
    learned = next((r for r in report["learning"] if r["learning_rate"] == report["defaults"]["learning_rate"]), None)
    learning_line = (f" With learning switched off the Sharpe edge is {no_learning['sharpe_diff']:+.2f}, against "
                     f"{learned['sharpe_diff']:+.2f} with it: on prices alone, learning has not yet added much. Its "
                     "job live is to find out how much the news is worth." if no_learning and learned else "")
    st.markdown(
        f'<div class="hs-foot" style="margin-top:.4rem">The same brain replayed day by day on real prices: decides '
        f"after each close, fills at the next open, pays NSE costs, idle cash in a liquid fund. Its edge is mostly "
        f"<b>smaller falls</b>, not bigger gains. It beat holding on Sharpe in <b>{beat} of {len(grid)}</b> "
        f"settings of its tuning knobs, so the result is not one lucky setting.{learning_line} &ldquo;Chance it is "
        f"real&rdquo; is a block-bootstrap probability that its Sharpe beats holding over the same days. News is "
        f"not in this replay: there is no archive of past headlines.</div>", unsafe_allow_html=True)

# --------------------------------------------------------------- account
rule("Your real Zerodha account · read-only")
if load_access_token() is None:
    st.markdown(
        '<div class="hs-card"><div class="hs-lab">Not connected</div>'
        '<div class="hs-num" style="font-size:1.15rem;margin-top:.35rem">Kite session expired</div>'
        '<div class="hs-sub" style="margin-top:.35rem">Tokens reset at 06:00 IST. Run '
        '<code>python -m scaata.live.kite_login</code> in a terminal, then reload.</div></div>',
        unsafe_allow_html=True,
    )
else:
    try:
        profile, _ = get_profile()
        funds, _ = get_funds()
        holdings, _ = get_holdings()
        if profile:
            st.markdown(f'<span class="hs-pill mid">{profile["user_name"]} · {profile["user_id"]}</span>',
                        unsafe_allow_html=True)
        if funds:
            invested = sum(h["qty"] * h["last_price"] for h in holdings)
            pnl = sum(h["pnl"] for h in holdings)
            cards = [("Usable margin", f"₹{funds['net']:,.0f}", ""), ("Cash", f"₹{funds['cash']:,.0f}", ""),
                     ("Invested", f"₹{invested:,.0f}", f"{len(holdings)} holdings"),
                     ("Unrealised P&L", f"₹{pnl:,.0f}", "on holdings")]
            for col, (label, value, sub) in zip(st.columns(4), cards):
                colour = MAROON if value.startswith("₹-") else MID
                col.markdown(
                    f'<div class="hs-card"><div class="hs-lab">{label}</div>'
                    f'<div class="hs-num" style="color:{colour}">{value}</div>'
                    f'<div class="hs-sub">{sub}</div></div>', unsafe_allow_html=True)
        if holdings:
            frame = pd.DataFrame(holdings)
            frame["value"] = frame["qty"] * frame["last_price"]
            st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
            st.dataframe(
                frame[["symbol", "qty", "avg_price", "last_price", "value", "pnl"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "symbol": st.column_config.TextColumn("Stock"),
                    "qty": st.column_config.NumberColumn("Qty", format="%g"),
                    "avg_price": st.column_config.NumberColumn("Avg", format="₹%.2f"),
                    "last_price": st.column_config.NumberColumn("Last", format="₹%.2f"),
                    "value": st.column_config.NumberColumn("Value", format="₹%.0f"),
                    "pnl": st.column_config.NumberColumn("P&L", format="₹%.0f"),
                },
            )
    except KiteError as e:
        st.markdown(f'<div class="hs-note">Kite: {e}<br>The daily token may have expired — run '
                    '<code>python -m scaata.live.kite_login</code>.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- market
rule("Watchlist")
symbols = tuple(INDIA_TICKERS + [INDIA_BENCHMARK])
try:
    prices = load_prices(symbols)
except Exception as e:  # yfinance rate limits are routine, not a page-breaking failure
    prices = None
    st.markdown(f'<div class="hs-note">Price data unavailable right now ({type(e).__name__}). '
                'Yahoo rate-limits; it will return on the next refresh.</div>', unsafe_allow_html=True)

if prices is not None:
    available = [s for s in symbols if s in prices and not prices[s].dropna().empty]
    for row_start in range(0, len(available), 3):
        for col, symbol in zip(st.columns(3), available[row_start:row_start + 3]):
            series = prices[symbol].dropna()
            day, month, half = pct(series, 1), pct(series, 21), pct(series, 126)
            tail = tuple(float(v) for v in series.tail(126))
            image = sparkline(tail, rising=(half or 0) >= 0)
            col.markdown(
                f'<div class="hs-card" style="margin-bottom:1rem">'
                f'<div class="hs-lab">{symbol.replace(".NS", "")}</div>'
                f'<div style="color:{MID};font-weight:700;font-size:.95rem;margin:.1rem 0 .3rem">'
                f'{NAMES.get(symbol, symbol)}</div>'
                f'<div class="hs-num">₹{series.iloc[-1]:,.2f}</div>'
                f'<div style="margin:.35rem 0 .1rem">{move(day)}<span class="hs-sub"> today</span></div>'
                f'<img src="data:image/png;base64,{image}" style="width:100%;margin:.3rem 0 .2rem"/>'
                f'<div style="display:flex;gap:1.4rem">'
                f'<div><span class="hs-lab">1 month</span><br>{move(month)}</div>'
                f'<div><span class="hs-lab">6 months</span><br>{move(half)}</div></div>'
                f'</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hs-foot">Daily closes, split and dividend adjusted, via Yahoo Finance. '
                f'Cached 15 minutes. Latest bar {prices.index[-1].date()}.</div>', unsafe_allow_html=True)

# -------------------------------------------------------------- research
rule("Earlier research")
with st.expander("The PPO policy HANSEI replaced: walk-forward evaluation"):
    st.markdown(
        '<div class="hs-foot">9 rolling folds × 5 seeds × 5 stocks. Every method pays real NSE costs — STT, '
        'stamp duty, exchange and SEBI fees, GST, plus spread. A policy has to beat buy-and-hold here before '
        'anything trades.</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

    for col, row in zip(st.columns(len(ARMS)), run_state()):
        done, expected, summary = row["done"], row["expected"], row["summary"]
        with col:
            st.markdown(f'<div class="hs-lab">{ARMS[row["arm"]]}</div>', unsafe_allow_html=True)
            if expected:
                st.progress(min(done / expected, 1.0), text=f"{done}/{expected} training jobs")
            else:
                st.markdown('<div class="hs-sub">not started</div>', unsafe_allow_html=True)
            if summary and summary.get("comparisons"):
                comparison = summary["comparisons"]["ppo_vs_buy_and_hold"]
                verdict, colour = (("Beats buy-and-hold", MID) if comparison["ci_low"] > 0
                                   else ("Loses to buy-and-hold", MAROON) if comparison["ci_high"] < 0
                                   else ("Ties with buy-and-hold", MID))
                means = summary["mean_sharpe_never_traded_as_zero"]
                st.markdown(
                    f'<div class="hs-card"><div class="hs-lab" style="color:{colour}">{verdict}</div>'
                    f'<div class="hs-num" style="color:{colour};font-size:1.5rem">'
                    f'{comparison["mean_diff"]:+.2f} Sharpe</div>'
                    f'<div class="hs-sub">95% CI [{comparison["ci_low"]:+.2f}, {comparison["ci_high"]:+.2f}] · '
                    f'ahead in {comparison["share_folds_a_better"]:.0%} of {comparison["n_folds"]} folds</div>'
                    f'<div class="hs-sub" style="margin-top:.5rem">Mean Sharpe — policy {means["ppo"]:+.2f} · '
                    f'buy &amp; hold {means["buy_and_hold"]:+.2f} · rule-based {means["rule_based"]:+.2f} · '
                    f'momentum {means["momentum_fallback"]:+.2f}</div>'
                    f'<div class="hs-sub">Never traded: {summary["ppo_never_traded_rows"]} of '
                    f'{summary["rows"]} runs</div></div>', unsafe_allow_html=True)
            elif done:
                st.markdown('<div class="hs-sub">Running — the verdict appears when every job has finished.</div>',
                            unsafe_allow_html=True)

    st.markdown(
        '<div class="hs-foot" style="margin-top:2rem">A policy that never trades counts as Sharpe 0, the same as '
        'holding cash, rather than being dropped. Folds, not (stock, fold) pairs, are the unit of comparison: the '
        'five stocks share each six-month window, so they are not independent observations.</div>',
        unsafe_allow_html=True)
