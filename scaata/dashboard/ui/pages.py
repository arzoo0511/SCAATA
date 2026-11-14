"""The dashboard's pages. Each is a function that reads the agent's saved
state and draws it; none can place an order, change a setting or write a file.

    Now       -- is it beating holding, and why or why not
    Mind      -- how each day's decisions were reached (interactive replay)
    Learning  -- what it has judged about its own advisors
    News      -- what it read
    Evidence  -- backtests and studies, including the ones that failed
    Lab       -- experiments filed in advance, run on the real brain, and reviewed (ui/lab.py)
    Macroverse -- real crises replayed, and synthetic shocks, through the real brain (ui/macro.py)
    System    -- how the pieces fit, what is live, what is only planned
"""
from __future__ import annotations

import base64
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from scaata.agent.daily import load_journal
from scaata.agent.memory import HORIZON, load_memory
from scaata.config import INDIA_BENCHMARK, INDIA_TICKERS, ROOT_DIR
from scaata.dashboard.ui import charts, data, lab, macro, mind, now
from scaata.dashboard.ui import theme as T
from scaata.live.paper_book import load_book, summary as book_summary

PPO_ROOT = ROOT_DIR / "results" / "phase1_rerun_india"
PPO_ARMS = {"vanilla": "Original setup", "env_fixes": "With environment fixes"}


def plot(fig) -> None:
    # Keyed by call site: two charts that happen to be identical (e.g. two "no data
    # yet" placeholders on one page) would otherwise share an auto-generated id.
    st.plotly_chart(fig, use_container_width=True, config=charts.CONFIG, key=f"chart_{sys._getframe(1).f_lineno}")


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(symbols: tuple[str, ...]) -> pd.DataFrame:
    """Daily closes for the last year. Cached 15 min: Yahoo rate-limits hard."""
    import yfinance as yf

    frame = yf.download(list(symbols), period="1y", progress=False, auto_adjust=True)["Close"]
    return frame.to_frame() if isinstance(frame, pd.Series) else frame


def _state():
    return load_book(), load_journal(), load_memory(), data.read_json(data.EVALUATION_PATH)


# ------------------------------------------------------------------- Now
def now_page() -> None:
    book, journal, _, evaluation = _state()
    if book is None or not book.get("equity_history"):
        T.intro("Paper book", "HANSEI has not traded yet.",
                "The first scheduled run after an NSE close opens the book with ₹10,000 of fake money.")
        return
    components.html(now.cockpit_html(data.now_payload(book, journal, evaluation)), height=now.HEIGHT,
                    scrolling=False)

    T.heading("When it stepped back", "each cell: share of a stock's slot it held that day")
    plot(charts.exposure_map(data.decisions_frame(journal)))
    T.caption("Dark cells are days it was out of a stock. Triangles and diamonds mark the days it decided to "
              "trade. Open <b>Mind</b> to replay any of these days and see why.")

    trades = data.trades_frame(book)
    if len(trades):
        with st.expander(f"Every trade so far ({len(trades)})"):
            table = trades[["date", "symbol", "action", "qty", "price", "cost", "reason"]].copy()
            table["symbol"] = table["symbol"].map(data.short)
            table["date"] = table["date"].dt.date
            st.dataframe(table.iloc[::-1], use_container_width=True, hide_index=True,
                         column_config={"price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                                        "cost": st.column_config.NumberColumn("Cost", format="₹%.2f")})

    _market()


def _market() -> None:
    T.heading("The market", "five stocks and the Nifty 50, rebased to 100")
    window = st.segmented_control("Window", ["1M", "3M", "6M", "1Y"], default="3M", label_visibility="collapsed",
                                  key="market_window") or "3M"
    try:
        prices = load_prices(tuple(INDIA_TICKERS + [INDIA_BENCHMARK]))
    except Exception as e:  # Yahoo rate limits are routine, not a page-breaking failure
        st.markdown(f'<div class="hs-note"><b>Live prices are unavailable right now</b> ({type(e).__name__}). '
                    "Yahoo rate-limits; they return on a later refresh. Everything above comes from HANSEI&rsquo;s "
                    "own saved records and is unaffected.</div>", unsafe_allow_html=True)
        return
    if prices is None or prices.dropna(how="all").empty:
        st.markdown('<div class="hs-note">Yahoo returned no prices. They return on a later refresh.</div>',
                    unsafe_allow_html=True)
        return
    plot(charts.market(prices, {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}[window]))
    T.foot(f"Daily closes, split and dividend adjusted, via Yahoo Finance; cached 15 minutes. "
           f"Latest bar {prices.index[-1].date()}.")


# ------------------------------------------------------------------ Mind
def mind_page() -> None:
    _, journal, _, _ = _state()
    T.intro("How it thinks", "Replay any day it has lived.",
            "Four advisors each look at one kind of evidence and give every stock a view from −1 (get out) to +1 "
            "(own it). The brain weighs each view by how far it has learned to trust that advisor. It holds by "
            "default and trades only when the weighted evidence clearly disagrees with what it owns. "
            "<b>Press play</b>, drag the timeline, or click a stock.")
    if not journal:
        st.markdown('<div class="hs-note">HANSEI has not run yet, so there is nothing to replay.</div>',
                    unsafe_allow_html=True)
        return
    components.html(mind.explorer_html(data.mind_payload(journal)), height=mind.HEIGHT, scrolling=True)
    T.caption("Everything shown is what the journal recorded on the day: the views, the trust, the reasons and "
              "the headlines as they were then. Nothing is recomputed with hindsight. The tan tick on each "
              "stock&rsquo;s bar is what it held; the bar is what it wanted.")

    decisions = data.decisions_frame(journal)
    T.heading("One stock, every day", "each advisor's view over time")
    symbols = list(dict.fromkeys(decisions["symbol"]))
    choice = st.segmented_control("Stock", symbols, default=symbols[0], format_func=data.short,
                                  label_visibility="collapsed", key="mind_stock") or symbols[0]
    plot(charts.views_map(decisions, choice))
    T.caption("Blue: that advisor wanted to own the stock that day. Red: it wanted out. Dark: no opinion. "
              "A stock is cut only when the trust-weighted sum goes negative.")

    T.heading("The rules it cannot break")
    rules = [("Hold by default", "Starts fully invested; positive evidence never adds leverage."),
             ("Act only on a clear gap", "A trade needs wanted and held exposure to differ by at least a third of "
                                         "the stock&rsquo;s share."),
             ("Wait between trades", "10 sessions between trades in one stock, unless material news is strong "
                                     "(view of ±0.5 or more)."),
             ("Rebalance on drift", "A holding more than 20% away from an equal share is brought back."),
             ("Decide after the close, fill at the next open", "It never trades at the price that produced the "
                                                               "decision.")]
    st.markdown('<div class="hs-panel">' + "".join(
        f'<div class="hs-row"><div><div class="a">{a}</div><div class="b">{b}</div></div></div>' for a, b in rules)
        + "</div>", unsafe_allow_html=True)


# -------------------------------------------------------------- Learning
def learning_page() -> None:
    _, journal, memory, evaluation = _state()
    trust, lessons, beliefs = data.trust_frame(journal), data.lessons_frame(memory), data.beliefs_frame(memory)
    judged = int(beliefs["judged"].sum()) if len(beliefs) else 0
    T.intro("Hansei · 反省 · reflecting on one's own mistakes",
            "It writes down what it believed, then checks.",
            f"Every day it records each advisor&rsquo;s view on each stock. <b>{HORIZON} sessions later</b> it "
            "compares that view with what the stock actually did, against a liquid fund, and shifts trust toward "
            "the advisors that were right (the Hedge algorithm). An advisor with no opinion that day is neither "
            "rewarded nor punished.")
    T.figures([("Beliefs on file", f"{len(beliefs)}", "one per stock per day", None),
               ("Judged so far", f"{judged}", f"{len(beliefs) - judged} still too young", None),
               ("Lessons", f"{len(lessons)}", "trust updates", None),
               ("Most trusted", T.ADVISOR_LABELS[max(memory["weights"], key=memory["weights"].get)]
                if memory.get("weights") else "—",
                f"{max(memory['weights'].values()):.0%} of the vote" if memory.get("weights") else "", None)])

    T.heading("Trust over time", "starts equal · 25% each")
    plot(charts.trust_lines(trust, lessons))
    T.caption("Trust stays flat until the first beliefs are old enough to judge. It moves slowly on purpose: in "
              "backtests, faster learning chased noise and traded more.")

    left, right = st.columns(2, gap="large")
    with left:
        T.heading("Each lesson", "how far it moved trust")
        plot(charts.lessons_strip(lessons))
        T.caption("One bar per judged day. Above zero an advisor gained trust from that day&rsquo;s outcome; "
                  "below, it lost.")
    with right:
        T.heading("How often each was right", "live record")
        scores = (evaluation or {}).get("advisors")
        if scores:
            plot(charts.hit_rates(scores))
            T.caption("Circles: judged after 5 sessions. Diamonds: after 20. Hover for the sample size. These "
                      "are a few weeks of five correlated stocks: <b>too little to conclude anything</b>. "
                      "Relative strength is a shadow advisor: scored, never traded on.")
        else:
            T.caption("Hit rates appear once the evaluation has run.")

    T.heading("The belief ledger", "what it thought, and what happened next")
    plot(charts.beliefs_map(beliefs))
    T.caption("Each square is one day&rsquo;s belief about one stock. Filled squares have been judged and are "
              "coloured by the stock&rsquo;s next 20 sessions; hollow ones are waiting. Beliefs are stored as "
              "they were and never edited afterwards.")

    st.markdown(
        '<div class="hs-note"><b>What this does and does not prove.</b> The mechanism is real and runs every day. '
        "Whether it <i>helps</i> is not yet shown: in the 2020–2026 replay, switching learning off changed the "
        "result by almost nothing, and three alternative learners tested no better than equal trust. A "
        "trajectory-based self-critique (replay each advisor alone, score by return minus risk penalties) is in "
        "testing and is not live. See <b>Evidence</b>.</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ News
def news_page() -> None:
    _, journal, _, _ = _state()
    news, decisions = data.news_frame(journal), data.decisions_frame(journal)
    T.intro("What it reads", "Headlines, judged one by one.",
            "After each close it gathers three days of headlines per stock from Google News India and Yahoo "
            "Finance. A language model judges each: is it about the company, does it matter, is it good or bad. "
            "Price-update and listicle noise is dropped, and one story copied by many outlets counts once. "
            "Only stories it judged material are kept below.")
    if news.empty:
        st.markdown('<div class="hs-note">No material stories recorded yet.</div>', unsafe_allow_html=True)
        return
    symbols = list(dict.fromkeys(news["symbol"]))
    chosen = st.pills("Stocks", symbols, selection_mode="multi", default=symbols, format_func=data.short,
                      label_visibility="collapsed", key="news_stocks") or symbols
    part = news[news["symbol"].isin(chosen)]
    good, bad = int((part["sentiment"] > 0.15).sum()), int((part["sentiment"] < -0.15).sum())
    T.figures([("Material stories", f"{len(part)}", f"over {news['session'].nunique()} sessions", None),
               ("Judged good", f"{good}", "sentiment above +0.15", T.UP),
               ("Judged bad", f"{bad}", "sentiment below −0.15", T.DOWN),
               ("Neutral", f"{len(part) - good - bad}", "", T.FLAT)])
    T.heading("Every story it found material", "hover a dot to read it")
    plot(charts.news_bubbles(part))
    T.caption("Bigger dots mattered more or ran in more outlets. Blue is good news for the company, red is bad.")

    T.heading("What the news advisor concluded", "its view per stock, per day")
    plot(charts.news_views(decisions[decisions["symbol"].isin(chosen)]))

    T.heading("Latest headlines", f"after the {journal[-1]['session']} session")
    columns = st.columns(2, gap="large")
    for i, (symbol, digest) in enumerate((s, d) for s, d in journal[-1].get("news", {}).items() if s in chosen):
        heads = "".join(
            f'<div class="hs-head"><span class="hs-sent" style="background:'
            f'{T.UP if h["sentiment"] > .15 else T.DOWN if h["sentiment"] < -.15 else T.FLAT}">'
            f'{h["sentiment"]:+.1f}</span>{h["title"]}<span class="hs-foot"> · {h.get("source") or ""}'
            + (f' and {h.get("count", 1) - 1} more' if h.get("count", 1) > 1 else "") + "</span></div>"
            for h in digest.get("top", [])) or '<div class="hs-foot">Nothing material. The news advisor is silent.</div>'
        columns[i % 2].markdown(
            f'<div class="hs-panel" style="margin-bottom:1rem"><div class="hs-eyebrow">{data.short(symbol)} · view '
            f'{digest.get("signal", {}).get("view", 0):+.2f}</div>{heads}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hs-note"><b>Does reading the news help?</b> Not measurably, so far. On the one honest test '
        "window (Jul 2024 onward, 1,700+ scored opinions) the news view alone was right less than half the time, "
        "and adding it to the agent changed returns by an amount indistinguishable from noise. It stays in "
        "because it is cheap and its live record is still short.</div>", unsafe_allow_html=True)


# -------------------------------------------------------------- Evidence
def evidence_page() -> None:
    report = data.read_json(data.RESULTS / "hansei_backtest.json")
    long_report = data.read_json(data.RESULTS / "hansei_backtest_long.json")
    study = data.read_json(data.RESULTS / "hansei_signal_study.json")
    news_report = data.read_json(data.RESULTS / "hansei_backtest_news.json")
    T.intro("Would it have worked?", "Less risk, yes. More return, not proven.",
            "The same brain replayed day by day on real prices: it decides after each close, fills at the next "
            "open, pays NSE costs, and keeps idle cash in a liquid fund. These are simulations of the past, "
            "<b>not live results</b>, and they are shown with the parts that did not work.")
    if report:
        full = report["periods"].get("Full period")
        if full:
            T.figures([
                ("HANSEI a year", f"{full['hansei']['cagr']:.1%}", f"Sharpe {full['hansei']['sharpe']:.2f}", T.HANSEI),
                ("Holding a year", f"{full['buy_and_hold']['cagr']:.1%}", f"Sharpe {full['buy_and_hold']['sharpe']:.2f}",
                 T.HOLD),
                ("HANSEI worst fall", f"{full['hansei']['max_drawdown']:.0%}", "from its peak", T.HANSEI),
                ("Holding worst fall", f"{full['buy_and_hold']['max_drawdown']:.0%}", "from its peak", T.HOLD),
                ("Chance the Sharpe edge is real", f"{full['prob_sharpe_better']:.0%}", "block bootstrap", None)])
        T.heading("The replay, 2020 to Sep 2026", "drag to zoom · double-click to reset")
        plot(charts.backtest(data.backtest_curve(report)))
        T.caption("Top: ₹10,000 in each. Bottom: how far each had fallen from its own peak. The gap in the "
                  "bottom panel is the edge; the top lines stay close.")

    blocks = data.blocks_frame(long_report)
    left, right = st.columns(2, gap="large")
    with left:
        T.heading("Worst fall, block by block", "2007 onward")
        plot(charts.drawdown_dumbbell(blocks))
        if len(blocks):
            wins = int((blocks["hansei_dd"] > blocks["hold_dd"]).sum())
            T.caption(f"HANSEI&rsquo;s worst fall was smaller than holding&rsquo;s in <b>{wins} of {len(blocks)}</b> "
                      "separate blocks, including 2008.")
    with right:
        T.heading("Return against holding, block by block", "% a year")
        plot(charts.return_bars(blocks))
        if len(blocks):
            wins = int((blocks["cagr_diff"] > 0).sum())
            T.caption(f"It beat holding on return in only <b>{wins} of {len(blocks)}</b> blocks. It tends to win "
                      "when markets fall and lag when they rally hard.")

    left, right = st.columns(2, gap="large")
    with left:
        T.heading("Is it one lucky setting?", "Sharpe edge over holding, 2020–2026")
        if report:
            plot(charts.knob_grid(report.get("grid", [])))
            beat = sum(g["sharpe_diff"] > 0 for g in report.get("grid", []))
            T.caption(f"Positive in <b>{beat} of {len(report.get('grid', []))}</b> combinations of its two tuning "
                      "knobs. The live setting is the middle cell.")
    with right:
        T.heading("Do the advisors predict returns?", "105 NSE large caps · 20 sessions ahead")
        plot(charts.forest(data.signal_frame(study)))
        T.caption("Return of stocks an advisor liked minus those it disliked, with 95% intervals. <b>Every "
                  "interval crosses zero</b>: none is a statistically reliable return signal. The volatility "
                  "brake predicts bigger swings, not lower returns.")

    T.heading("What was tested and turned down")
    learned = no_learning = None
    if report:
        no_learning = next((r for r in report.get("learning", []) if r["learning_rate"] == 0), None)
        learned = next((r for r in report.get("learning", [])
                        if r["learning_rate"] == report["defaults"]["learning_rate"]), None)
    news_line = ""
    if news_report:
        test = next((v for k, v in news_report["periods"].items() if "true test" in k), None)
        signal = next((v for k, v in news_report["news_signal"].items() if "true test" in k), None)
        if test and signal:
            news_line = (f"With news {test['with_news']['cagr_diff']:+.1%} a year against holding, without "
                         f"{test['without_news']['cagr_diff']:+.1%}; the news view alone was right "
                         f"{signal['20_sessions']['hit_rate']:.0%} of the time over "
                         f"{signal['20_sessions']['opinions']:,} opinions.")
    rejected = [
        ("Deep reinforcement learning (PPO)", "off", "Across 9 folds × 5 seeds it trailed buy-and-hold on Sharpe. "
         "This is why HANSEI replaced it with a simple, explainable agent."),
        ("Learning from forecasts", "part",
         (f"Sharpe edge {learned['sharpe_diff']:+.2f} with learning, {no_learning['sharpe_diff']:+.2f} with it "
          "switched off. " if learned and no_learning else "") + "It runs live, but has not been shown to add value."),
        ("News as a return signal", "part", news_line or "No measurable edge on the honest test window."),
        ("Earnings-surprise advisor", "off", "A real but small effect across 104 stocks; as a fifth advisor it "
         "lowered returns in the tuning years. Not added."),
        ("Staying more invested (floor 50%)", "off", "Won in the tuning years, lost to the live rule by about "
         "2.5% a year in the falling 2024–26 window. Not switched on."),
        ("Rotating freed cash into strong stocks", "off", "Made results worse. Not added."),
    ]
    st.markdown('<div class="hs-panel">' + "".join(
        f'<div class="hs-row"><div><div class="a">{a}</div><div class="b">{b}</div></div>'
        f'{T.chip({"off": "rejected", "part": "live · unproven"}[k], k)}</div>' for a, k, b in rejected)
        + "</div>", unsafe_allow_html=True)

    _ppo()
    T.heading("Limits of this evidence")
    limits = ["Five stocks, chosen because the owner already held them: survivorship and selection bias are built "
              "in, and the five move together.",
              "Live history is a few weeks. At the measured day-to-day noise, proving the backtest&rsquo;s return "
              "edge live would take centuries; only the drawdown edge is checkable in a human timescale.",
              "The 2020–2026 replay has no news (no archive existed); the separate news replay starts in 2012 and "
              "its honest window starts mid-2024.",
              "Yahoo prices are adjusted after the fact; fills assume the next open with a fixed cost model and no "
              "market impact, which is fair at ₹10,000 and would not be at size.",
              "Settings were tuned on 2007–2019 and checked once on 2020 onward. Every later look at the same "
              "data weakens that check."]
    st.markdown('<div class="hs-panel">' + "".join(f'<div class="hs-row"><div class="b">{x}</div></div>'
                                                   for x in limits) + "</div>", unsafe_allow_html=True)
    T.foot(f'Full list of risks: <a href="{data.REPO_URL}/blob/main/docs/THREATS_TO_VALIDITY.md" target="_blank">'
           f'docs/THREATS_TO_VALIDITY.md</a>. Reproduce: <code>python -m scaata.agent.backtest</code>, '
           f'<code>--long</code>, <code>--news</code>, and <code>python -m scaata.agent.signal_study</code>.')


def _ppo() -> None:
    with st.expander("Earlier research: the PPO policy HANSEI replaced"):
        T.caption("9 rolling folds × 5 seeds × 5 stocks, every method paying real NSE costs. Folds, not "
                  "(stock, fold) pairs, are the unit of comparison: the five stocks share each window.")
        for column, arm in zip(st.columns(len(PPO_ARMS)), PPO_ARMS):
            summary = data.read_json(PPO_ROOT / arm / "summary.json")
            with column:
                st.markdown(f'<div class="hs-eyebrow">{PPO_ARMS[arm]}</div>', unsafe_allow_html=True)
                if not summary or not summary.get("comparisons"):
                    T.caption("No finished run on this machine.")
                    continue
                c = summary["comparisons"]["ppo_vs_buy_and_hold"]
                means = summary["mean_sharpe_never_traded_as_zero"]
                verdict = ("Beats buy-and-hold" if c["ci_low"] > 0 else "Loses to buy-and-hold" if c["ci_high"] < 0
                           else "Ties with buy-and-hold")
                T.figures([(verdict, f"{c['mean_diff']:+.2f}", f"Sharpe · 95% CI [{c['ci_low']:+.2f}, "
                                                              f"{c['ci_high']:+.2f}]", None)])
                T.caption(f"Mean Sharpe: policy {means['ppo']:+.2f} · buy and hold {means['buy_and_hold']:+.2f} · "
                          f"rule-based {means['rule_based']:+.2f} · momentum {means['momentum_fallback']:+.2f}. "
                          f"Never traded: {summary['ppo_never_traded_rows']} of {summary['rows']} runs.")


lab_page = lab.page


# ------------------------------------------------------------ Macroverse
MACRO_RUNS = ROOT_DIR / "macroverse" / "runs"


SCENARIO_ORDER = ("crude-up-60", "market-down-20", "rupee-down-10", "all-three")


def macro_runs() -> tuple[list[dict], dict | None]:
    """Every saved Macroverse run -- replays by date, then scenarios -- plus the sensitivities.

    Reads the saved files only. It must not import `scaata.macroverse.replay` or `.shock`: those
    pull in the backtest engine and scipy, which the hosted dashboard does not install."""
    found = [data.read_json(path) for path in sorted(MACRO_RUNS.glob("*.json"))] if MACRO_RUNS.exists() else []
    replays = sorted((r for r in found if r and r.get("kind") == "replay"), key=lambda r: r["start"])
    scenarios = sorted((r for r in found if r and r.get("kind") == "scenario"),
                       key=lambda r: (SCENARIO_ORDER.index(r["id"]) if r["id"] in SCENARIO_ORDER else 99, r["id"]))
    return replays + scenarios, data.read_json(MACRO_RUNS / "sensitivities.json")


def macro_page() -> None:
    runs, sens = macro_runs()
    T.intro("Macroverse", "Put HANSEI through a crisis.",
            "<b>Replay history</b> runs a real episode, on real prices, through the same brain the live agent "
            "uses, and lets you watch it day by day. <b>What if</b> applies a shock to the market, crude or the "
            "rupee and shows the range of outcomes. The two are kept apart: one is what happened, the other is a "
            "model&rsquo;s output and is marked <b>synthetic</b> everywhere.")
    if not runs:
        st.markdown('<div class="hs-note"><b>Nothing has been run yet.</b> Run '
                    "<code>python -m scaata.macroverse refresh</code>, then "
                    "<code>python -m scaata.macroverse all</code>.</div>", unsafe_allow_html=True)
        return
    components.html(macro.stage_html(runs, sens), height=macro.HEIGHT, scrolling=False)

    replays = [r for r in runs if r["kind"] == "replay"]
    if replays:
        T.heading("Every crisis at a glance", "worst fall during each episode")
        frame = pd.DataFrame([{"block": r["title"], "hansei_dd": r["stats"]["hansei"]["max_drawdown"],
                               "hold_dd": r["stats"]["holding"]["max_drawdown"],
                               "hansei": r["stats"]["hansei"]["return"], "hold": r["stats"]["holding"]["return"]}
                              for r in replays])
        plot(charts.drawdown_dumbbell(frame))
        less = int((frame["hansei_dd"] > frame["hold_dd"] + 0.005).sum())
        ahead = int((frame["hansei"] > frame["hold"]).sum())
        T.caption(f"HANSEI&rsquo;s worst fall was clearly smaller than holding&rsquo;s in <b>{less} of {len(frame)}"
                  f"</b> episodes, and it finished the episode ahead of holding in <b>{ahead} of {len(frame)}</b>. "
                  "Falling less and ending ahead are different things: it tends to give the difference back by "
                  "being slow to return.")
        table = frame.rename(columns={"block": "Episode", "hansei": "HANSEI return", "hold": "Holding return",
                                      "hansei_dd": "HANSEI worst fall", "hold_dd": "Holding worst fall"})
        columns = ["HANSEI return", "Holding return", "HANSEI worst fall", "Holding worst fall"]
        for column in columns:
            table[column] = table[column] * 100
        st.dataframe(table[["Episode", *columns]], hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(c, format="%.1f%%") for c in columns})
    stamp = runs[0].get("provenance", {})
    T.foot("Prices: the research snapshot <code>" + (stamp.get("prices_sha256") or "")[:12] + "</code> through "
           + str(stamp.get("prices_through")) + ". Factors (Nifty 50, Brent, USD/INR): Yahoo Finance, frozen "
           "snapshot <code>" + (stamp.get("factors_sha256") or "")[:12] + "</code>. Everything on this page was "
           "computed offline by <code>python -m scaata.macroverse all</code> and saved under "
           "<code>macroverse/runs/</code>; it never reads or writes the live paper book, journal or memory.")


# ---------------------------------------------------------------- System
_ARCH = f"""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 430" role="img" aria-label="How HANSEI, its memory, the Research Lab and Macroverse connect"
     font-family="Segoe UI, system-ui, -apple-system, sans-serif">
  <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="{T.MUTED}"/></marker></defs>
  <g fill="none" stroke="{T.MUTED}" stroke-width="1.4" marker-end="url(#ar)">
    <path d="M330,120 L395,120"/><path d="M500,190 L500,236"/>
    <path d="M605,120 L670,120"/><path d="M670,330 L605,330" stroke-dasharray="5 5"/>
    <path d="M835,190 L835,262"/><path d="M395,330 L330,330" stroke-dasharray="5 5"/>
    <path d="M165,262 L165,190" stroke-dasharray="5 5"/>
  </g>
  <g font-size="11" fill="{T.MUTED}">
    <text x="362" y="110" text-anchor="middle">decisions</text><text x="510" y="218">beliefs, outcomes</text>
    <text x="638" y="110" text-anchor="middle">questions</text><text x="638" y="320" text-anchor="middle">findings</text>
    <text x="845" y="230">scenarios</text><text x="362" y="320" text-anchor="middle">proposal</text>
    <text x="175" y="230">only if it passes</text>
  </g>
  <g>
    <rect x="30" y="50" width="300" height="140" rx="14" fill="{T.PANEL}" stroke="{T.HANSEI}" stroke-width="2"/>
    <text x="50" y="82" fill="{T.INK}" font-size="17" font-weight="700">HANSEI · trading mind</text>
    <text x="50" y="106" fill="{T.SOFT}" font-size="12.5">Advisors → brain → paper book.</text>
    <text x="50" y="126" fill="{T.SOFT}" font-size="12.5">Runs after every NSE close.</text>
    <text x="50" y="168" fill="#7BD3A6" font-size="11" font-weight="700" letter-spacing="1.5">LIVE</text>

    <rect x="395" y="50" width="210" height="140" rx="14" fill="{T.PANEL}" stroke="{T.HANSEI}" stroke-width="2"/>
    <text x="415" y="82" fill="{T.INK}" font-size="17" font-weight="700">Journal</text>
    <text x="415" y="106" fill="{T.SOFT}" font-size="12.5">What it saw, believed</text>
    <text x="415" y="126" fill="{T.SOFT}" font-size="12.5">and did, as of that day.</text>
    <text x="415" y="168" fill="#7BD3A6" font-size="11" font-weight="700" letter-spacing="1.5">LIVE · APPEND-ONLY</text>

    <rect x="395" y="240" width="210" height="140" rx="14" fill="{T.PANEL}" stroke="{T.HOLD}" stroke-width="2"/>
    <text x="415" y="272" fill="{T.INK}" font-size="17" font-weight="700">Cognitive memory</text>
    <text x="415" y="296" fill="{T.SOFT}" font-size="12.5">Trust weights and lessons</text>
    <text x="415" y="316" fill="{T.SOFT}" font-size="12.5">today; findings next.</text>
    <text x="415" y="358" fill="{T.TAN}" font-size="11" font-weight="700" letter-spacing="1.5">PARTLY LIVE</text>

    <rect x="670" y="50" width="300" height="140" rx="14" fill="{T.PANEL}" stroke="{T.HANSEI}" stroke-width="2"/>
    <text x="690" y="82" fill="{T.INK}" font-size="17" font-weight="700">Research Lab</text>
    <text x="690" y="106" fill="{T.SOFT}" font-size="12.5">Hypothesis → experiment → devil's</text>
    <text x="690" y="126" fill="{T.SOFT}" font-size="12.5">advocate → recorded finding.</text>
    <text x="690" y="168" fill="#7BD3A6" font-size="11" font-weight="700" letter-spacing="1.5">LIVE · RUN BY HAND</text>

    <rect x="670" y="262" width="300" height="118" rx="14" fill="{T.PANEL}" stroke="{T.HANSEI}" stroke-width="2"/>
    <text x="690" y="294" fill="{T.INK}" font-size="17" font-weight="700">Macroverse</text>
    <text x="690" y="318" fill="{T.SOFT}" font-size="12.5">Replay real crises and shocks</text>
    <text x="690" y="338" fill="{T.SOFT}" font-size="12.5">through the real brain.</text>
    <text x="690" y="364" fill="#7BD3A6" font-size="11" font-weight="700" letter-spacing="1.5">LIVE · COMPUTED OFFLINE</text>

    <rect x="30" y="262" width="300" height="118" rx="14" fill="{T.PANEL}" stroke="{T.TAN}" stroke-width="2"/>
    <text x="50" y="294" fill="{T.INK}" font-size="17" font-weight="700">Promotion gate</text>
    <text x="50" y="318" fill="{T.SOFT}" font-size="12.5">Beats holding in both halves and</text>
    <text x="50" y="338" fill="{T.SOFT}" font-size="12.5">across the settings grid, or it stays out.</text>
    <text x="50" y="364" fill="{T.TAN}" font-size="11" font-weight="700" letter-spacing="1.5">LIVE AS A MANUAL RULE</text>
  </g>
</svg>
"""


def system_page() -> None:
    T.intro("What HANSEI is", "One agent, judged on one thing.",
            "HANSEI manages ₹10,000 of <b>fake money</b> across five NSE stocks its owner already holds. It reads "
            "the charts and the news, trades only when it has a reason to, and counts as working only if it does "
            "better than simply holding those five stocks. Nothing here is investment advice, and no real order "
            "is ever placed.")
    T.heading("How the pieces fit", "solid: running today · dashed: planned")
    # As an image: Streamlit's markdown sanitiser rejects inline SVG with <defs>.
    uri = "data:image/svg+xml;base64," + base64.b64encode(_ARCH.strip().encode("utf-8")).decode()
    st.markdown(f'<div class="hs-panel"><img src="{uri}" style="width:100%;height:auto" alt="The trading '
                'mind writes to a journal and a cognitive memory. A planned Research Lab reads them and records '
                'findings; a planned Macroverse feeds it scenarios. Proposals reach the trading mind only through '
                'the promotion gate."></div>', unsafe_allow_html=True)
    T.caption("Research and simulation never write to the live paper book or its memory. A finding can only "
              "reach the trading mind as a proposal that passes the same gate every past change had to pass.")

    T.heading("What is real today")
    status = [
        ("Daily paper trading", "live", "GitHub Actions runs the agent after each NSE close and commits the new "
         "book, journal and memory. Re-running is safe: a processed session is skipped."),
        ("Four advisors and the brain", "live", "Trend, volatility brake, momentum, news. Hold by default."),
        ("News reading", "live", "Headlines scored by a language model (Groq), with a keyword fallback."),
        ("Memory: beliefs, lessons, trust", "live", "Hedge learning over a 20-session horizon. Working, but not "
         "yet shown to improve results."),
        ("Daily report card", "live", "Splits the gap to holding into exposure and stock choice; tracks each "
         "advisor&rsquo;s hit rate and data health."),
        ("Backtests and signal studies", "live", "2007 onward, tune and holdout splits, a settings grid, block "
         "bootstrap. Run by hand."),
        ("Trajectory self-critique", "part", "Replays each advisor alone and scores the whole path. Passed its "
         "wiring test; awaiting grid validation before it can go live."),
        ("Research Lab", "live", "Experiments filed in advance, run through the real brain on a frozen price "
         "snapshot, and reviewed by code. Run by hand; it cannot change the agent."),
        ("Macroverse", "live", "Six real crises replayed, and four synthetic shocks, through the real brain. "
         "Computed offline; rate, inflation and credit shocks are not available."),
        ("Real-money trading", "off", "Not built and not planned. No broker is connected to this site."),
    ]
    kinds = {"live": "live", "part": "in testing", "plan": "planned", "off": "never"}
    st.markdown('<div class="hs-panel">' + "".join(
        f'<div class="hs-row"><div><div class="a">{a}</div><div class="b">{b}</div></div>{T.chip(kinds[k], k)}</div>'
        for a, k, b in status) + "</div>", unsafe_allow_html=True)

    T.heading("Read more")
    links = [("Source code", data.REPO_URL, "The whole project, including the research that did not work."),
             ("Architecture and roadmap", f"{data.REPO_URL}/blob/main/docs/HANSEI_ARCHITECTURE.md",
              "The audit, the design of the Research Lab and Macroverse, and the build order."),
             ("Threats to validity", f"{data.REPO_URL}/blob/main/docs/THREATS_TO_VALIDITY.md",
              "Every known way these results could be wrong."),
             ("README", f"{data.REPO_URL}#readme", "How to run it yourself.")]
    st.markdown('<div class="hs-panel">' + "".join(
        f'<div class="hs-row"><div><div class="a"><a href="{url}" target="_blank" rel="noopener">{name} ↗</a></div>'
        f'<div class="b">{note}</div></div></div>' for name, url, note in links) + "</div>", unsafe_allow_html=True)
