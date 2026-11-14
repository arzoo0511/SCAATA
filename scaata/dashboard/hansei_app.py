"""HANSEI dashboard -- Indian market (NSE).

The public face of HANSEI, the paper-trading agent that manages ₹10,000 of
fake money on five NSE stocks: HDFC Bank, ICICI Bank, Petronet LNG, IOC and
ITC.

The agent does not live here. It runs by itself after every NSE close
(`.github/workflows/hansei-daily.yml`, `python -m scaata.agent.daily`),
whether or not this page is open. This app only reads what it wrote: the
paper book, its journal (decisions, reasons, the news it read), its memory
(how far it trusts each advisor), the daily evaluation, and the backtests in
`results/`. No control on any page places an order, changes a setting or
writes a file.

Pages live in `scaata/dashboard/ui/pages.py`; the look in `ui/theme.py`; the
figures in `ui/charts.py`; the interactive replay in `ui/mind.py`.

Run with:  streamlit run scaata/dashboard/hansei_app.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

# Streamlit Cloud runs this file without installing the package, so put the
# repo root on the path before importing scaata.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scaata.agent.daily import IST
from scaata.config import ROOT_DIR
from scaata.dashboard.ui import pages, theme
from scaata.live.paper_book import load_book

ICON = ROOT_DIR / "docs" / "logo" / "hansei_icon.png"
RUN_TIMES = ((16, 15), (21, 0))     # the daily workflow's main weekday triggers, IST
STALE_DAYS = 4                      # no new session for longer than this means the daily run is stuck

st.set_page_config(page_title="HANSEI · a paper-trading agent that checks its own work",
                   page_icon=str(ICON) if ICON.exists() else "反", layout="wide",
                   initial_sidebar_state="collapsed")
theme.inject()


def next_run(now: datetime) -> datetime:
    """The daily workflow's next weekday trigger after `now`, in IST."""
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


def top_bar(now: datetime) -> None:
    book = load_book()
    last = (book or {}).get("last_session")
    left, right = st.columns([3, 2], vertical_alignment="center")
    icon = theme.image_uri(ICON)
    left.markdown(
        '<div class="hs-brand">' + (f'<img src="{icon}" alt="">' if icon else "")
        + '<div><div class="w">HANSEI <span style="font-weight:400;opacity:.55">反省</span></div>'
        '<div class="k">A paper-trading agent that checks its own work</div></div></div>', unsafe_allow_html=True)
    if last:
        stale = (now.astimezone(IST).date() - date.fromisoformat(last)).days > STALE_DAYS
        state = ('<span class="hs-live"><span class="hs-dot stale"></span>Daily run overdue</span>' if stale else
                 '<span class="hs-live"><span class="hs-dot"></span>Paper agent live</span>')
        detail = (f"last session {date.fromisoformat(last):%d %b %Y}<br>"
                  f"next run {next_run(now):%a %d %b · %H:%M} IST · fake money")
    else:
        state = '<span class="hs-live"><span class="hs-dot stale"></span>Not started</span>'
        detail = "the first run after an NSE close opens the book"
    right.markdown(f'<div class="hs-status">{state}<br>{detail}</div>', unsafe_allow_html=True)


PAGES = [
    st.Page(pages.now_page, title="Now", url_path="now", default=True),
    st.Page(pages.mind_page, title="Mind", url_path="mind"),
    st.Page(pages.learning_page, title="Learning", url_path="learning"),
    st.Page(pages.news_page, title="News", url_path="news"),
    st.Page(pages.evidence_page, title="Evidence", url_path="evidence"),
    st.Page(pages.lab_page, title="Lab", url_path="lab"),
    st.Page(pages.macro_page, title="Macroverse", url_path="macroverse"),
    st.Page(pages.system_page, title="System", url_path="system"),
]

current = st.navigation(PAGES, position="hidden")
top_bar(datetime.now(timezone.utc))
for column, page in zip(st.columns(len(PAGES)), PAGES):
    if page.title == current.title:
        column.markdown(f'<div class="hs-nav-on">{page.title}</div>', unsafe_allow_html=True)
    else:
        column.page_link(page, label=page.title, use_container_width=True)

current.run()

st.markdown(
    '<div class="hs-foot" style="margin-top:3rem;border-top:1px solid #1B4650;padding-top:1rem">'
    "HANSEI trades fake money on real prices. It is a research project, not investment advice. "
    f'<a href="https://github.com/arzoo0511/SCAATA" target="_blank" rel="noopener">Source on GitHub ↗</a></div>',
    unsafe_allow_html=True)
