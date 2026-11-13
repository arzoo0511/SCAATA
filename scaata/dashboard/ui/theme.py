"""The HANSEI look: a dark terminal built from the brand palette.

Brand: midnight #0F414A, maroon #7F0303, alabaster #EFE8DF, tan #D8BA98,
light blue #96C0CE. The surfaces are deepened midnight; text is alabaster.
Chart series use more saturated steps of the brand's blue and tan (plus a
green and a violet for the four advisors), because the pastel brand tones
fail contrast and colour-blind separation as data marks on a dark surface.
Those steps were checked with the dataviz palette validator against BG.
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

# surfaces and ink
BG, PANEL, RAISED, LINE = "#071A1E", "#0B242A", "#0F3038", "#1B4650"
INK, SOFT, MUTED = "#EFE8DF", "#B8C7C6", "#7E9A9C"
# brand tones (decoration, not data)
MID, MAROON, ALAB, TAN, LBLUE = "#0F414A", "#7F0303", "#EFE8DF", "#D8BA98", "#96C0CE"
# data colours
HANSEI, HOLD = "#2F9BB8", "#C4822B"          # the agent, and just holding
UP, DOWN, FLAT = "#2F9BB8", "#E5604F", "#4A6468"   # supportive / against / no opinion
ADVISOR_COLOURS = {"trend": "#2F9BB8", "volatility": "#C4822B", "momentum": "#3FA873", "news": "#B26BC9",
                   "relstr": "#7E9A9C"}
ADVISOR_LABELS = {"trend": "Trend", "volatility": "Volatility", "momentum": "Momentum", "news": "News",
                  "relstr": "Relative strength (shadow)"}
ACTION_COLOURS = {"BUY": UP, "SELL": DOWN, "REBALANCE": TAN, "HOLD": MUTED}
DIVERGING = [[0.0, DOWN], [0.5, "#24383C"], [1.0, UP]]
SEQUENTIAL = [[0.0, "#0E2A31"], [1.0, "#58C2DE"]]

FONT = "'Space Grotesk', 'Segoe UI', system-ui, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, Consolas, monospace"
FONT_IMPORT = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600"
               "&family=Space+Grotesk:wght@400;500;600;700&display=swap")

CSS = f"""
<style>
@import url('{FONT_IMPORT}');
html, body, .stApp, [class*="st-"], button, input {{ font-family: {FONT}; }}
.stApp {{ background:
    radial-gradient(1200px 500px at 85% -10%, rgba(47,155,184,.10), transparent 60%), {BG}; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1240px; }}
h1, h2, h3, h4 {{ color: {INK}; font-weight: 700; letter-spacing: -0.02em; }}
p, li {{ color: {SOFT}; }}
a {{ color: {LBLUE}; }}
code {{ font-family: {MONO}; color: {TAN}; background: {RAISED}; }}

/* top bar */
.hs-brand {{ display: flex; align-items: center; gap: .8rem; }}
.hs-brand img {{ width: 42px; height: 42px; border-radius: 10px; }}
.hs-brand .w {{ font-weight: 700; font-size: 1.45rem; letter-spacing: .14em; color: {INK}; line-height: 1; }}
.hs-brand .k {{ font-size: .7rem; letter-spacing: .2em; text-transform: uppercase; color: {MUTED};
               margin-top: .3rem; }}
.hs-status {{ text-align: right; font-family: {MONO}; font-size: .74rem; color: {MUTED}; line-height: 1.7; }}
.hs-live {{ display: inline-flex; align-items: center; gap: .45rem; color: {INK}; font-weight: 600;
           letter-spacing: .08em; text-transform: uppercase; font-family: {FONT}; font-size: .72rem; }}
.hs-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #3FA873;
          box-shadow: 0 0 0 0 rgba(63,168,115,.6); animation: hs-pulse 2.4s infinite; }}
.hs-dot.stale {{ background: {DOWN}; animation: none; }}
@keyframes hs-pulse {{ 70% {{ box-shadow: 0 0 0 9px rgba(63,168,115,0); }} 100% {{ box-shadow: 0 0 0 0 rgba(63,168,115,0); }} }}
@media (prefers-reduced-motion: reduce) {{ .hs-dot {{ animation: none; }} }}

/* navigation: one bright bar, the current page filled */
div[data-testid="stHorizontalBlock"]:has(.hs-nav-on) {{ background: {PANEL}; border: 1px solid {LINE};
    border-radius: 16px; padding: 6px; gap: 6px; margin: .9rem 0 1.8rem; align-items: stretch;
    box-shadow: 0 10px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04); }}
div[data-testid="stHorizontalBlock"]:has(.hs-nav-on) div[data-testid="stVerticalBlock"] {{ gap: 0; }}
a[data-testid="stPageLink-NavLink"] {{ border-radius: 11px; padding: .7rem .3rem; justify-content: center;
    border: 1px solid transparent; transition: background .15s, border-color .15s; }}
a[data-testid="stPageLink-NavLink"] p {{ color: {SOFT}; font-weight: 600; font-size: .95rem; letter-spacing: .06em;
    text-transform: uppercase; }}
a[data-testid="stPageLink-NavLink"]:hover {{ background: {RAISED}; border-color: {LINE}; }}
a[data-testid="stPageLink-NavLink"]:hover p {{ color: {INK}; }}
a[data-testid="stPageLink-NavLink"]:focus-visible {{ outline: 2px solid {LBLUE}; }}
.hs-nav-on {{ text-align: center; padding: .7rem .3rem; font-weight: 700; font-size: .95rem; letter-spacing: .06em;
    text-transform: uppercase; color: #04161A; background: linear-gradient(180deg, #58C2DE, {HANSEI});
    border-radius: 11px; border: 1px solid #7FD3EA; line-height: 1.6;
    box-shadow: 0 6px 18px rgba(47,155,184,.35); }}

/* type */
.hs-eyebrow {{ text-transform: uppercase; letter-spacing: .2em; font-size: .7rem; font-weight: 600; color: {TAN}; }}
.hs-title {{ font-size: 2.1rem; font-weight: 700; letter-spacing: -0.03em; color: {INK}; line-height: 1.12;
            margin: .35rem 0 .6rem; max-width: 24ch; }}
.hs-lede {{ font-size: 1.02rem; color: {SOFT}; max-width: 68ch; line-height: 1.55; }}
.hs-lede b {{ color: {INK}; }}
.hs-h {{ display: flex; align-items: baseline; gap: .8rem; margin: 2.4rem 0 .3rem; }}
.hs-h .t {{ font-size: 1.15rem; font-weight: 700; color: {INK}; letter-spacing: -0.01em; }}
.hs-h .r {{ flex: 1; height: 1px; background: {LINE}; }}
.hs-h .n {{ font-family: {MONO}; font-size: .7rem; color: {MUTED}; letter-spacing: .08em; }}
.hs-cap {{ color: {MUTED}; font-size: .86rem; max-width: 80ch; line-height: 1.5; margin-bottom: .5rem; }}
.hs-cap b {{ color: {SOFT}; }}
.hs-foot {{ color: {MUTED}; font-size: .78rem; line-height: 1.55; }}

/* figures */
.hs-fig {{ display: flex; gap: 2.4rem; flex-wrap: wrap; margin: 1.1rem 0 .4rem; }}
.hs-fig .i .l {{ text-transform: uppercase; letter-spacing: .14em; font-size: .66rem; color: {MUTED}; font-weight: 600;
                display: flex; align-items: center; gap: .4rem; }}
.hs-fig .i .v {{ font-family: {MONO}; font-size: 1.9rem; font-weight: 500; color: {INK}; line-height: 1.2;
                font-variant-numeric: tabular-nums; }}
.hs-fig .i .s {{ font-family: {MONO}; font-size: .78rem; color: {MUTED}; }}
.hs-sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; }}
.hs-up {{ color: {UP}; }} .hs-down {{ color: {DOWN}; }}

/* panels and chips */
.hs-panel {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 14px; padding: 16px 18px; }}
.hs-note {{ background: {PANEL}; border: 1px solid {LINE}; border-left: 3px solid {TAN}; border-radius: 0 12px 12px 0;
           padding: 12px 16px; color: {SOFT}; font-size: .9rem; line-height: 1.55; }}
.hs-note b {{ color: {INK}; }}
.hs-chip {{ display: inline-block; padding: .14rem .55rem; border-radius: 999px; font-size: .64rem; font-weight: 700;
           letter-spacing: .12em; text-transform: uppercase; border: 1px solid {LINE}; color: {SOFT};
           white-space: nowrap; }}
.hs-chip.live {{ color: #7BD3A6; border-color: #2C6B4D; background: rgba(63,168,115,.10); }}
.hs-chip.part {{ color: {TAN}; border-color: #6B5536; background: rgba(196,130,43,.10); }}
.hs-chip.plan {{ color: {MUTED}; border-style: dashed; }}
.hs-chip.off {{ color: #F0958A; border-color: #7A3A33; background: rgba(229,96,79,.08); }}
.hs-row {{ display: flex; justify-content: space-between; gap: 1rem; padding: .7rem 0; border-bottom: 1px solid {LINE}; }}
.hs-row:last-child {{ border-bottom: none; }}
.hs-row .hs-chip {{ align-self: flex-start; flex: none; }}
.hs-row .a {{ color: {INK}; font-weight: 600; font-size: .92rem; }}
.hs-row .b {{ color: {MUTED}; font-size: .82rem; line-height: 1.5; margin-top: .15rem; max-width: 78ch; }}
.hs-head {{ padding: .55rem 0; border-bottom: 1px solid {LINE}; font-size: .88rem; color: {SOFT}; line-height: 1.45; }}
.hs-head:last-child {{ border-bottom: none; }}
.hs-sent {{ display: inline-block; min-width: 3rem; text-align: center; padding: .05rem .4rem; border-radius: 5px;
           font-family: {MONO}; font-size: .72rem; margin-right: .55rem; color: {INK}; }}

/* streamlit widgets */
div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 12px; overflow: hidden; }}
div[data-testid="stExpander"] details {{ border: 1px solid {LINE}; border-radius: 12px; background: {PANEL}; }}
div[data-testid="stPlotlyChart"] {{ border: 1px solid {LINE}; border-radius: 14px; background: {PANEL};
    padding: 6px 6px 0; }}
iframe {{ border-radius: 14px; }}
@media (max-width: 640px) {{
  div[data-testid="stHorizontalBlock"]:has(.hs-nav-on) {{ flex-direction: row !important; flex-wrap: wrap !important;
      gap: .2rem .4rem !important; }}
  div[data-testid="stHorizontalBlock"]:has(.hs-nav-on) > div[data-testid="stColumn"] {{
      min-width: 30% !important; width: 30% !important; flex: 1 1 30% !important; }}
}}
@media (max-width: 640px) {{ .hs-title {{ font-size: 1.6rem; }} .hs-fig {{ gap: 1.3rem; }}
    .hs-status {{ text-align: left; }} }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def heading(title: str, note: str = "") -> None:
    st.markdown(f'<div class="hs-h"><span class="t">{title}</span><span class="r"></span>'
                f'<span class="n">{note}</span></div>', unsafe_allow_html=True)


def caption(html: str) -> None:
    st.markdown(f'<div class="hs-cap">{html}</div>', unsafe_allow_html=True)


def foot(html: str) -> None:
    st.markdown(f'<div class="hs-foot">{html}</div>', unsafe_allow_html=True)


def intro(eyebrow: str, title: str, lede: str) -> None:
    st.markdown(f'<div class="hs-eyebrow">{eyebrow}</div><div class="hs-title">{title}</div>'
                f'<div class="hs-lede">{lede}</div>', unsafe_allow_html=True)


def figures(items: list[tuple[str, str, str, str | None]]) -> None:
    """A row of headline figures: (label, value, sub, swatch colour or None)."""
    cells = "".join(
        f'<div class="i"><div class="l">'
        + (f'<span class="hs-sw" style="background:{colour}"></span>' if colour else "")
        + f'{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'
        for label, value, sub, colour in items)
    st.markdown(f'<div class="hs-fig">{cells}</div>', unsafe_allow_html=True)


def chip(text: str, kind: str = "") -> str:
    return f'<span class="hs-chip {kind}">{text}</span>'


def signed(value: float, fmt: str = "+.2%") -> str:
    cls = "hs-up" if value >= 0 else "hs-down"
    return f'<span class="{cls}">{value:{fmt}}</span>'


def image_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
