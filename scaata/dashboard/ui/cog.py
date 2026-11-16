"""The Beliefs page's board: everything HANSEI currently believes, sorted by
what kind of belief it is and coloured by how far the evidence goes.

Click any belief to see the numbers behind it and where they came from.
Filter by status to see only what is established, or only what the evidence
contradicts. Below the board: how its own predictions have turned out, the
loop it runs each day, and what it is still unsure about.

The board draws what `scaata.cognition.beliefs.build()` assembled from the
saved records. It writes nothing and generates no text of its own.
"""
from __future__ import annotations

import json

from scaata.dashboard.ui import theme as T

HEIGHT = 1900   # first paint only; the frame then sizes itself to its content

LOOP = [
    ("Observe", "live", "Prices and three days of headlines per stock, after every close."),
    ("Interpret", "part", "Each headline is judged good or bad for its company and labelled by what it is about "
                          "(results, crude, rates, ...). The labels feed research; they do not yet inform a view."),
    ("Hypothesise", "part", "The Lab turns anomalies in the daily record into testable questions. A person "
                            "chooses which to run."),
    ("Predict", "live", "Every advisor's view on every stock is written down as a prediction for the next 20 "
                        "sessions."),
    ("Decide", "live", "The brain weighs the views by trust and trades only on a clear gap."),
    ("See the outcome", "live", "20 sessions later each prediction is compared with what the stock did."),
    ("Critique", "live", "Trust shifts toward the advisors that were right, with a small share handed back "
                         "equally each time so a fallen advisor can recover (Fixed-Share)."),
    ("Update memory", "live", "Journal, beliefs and lessons are saved as they were and never edited; a belief "
                              "snapshot is added after every run."),
    ("Validate", "part", "The Lab re-runs every week on its own. A change still reaches the agent only when a "
                         "person approves it."),
]

_TEMPLATE = r"""
<!doctype html><html><head><meta charset="utf-8">
<link href="__FONTS__" rel="stylesheet">
<style>
:root { --bg:__BG__; --panel:__PANEL__; --raised:__RAISED__; --line:__LINE__; --ink:__INK__; --soft:__SOFT__;
        --muted:__MUTED__; --up:__UP__; --down:__DOWN__; --tan:__TAN__; --hold:__HOLD__; --flat:__FLAT__;
        --established:#3FA873; --leaning:__HOLD__; --cannot:#7E9A9C; --contradicted:__DOWN__; --untested:#4A6468; }
* { box-sizing: border-box; }
html, body { margin: 0; background: transparent; color: var(--ink); font-family: __FONT__; overflow: hidden; }
.mono { font-family: __MONO__; font-variant-numeric: tabular-nums; }
h2 { font-size: 1.12rem; font-weight: 700; letter-spacing: -.01em; margin: 30px 0 4px; display: flex; align-items: baseline; gap: 12px; }
h2::after { content: ""; flex: 1; height: 1px; background: var(--line); }
h2 small { font-family: __MONO__; font-weight: 400; font-size: .68rem; color: var(--muted); letter-spacing: .06em; order: 2; }
.cap { color: var(--muted); font-size: .84rem; line-height: 1.5; max-width: 82ch; margin-bottom: 12px; }
.cap b { color: var(--soft); font-weight: 600; }
.filters { display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 14px; }
.f { display: inline-flex; align-items: center; gap: 8px; background: var(--panel); border: 1px solid var(--line); color: var(--soft);
     border-radius: 999px; padding: 7px 13px; font-family: __FONT__; font-size: .82rem; font-weight: 600; cursor: pointer; }
.f i { width: 10px; height: 10px; border-radius: 3px; background: var(--c); } .f b { font-family: __MONO__; font-weight: 500; color: var(--ink); }
.f[aria-pressed=true] { border-color: var(--c); box-shadow: 0 0 0 1px var(--c); color: var(--ink); }
.f:focus-visible, .card:focus-visible, .cell:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
.board { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; align-items: start; }
.col { border: 1px solid var(--line); border-radius: 14px; background: var(--panel); padding: 12px; }
.col > .t { font-size: .64rem; letter-spacing: .18em; text-transform: uppercase; color: var(--tan); font-weight: 700; }
.col > .d { color: var(--muted); font-size: .76rem; line-height: 1.4; margin: 3px 0 10px; }
.card { display: block; width: 100%; text-align: left; background: var(--bg); border: 1px solid var(--line); border-left: 4px solid var(--c);
        border-radius: 9px; padding: 9px 11px; margin-bottom: 7px; color: var(--ink); font-family: __FONT__; cursor: pointer;
        transition: transform .15s, border-color .15s, opacity .2s; }
.card:hover { transform: translateX(3px); border-color: var(--soft); border-left-color: var(--c); }
.card.sel { box-shadow: 0 0 0 1px var(--ink); }
.card .s { font-size: .58rem; letter-spacing: .14em; text-transform: uppercase; font-weight: 700; color: var(--c); }
.card .m { font-size: .84rem; line-height: 1.4; margin-top: 2px; }
.card .bar { height: 3px; background: #16363D; border-radius: 2px; margin-top: 7px; } .card .bar i { display: block; height: 3px; border-radius: 2px; background: var(--c); }
.hide { display: none !important; }
.matrix { display: grid; grid-template-columns: 74px repeat(3, minmax(0, 1fr)); gap: 5px; align-items: stretch; }
.matrix .h { font-size: .66rem; color: var(--muted); text-align: center; padding-bottom: 2px; } .matrix .r { font-size: .68rem; font-weight: 600; align-self: center; overflow: hidden; text-overflow: ellipsis; }
.cell { border: 1px solid var(--line); border-radius: 8px; height: 44px; display: flex; flex-direction: column; align-items: center; justify-content: center;
        font-family: __MONO__; font-size: .8rem; color: var(--ink); cursor: pointer; background: var(--bg); transition: transform .15s;
        min-width: 0; padding: 0; overflow: hidden; }
.cell:hover { transform: scale(1.05); } .cell small { display: none; }
.cell.unk { border-style: dashed; color: var(--muted); background: repeating-linear-gradient(135deg, transparent 0 6px, rgba(126,154,156,.09) 6px 12px); }
.drawer { border: 1px solid var(--line); border-radius: 14px; background: var(--raised); padding: 16px 18px; margin: 0 0 14px; border-left: 5px solid var(--c, var(--line)); min-height: 128px; }
.drawer .k { font-size: .62rem; letter-spacing: .16em; text-transform: uppercase; color: var(--c); font-weight: 700; }
.drawer .st { font-size: 1.15rem; font-weight: 700; line-height: 1.3; margin: 5px 0 10px; }
.drawer .ev { color: var(--soft); font-size: .9rem; line-height: 1.55; } .drawer .src { font-family: __MONO__; font-size: .72rem; color: var(--muted); margin-top: 10px; }
.score { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }
.box { border: 1px solid var(--line); border-radius: 14px; background: var(--panel); padding: 14px 16px; }
.row { display: grid; grid-template-columns: 92px 1fr 150px; gap: 10px; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--line); }
.row:last-child { border-bottom: 0; } .row .n { font-size: .86rem; font-weight: 600; } .row .n small { display: block; font-family: __MONO__; font-weight: 400; font-size: .64rem; color: var(--muted); }
.track { position: relative; height: 22px; } .track::before { content: ""; position: absolute; left: 50%; top: -6px; bottom: -6px; width: 1px; background: var(--muted); }
.track .iv { position: absolute; top: 9px; height: 4px; border-radius: 2px; background: var(--c); opacity: .45; }
.track .pt { position: absolute; top: 4px; width: 14px; height: 14px; margin-left: -7px; border-radius: 50%; background: var(--c); border: 2px solid var(--bg); }
.row .v { font-family: __MONO__; font-size: .76rem; color: var(--soft); text-align: right; }
.slope { display: grid; grid-template-columns: 92px 1fr; gap: 10px; align-items: center; padding: 6px 0; }
.slope svg { width: 100%; height: 40px; display: block; }
.loop { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; counter-reset: step; }
.step { position: relative; border: 1px solid var(--line); border-radius: 12px; background: var(--panel); padding: 11px 12px; border-top: 3px solid var(--c); }
.step .n { font-weight: 700; font-size: .9rem; } .step .n::before { counter-increment: step; content: counter(step) " "; font-family: __MONO__; color: var(--muted); font-weight: 400; }
.step .w { color: var(--muted); font-size: .74rem; line-height: 1.45; margin-top: 4px; }
.step .c { font-size: .56rem; letter-spacing: .14em; text-transform: uppercase; font-weight: 700; color: var(--c); }
.mem { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; }
.mem div { border: 1px solid var(--line); border-radius: 12px; background: var(--panel); padding: 11px 12px; }
.mem b { font-family: __MONO__; font-size: 1.6rem; font-weight: 500; display: block; line-height: 1.1; } .mem span { font-size: .62rem; letter-spacing: .14em; text-transform: uppercase; color: var(--tan); font-weight: 700; }
.mem small { display: block; color: var(--muted); font-size: .74rem; line-height: 1.4; margin-top: 2px; }
.q { border: 1px dashed var(--line); border-radius: 12px; padding: 10px 14px; margin-bottom: 8px; } .q b { font-size: .9rem; } .q div { color: var(--muted); font-size: .8rem; line-height: 1.45; }
@media (max-width: 640px) { .row { grid-template-columns: 78px 1fr; } .row .v { grid-column: 1 / -1; text-align: left; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style></head><body>
<div id="root">
  <div class="filters" id="filters" role="group" aria-label="Filter beliefs by how far the evidence goes"></div>
  <div class="drawer" id="drawer" aria-live="polite"></div>
  <div class="board" id="board"></div>
  <h2>How its own predictions turned out <small id="pn"></small></h2>
  <div class="cap">Every view is a prediction about a stock's next 20 sessions against a liquid fund. A dot is how often each source pointed the right way; the bar is the range the truth plausibly lies in. <b>A bar that crosses the middle line cannot be told from a coin flip.</b> Judged over days, not opinions, because five stocks on one day are not five separate tests.</div>
  <div class="score"><div class="box" id="hits"></div><div class="box"><div class="cap" style="margin:0 0 6px"><b>Does confidence mean anything?</b> Right more often on its strong views than its weak ones would be a line rising to the right.</div><div id="slopes"></div></div></div>
  <h2>The loop it runs <small>solid: automatic · amber: partly, or by hand</small></h2>
  <div class="loop" id="loop"></div>
  <h2>What it remembers <small>five kinds of memory, all saved as files</small></h2>
  <div class="mem" id="mem"></div>
  <h2>What it is unsure about <small>open questions raised by its own record</small></h2>
  <div id="open"></div>
  <div id="hist"></div>
</div>
<script>
const D = __DATA__, LOOP = __LOOP__, HIST = __HIST__;
const $ = id => document.getElementById(id);
const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const VAR = {"established": "--established", "leaning": "--leaning", "cannot tell": "--cannot", "contradicted": "--contradicted", "untested": "--untested"};
const WORD = {"established": "Established", "leaning": "Leaning, not proven", "cannot tell": "Cannot tell yet", "contradicted": "Contradicted", "untested": "Untested"};
const col = s => `var(${VAR[s] || "--untested"})`;
const pct = v => v == null ? "n/a" : Math.round(v * 100) + "%";
function fit() { const h = Math.ceil($("root").getBoundingClientRect().height) + 6;
  try { window.frameElement.style.height = h + "px"; window.frameElement.height = h; } catch (e) { document.documentElement.style.overflowY = "auto"; document.body.style.overflow = "auto"; } }
let filter = null, selected = null;
const B = Object.fromEntries(D.beliefs.map(b => [b.id, b]));

$("filters").innerHTML = Object.keys(VAR).filter(s => D.counts[s]).map(s => `<button class="f" data-s="${s}" aria-pressed="false" style="--c:${col(s)}"><i></i>${WORD[s]} <b>${D.counts[s]}</b></button>`).join("")
  + `<button class="f" data-s="" aria-pressed="true" style="--c:var(--soft)">Everything <b>${D.beliefs.length}</b></button>`;
document.querySelectorAll(".f").forEach(b => b.addEventListener("click", () => { filter = b.dataset.s || null;
  document.querySelectorAll(".f").forEach(x => x.setAttribute("aria-pressed", (x.dataset.s || null) === filter)); apply(); }));

const card = b => `<button class="card" data-id="${b.id}" data-s="${b.status}" style="--c:${col(b.status)}"><div class="s">${WORD[b.status]}</div><div class="m">${esc(b.statement)}</div><div class="bar"><i style="width:${Math.round(b.strength * 100)}%"></i></div></button>`;
const of = k => D.beliefs.filter(b => b.kind === k);
const rel = of("relationship"), stocks = [...new Set(rel.map(b => b.detail.symbol))], factors = ["market", "crude", "rupee"];
const FL = {market: "Market", crude: "Crude", rupee: "Rupee"};
const matrix = rel.length ? `<div class="matrix"><div></div>${factors.map(f => `<div class="h">${FL[f]}</div>`).join("")}` + stocks.map(s => `<div class="r">${esc(s)}</div>` + factors.map(f => {
    const b = rel.find(x => x.detail.symbol === s && x.detail.factor === f); if (!b) return "<div></div>";
    const known = b.status === "established", a = Math.min(.75, Math.abs(b.detail.beta) / 1.6 + .12);
    return `<button class="cell ${known ? "" : "unk"}" data-id="${b.id}" data-s="${b.status}" title="${esc(b.statement)}" style="${known ? `background:${b.detail.beta > 0 ? `rgba(47,155,184,${a})` : `rgba(229,96,79,${Math.max(a, .3)})`}` : ""}">${known ? (b.detail.beta > 0 ? "+" : "−") + Math.abs(b.detail.beta).toFixed(2) : "?"}<small>${known ? "measured" : "cannot tell"}</small></button>`; }).join("")).join("") + `</div>` : `<div class="cap">No sensitivities measured yet.</div>`;
$("board").innerHTML = `
  <div class="col"><div class="t">Facts</div><div class="d">Measured, not inferred.</div>${of("fact").map(card).join("") || '<div class="cap">None yet.</div>'}</div>
  <div class="col"><div class="t">Relationships</div><div class="d">How far each stock moves when a factor moves by 1. A dashed "?" means the data cannot tell it from zero. Click a cell.</div>${matrix}</div>
  <div class="col"><div class="t">Findings and hypotheses</div><div class="d">What it has claimed, and what testing made of the claim.</div>${of("finding").concat(of("hypothesis")).map(card).join("") || '<div class="cap">None yet.</div>'}</div>
  <div class="col"><div class="t">Assumptions</div><div class="d">Design choices it runs on, checked against whatever number can check them.</div>${of("assumption").map(card).join("")}</div>`;
function apply() { document.querySelectorAll(".card, .cell").forEach(n => { const off = filter && n.dataset.s !== filter; n.classList.contains("card") ? n.classList.toggle("hide", off) : (n.style.opacity = off ? .15 : 1); }); fit(); }
function pick(id) { selected = id; const b = B[id]; document.querySelectorAll(".card").forEach(n => n.classList.toggle("sel", n.dataset.id === id));
  $("drawer").style.setProperty("--c", col(b.status));
  $("drawer").innerHTML = `<div class="k">${esc(b.kind)} · ${WORD[b.status]}</div><div class="st">${esc(b.statement)}</div><div class="ev"><b style="color:var(--ink)">Evidence.</b> ${esc(b.evidence) || "None on file."}</div><div class="src">source: ${esc(b.source)}</div>`; fit(); }
document.querySelectorAll(".card, .cell").forEach(n => n.addEventListener("click", () => pick(n.dataset.id)));
pick((D.beliefs.find(b => b.kind === "fact") || D.beliefs[0]).id);

// prediction scorecard
const NAMES = {brain: "The brain", trend: "Trend", volatility: "Volatility", momentum: "Momentum", news: "News"};
const order = ["brain", "trend", "volatility", "momentum", "news"].filter(s => D.scorecard[s] && D.scorecard[s].opinions);
$("pn").textContent = `${D.predictions.judged} judged predictions over ${D.predictions.days} days`;
$("hits").innerHTML = order.length ? order.map(s => { const c = D.scorecard[s];
  return `<div class="row" style="--c:${col(c.status)}"><div class="n">${NAMES[s]}<small>${c.days} days · ${c.opinions} views</small></div>
    <div class="track"><span class="iv" style="left:${c.low * 100}%;width:${(c.high - c.low) * 100}%"></span><span class="pt" style="left:${c.hit_rate * 100}%"></span></div>
    <div class="v">${pct(c.hit_rate)} right · ${pct(c.low)}–${pct(c.high)}<br><span style="color:${col(c.status)}">${WORD[c.status].toLowerCase()}</span></div></div>`; }).join("")
  : `<div class="cap">No prediction is old enough to judge yet. The first are judged 20 sessions after they are made.</div>`;
$("slopes").innerHTML = order.map(s => { const c = D.scorecard[s]; if (c.weak == null || c.strong == null) return "";
  const y = v => 36 - v * 32, up = c.strong >= c.weak, colr = up ? "var(--up)" : "var(--down)";
  return `<div class="slope"><div class="n" style="font-size:.86rem;font-weight:600">${NAMES[s]}</div><svg viewBox="0 0 300 40" preserveAspectRatio="none" role="img" aria-label="${NAMES[s]}: weak views right ${pct(c.weak)}, strong views right ${pct(c.strong)}">
    <line x1="0" x2="300" y1="${y(.5)}" y2="${y(.5)}" stroke="var(--muted)" stroke-dasharray="3 5"/><line x1="70" y1="${y(c.weak)}" x2="230" y2="${y(c.strong)}" stroke="${colr}" stroke-width="2.5" vector-effect="non-scaling-stroke"/>
    <circle cx="70" cy="${y(c.weak)}" r="5" fill="${colr}"/><circle cx="230" cy="${y(c.strong)}" r="5" fill="${colr}"/>
    <text x="62" y="${y(c.weak) + 4}" text-anchor="end" font-size="11" fill="var(--soft)" font-family="monospace">weak ${pct(c.weak)}</text>
    <text x="238" y="${y(c.strong) + 4}" font-size="11" fill="var(--soft)" font-family="monospace">strong ${pct(c.strong)}</text></svg></div>`; }).join("")
  + (order.length ? `<div class="cap" style="margin:8px 0 0">Half of each source's views are "weak", half "strong". With this few days, a slope either way is not evidence yet.</div>` : "");

$("loop").innerHTML = LOOP.map(([n, k, w]) => `<div class="step" style="--c:${k === "live" ? "var(--established)" : "var(--leaning)"}"><div class="c">${k === "live" ? "automatic" : "partly"}</div><div class="n">${esc(n)}</div><div class="w">${esc(w)}</div></div>`).join("");
$("mem").innerHTML = Object.entries(D.memory).map(([k, m]) => `<div><span>${k}</span><b>${m.records}</b><small>${esc(m.what)}</small></div>`).join("");
$("open").innerHTML = D.open_questions.length ? D.open_questions.map(q => `<div class="q"><b>${esc(q.question)}</b><div>${esc(q.trigger)} ${esc(q.note || "")}</div></div>`).join("")
  : `<div class="cap">No open questions on the Lab's agenda.</div>`;
if (HIST && HIST.length) { const days = [...new Set(HIST.map(h => h.at.slice(0, 10)))], changed = HIST.filter(h => h.was);
  $("hist").innerHTML = `<h2>How its beliefs have changed <small>append-only record</small></h2>` + (changed.length
    ? changed.slice(-12).reverse().map(h => `<div class="q" style="border-style:solid"><b>${esc(h.statement)}</b><div><span style="color:${col(h.was)}">${WORD[h.was]}</span> → <span style="color:${col(h.status)}">${WORD[h.status]}</span> · ${h.at.slice(0, 10)}</div></div>`).join("")
    : `<div class="cap">${HIST.length} beliefs were first recorded on ${days[0]}. None has changed status since. Each later snapshot adds only what is new or has changed, so earlier beliefs are never rewritten.</div>`); }
new ResizeObserver(fit).observe($("root")); fit();
</script></body></html>
"""


def board_html(state: dict, history: list[dict] | None = None) -> str:
    """The Beliefs page for `components.html`, with the assembled state embedded."""
    slim = {**state, "predictions": {k: v for k, v in state["predictions"].items() if k != "rows"}}
    html = _TEMPLATE
    for key, value in {
        "__DATA__": json.dumps(slim, ensure_ascii=False).replace("</", "<\\/"),
        "__LOOP__": json.dumps(LOOP, ensure_ascii=False).replace("</", "<\\/"),
        "__HIST__": json.dumps(history or [], ensure_ascii=False).replace("</", "<\\/"),
        "__FONTS__": T.FONT_IMPORT, "__FONT__": T.FONT, "__MONO__": T.MONO, "__BG__": T.BG, "__PANEL__": T.PANEL,
        "__RAISED__": T.RAISED, "__LINE__": T.LINE, "__INK__": T.INK, "__SOFT__": T.SOFT, "__MUTED__": T.MUTED,
        "__UP__": T.UP, "__DOWN__": T.DOWN, "__FLAT__": T.FLAT, "__TAN__": T.TAN, "__HOLD__": T.HOLD,
    }.items():
        html = html.replace(key, value)
    return html
