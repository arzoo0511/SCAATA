"""The Mind explorer: one self-contained interactive view of how HANSEI
reached each day's decisions.

Left, the evidence flow: four advisors (circle size = how far each is
trusted) feed five stocks; each ribbon's width is that advisor's
contribution to that stock's score (trust × view), blue when it supports
owning the stock and red when it argues against. Right, the selected
stock's decision in words. Below, a timeline to scrub or play through every
session the agent has lived.

It draws only what the journal recorded at the time -- the views, trust,
reasons and headlines as they were, never recomputed -- so replaying an old
session shows what HANSEI believed then, not what it would think now.
"""
from __future__ import annotations

import json

from scaata.dashboard.ui import theme as T

HEIGHT = 700

_TEMPLATE = r"""
<!doctype html><html><head><meta charset="utf-8">
<link href="__FONTS__" rel="stylesheet">
<style>
:root { --bg:__BG__; --panel:__PANEL__; --raised:__RAISED__; --line:__LINE__; --ink:__INK__; --soft:__SOFT__;
        --muted:__MUTED__; --up:__UP__; --down:__DOWN__; --flat:__FLAT__; --tan:__TAN__; --hold:__HOLD__; }
* { box-sizing: border-box; }
html, body { margin: 0; background: transparent; color: var(--ink); font-family: __FONT__; }
.wrap { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px 14px; }
.top { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap; }
.date { font-size: 1.5rem; font-weight: 700; letter-spacing: -.02em; }
.sub { font-family: __MONO__; font-size: .76rem; color: var(--muted); margin-top: 2px; }
.sub b { color: var(--soft); font-weight: 500; }
.legend { display: flex; gap: 14px; font-size: .72rem; color: var(--muted); align-items: center; flex-wrap: wrap; }
.legend i { display: inline-block; width: 22px; height: 6px; border-radius: 3px; margin-right: 5px; vertical-align: middle; }
.main { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(250px, 1fr); gap: 18px; margin-top: 8px; }
svg { width: 100%; height: auto; display: block; }
.adv circle { fill: var(--raised); stroke-width: 2.5; transition: r .5s ease; }
.adv text.n { fill: var(--soft); font-size: 13px; font-weight: 600; }
.adv text.w { fill: var(--ink); font-size: 12px; font-family: __MONO__; }
.stk rect.box { fill: var(--raised); stroke: var(--line); stroke-width: 1.5; rx: 10; transition: stroke .2s; }
.stk { cursor: pointer; outline: none; }
.stk:hover rect.box, .stk:focus-visible rect.box { stroke: var(--soft); }
.stk.on rect.box { stroke: var(--ink); stroke-width: 2; }
.stk text.n { fill: var(--ink); font-size: 13px; font-weight: 600; }
.stk text.a { font-size: 10px; font-weight: 700; letter-spacing: .1em; }
.stk rect.tr { fill: #16363D; } .stk rect.fl { fill: var(--soft); transition: width .5s ease; }
.lk { fill: none; stroke-linecap: round; transition: stroke-width .5s ease, opacity .25s, stroke .3s; }
.fx { fill: none; stroke: #fff; stroke-opacity: .5; stroke-linecap: round; stroke-dasharray: 2 16;
      animation: flow 1.6s linear infinite; pointer-events: none; transition: opacity .25s; }
@keyframes flow { to { stroke-dashoffset: -18; } }
@media (prefers-reduced-motion: reduce) { .fx { animation: none; } .lk, .adv circle, .stk rect.fl { transition: none; } }
.dim { opacity: .08 !important; }
.side { border-left: 1px solid var(--line); padding-left: 18px; min-height: 420px; }
.side h3 { margin: 0; font-size: 1.15rem; letter-spacing: -.01em; }
.tk { font-family: __MONO__; font-size: .7rem; color: var(--muted); letter-spacing: .12em; }
.act { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .64rem; font-weight: 700;
       letter-spacing: .12em; border: 1px solid; margin-left: 8px; vertical-align: middle; }
.why { color: var(--soft); font-size: .88rem; line-height: 1.5; margin: 10px 0 12px; }
.why small { display: block; color: var(--muted); font-size: .78rem; margin-top: 4px; }
.lab { font-size: .64rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); font-weight: 600;
       margin: 14px 0 6px; }
.vr { display: grid; grid-template-columns: 78px 1fr 92px; align-items: center; gap: 8px; font-size: .78rem;
      color: var(--soft); margin: 5px 0; }
.vr .bar { position: relative; height: 8px; background: #16363D; border-radius: 4px; }
.vr .bar i { position: absolute; top: 0; height: 8px; border-radius: 4px; transition: all .45s ease; }
.vr .bar b { position: absolute; left: 50%; top: -3px; width: 1px; height: 14px; background: var(--muted); }
.vr .num { font-family: __MONO__; font-size: .72rem; color: var(--muted); text-align: right; white-space: nowrap; }
.expo { display: flex; justify-content: space-between; font-family: __MONO__; font-size: .76rem; color: var(--soft); }
.nw { font-size: .8rem; color: var(--soft); line-height: 1.4; padding: 6px 0; border-bottom: 1px solid var(--line); }
.nw:last-child { border-bottom: none; } .nw span { font-family: __MONO__; font-size: .7rem; margin-right: 6px; }
.nw em { color: var(--muted); font-style: normal; font-size: .72rem; }
.quiet { color: var(--muted); font-size: .8rem; }
.time { margin-top: 12px; border-top: 1px solid var(--line); padding-top: 12px; display: grid;
        grid-template-columns: auto 1fr; gap: 14px; align-items: center; }
.btns { display: flex; gap: 6px; }
button { background: var(--raised); color: var(--ink); border: 1px solid var(--line); border-radius: 8px;
         height: 34px; min-width: 34px; padding: 0 10px; font-family: __FONT__; font-size: .78rem; font-weight: 600;
         cursor: pointer; }
button:hover { border-color: var(--soft); } button:focus-visible { outline: 2px solid var(--up); }
.track { position: relative; }
.track svg { height: 54px; }
input[type=range] { width: 100%; margin: 0; accent-color: __UP__; cursor: pointer; }
.tip { position: fixed; pointer-events: none; background: var(--raised); border: 1px solid var(--line);
       border-radius: 8px; padding: 6px 9px; font-family: __MONO__; font-size: .72rem; color: var(--ink);
       opacity: 0; transition: opacity .12s; z-index: 5; max-width: 260px; }
@media (max-width: 760px) { .main { grid-template-columns: 1fr; } .side { border-left: none; padding-left: 0;
       border-top: 1px solid var(--line); padding-top: 14px; min-height: 0; } }
</style></head><body>
<div class="wrap">
  <div class="top">
    <div><div class="date" id="date"></div><div class="sub" id="sub"></div></div>
    <div class="legend">
      <span><i style="background:var(--up)"></i>supports owning it</span>
      <span><i style="background:var(--down)"></i>argues against</span>
      <span><i style="background:var(--flat);height:2px"></i>no opinion</span>
      <span>ribbon width = trust × view</span>
    </div>
  </div>
  <div class="main">
    <svg id="flow" viewBox="0 0 720 450" role="img" aria-label="Evidence flowing from four advisors to five stocks"></svg>
    <div class="side" id="side" aria-live="polite"></div>
  </div>
  <div class="time">
    <div class="btns">
      <button id="prev" aria-label="Previous session">◀</button>
      <button id="play" aria-label="Play through every session">▶ Play</button>
      <button id="next" aria-label="Next session">▶</button>
    </div>
    <div class="track"><svg id="spark" viewBox="0 0 1000 54" preserveAspectRatio="none"></svg>
      <input id="scrub" type="range" min="0" step="1" aria-label="Session"></div>
  </div>
</div>
<div class="tip" id="tip"></div>
<script>
const DATA = __DATA__, ADV = __ADVISORS__, LABEL = __LABELS__, ACOL = __ACOLOURS__, ACT = __ACTIONS__;
const NS = "http://www.w3.org/2000/svg", $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let at = DATA.length - 1, picked = null, timer = null;
const flow = $("flow"), tip = $("tip");
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || flow).appendChild(e); return e; };
const ay = i => 70 + i * 103, sy = (i, n) => 46 + i * (358 / Math.max(n - 1, 1));
const fmtDate = s => new Date(s + "T00:00:00").toLocaleDateString("en-GB", {weekday: "short", day: "numeric", month: "short", year: "numeric"});
const pct = v => Math.round(v * 100) + "%", sgn = v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
const rupee = v => "₹" + Math.round(v).toLocaleString("en-IN");
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

function build() {
  const day = DATA[at], n = day.stocks.length;
  flow.innerHTML = "";
  const links = el("g", {}), fx = el("g", {});
  ADV.forEach((a, i) => {
    const r = 15 + 62 * day.weights[a];
    day.stocks.forEach((s, j) => {
      const v = s.views[a] || 0, c = day.weights[a] * v, silent = Math.abs(v) < 0.05;
      const x1 = 120 + r, y1 = ay(i), x2 = 548, y2 = sy(j, n), mx = (x1 + x2) / 2;
      const d = `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
      const p = el("path", {d, class: "lk", "data-a": a, "data-s": s.symbol,
        stroke: silent ? css("--flat") : (v > 0 ? css("--up") : css("--down")),
        "stroke-width": silent ? 1 : 1.5 + 34 * Math.abs(c), opacity: silent ? .35 : .62,
        "stroke-dasharray": silent ? "3 5" : "none"}, links);
      p.addEventListener("mousemove", e => showTip(e, `${LABEL[a]} → ${s.name}<br>view ${sgn(v)} × trust ${pct(day.weights[a])} = ${sgn(c)}`));
      p.addEventListener("mouseleave", hideTip);
      p.addEventListener("mouseenter", () => focus(a, s.symbol)); p.addEventListener("mouseleave", () => focus());
      if (!silent) el("path", {d, class: "fx", "data-a": a, "data-s": s.symbol,
        "stroke-width": Math.min(3, 1 + 8 * Math.abs(c))}, fx);
    });
    const g = el("g", {class: "adv", "data-a": a});
    el("circle", {cx: 120, cy: ay(i), r, stroke: ACOL[a]}, g);
    const w = el("text", {x: 120, y: ay(i) + 4, class: "w", "text-anchor": "middle"}, g); w.textContent = pct(day.weights[a]);
    const t = el("text", {x: 120 - r - 9, y: ay(i) + 4, class: "n", "text-anchor": "end"}, g); t.textContent = LABEL[a];
    g.addEventListener("mouseenter", () => focus(a, null)); g.addEventListener("mouseleave", () => focus());
    g.addEventListener("mousemove", e => showTip(e, `${LABEL[a]}<br>trusted ${(day.weights[a] * 100).toFixed(1)}% (starts at 25%)`));
    g.addEventListener("mouseleave", hideTip);
  });
  day.stocks.forEach((s, j) => {
    const y = sy(j, n), g = el("g", {class: "stk" + (s.symbol === picked ? " on" : ""), "data-s": s.symbol, tabindex: 0,
      role: "button", "aria-label": `${s.name}: ${ACT[s.action].word}`});
    el("rect", {x: 548, y: y - 30, width: 164, height: 60, class: "box"}, g);
    const t = el("text", {x: 560, y: y - 9, class: "n"}, g); t.textContent = s.name;
    const a = el("text", {x: 700, y: y - 9, class: "a", "text-anchor": "end", fill: ACT[s.action].colour}, g);
    a.textContent = ACT[s.action].word.toUpperCase();
    el("rect", {x: 560, y: y + 6, width: 140, height: 6, rx: 3, class: "tr"}, g);
    el("rect", {x: 560, y: y + 6, width: 140 * Math.min(1, Math.max(0, s.target)), height: 6, rx: 3, class: "fl"}, g);
    const h = 560 + 140 * Math.min(1, Math.max(0, s.held));
    el("rect", {x: h - 1, y: y + 2, width: 2, height: 14, fill: css("--tan")}, g);
    g.addEventListener("click", () => pick(s.symbol));
    g.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(s.symbol); } });
    g.addEventListener("mouseenter", () => focus(null, s.symbol)); g.addEventListener("mouseleave", () => focus());
    g.addEventListener("mousemove", e => showTip(e, `${s.name}<br>wants ${pct(s.target)} of its share · holds ${pct(s.held)}<br>click for the full reasoning`));
    g.addEventListener("mouseleave", hideTip);
  });
}
function focus(a, s) {
  flow.querySelectorAll(".lk, .fx").forEach(p => {
    const on = (!a && !s) || ((!a || p.dataset.a === a) && (!s || p.dataset.s === s));
    p.classList.toggle("dim", !on); });
}
function showTip(e, html) { tip.innerHTML = html; tip.style.opacity = 1;
  const x = Math.min(e.clientX + 14, window.innerWidth - 270); tip.style.left = x + "px"; tip.style.top = (e.clientY + 14) + "px"; }
function hideTip() { tip.style.opacity = 0; }
function pick(symbol) { picked = symbol; flow.querySelectorAll(".stk").forEach(g => g.classList.toggle("on", g.dataset.s === symbol)); side(); }

function side() {
  const day = DATA[at], s = day.stocks.find(x => x.symbol === picked) || day.stocks[0];
  if (!s) { $("side").innerHTML = '<div class="quiet">No decisions recorded for this session.</div>'; return; }
  const act = ACT[s.action];
  const rows = ADV.map(a => { const v = s.views[a] || 0, c = v * day.weights[a], w = Math.min(Math.abs(v), 1) * 50;
    const left = v >= 0 ? 50 : 50 - w, col = Math.abs(v) < 0.05 ? css("--flat") : (v > 0 ? css("--up") : css("--down"));
    return `<div class="vr"><span>${LABEL[a]}</span><span class="bar"><i style="left:${left}%;width:${w}%;background:${col}"></i><b></b></span>
            <span class="num">${sgn(v)} → ${sgn(c)}</span></div>`; }).join("");
  const news = s.news.length ? s.news.map(h => { const col = h.sentiment > .15 ? css("--up") : h.sentiment < -.15 ? css("--down") : css("--muted");
      return `<div class="nw"><span style="color:${col}">${sgn(h.sentiment)}</span>${esc(h.title)} <em>· ${esc(h.source)}</em></div>`; }).join("")
    : '<div class="quiet">Nothing material that day. Quiet news leaves the news advisor silent.</div>';
  const more = s.reasons.slice(1).map(esc).join(" · ");
  $("side").innerHTML = `<div class="tk">${esc(s.symbol)}</div>
    <h3>${esc(s.name)}<span class="act" style="color:${act.colour};border-color:${act.colour}">${act.word.toUpperCase()}</span></h3>
    <div class="why">${esc(s.reasons[0] || "")}${more ? `<small>${more}</small>` : ""}</div>
    <div class="expo"><span>holds ${pct(s.held)} of its share</span><span>wants ${pct(s.target)}</span></div>
    <div class="lab">Each advisor's view → its pull on the score</div>${rows}
    <div class="expo" style="margin-top:8px"><span>weighted score</span><span>${sgn(s.score)}</span></div>
    <div class="lab">News it had read</div>${news}`;
}

function header() {
  const day = DATA[at];
  $("date").textContent = fmtDate(day.session);
  const gap = day.equity != null && day.hold != null ? ((day.equity - day.hold) / 100).toFixed(2) : null;
  const fills = day.filled.length ? " · filled at the open: " + day.filled.map(f => `${f.action} ${f.qty} ${f.symbol}`).join(", ") : "";
  $("sub").innerHTML = `session ${at + 1} of ${DATA.length} · HANSEI <b>${rupee(day.equity)}</b> · holding <b>${rupee(day.hold)}</b>`
    + (gap !== null ? ` · gap <b>${gap > 0 ? "+" : ""}${gap} pts</b>` : "") + (day.lessons ? ` · learned ${day.lessons} lesson(s)` : "") + esc(fills);
}
function spark() {
  const s = $("spark"); s.innerHTML = "";
  const all = DATA.flatMap(d => [d.equity, d.hold]).filter(v => v != null); if (all.length < 2) return;
  const lo = Math.min(...all), hi = Math.max(...all), n = DATA.length;
  const X = i => n > 1 ? 8 + i * 984 / (n - 1) : 500, Y = v => 48 - (v - lo) / Math.max(hi - lo, 1) * 42;
  const line = k => DATA.map((d, i) => `${i ? "L" : "M"}${X(i)},${Y(d[k])}`).join(" ");
  el("path", {d: line("hold"), fill: "none", stroke: css("--hold"), "stroke-width": 1.5, "vector-effect": "non-scaling-stroke"}, s);
  el("path", {d: line("equity"), fill: "none", stroke: css("--up"), "stroke-width": 1.5, "vector-effect": "non-scaling-stroke"}, s);
  DATA.forEach((d, i) => { if (d.filled.length) el("rect", {x: X(i) - 1, y: 0, width: 2, height: 54, fill: css("--tan"), opacity: .35}, s); });
  el("rect", {x: X(at) - 1.5, y: 0, width: 3, height: 54, fill: css("--ink")}, s);
}
function go(i, keep) {
  at = Math.max(0, Math.min(DATA.length - 1, i)); $("scrub").value = at;
  const day = DATA[at];
  if (!keep || !day.stocks.some(s => s.symbol === picked))
    picked = (day.stocks.find(s => s.action !== "HOLD") || day.stocks[0] || {}).symbol;
  header(); build(); side(); spark();
}
function stop() { clearInterval(timer); timer = null; $("play").textContent = "▶ Play"; }
$("scrub").max = DATA.length - 1;
$("scrub").addEventListener("input", e => { stop(); go(+e.target.value, true); });
$("prev").addEventListener("click", () => { stop(); go(at - 1, true); });
$("next").addEventListener("click", () => { stop(); go(at + 1, true); });
$("play").addEventListener("click", () => {
  if (timer) return stop();
  if (at >= DATA.length - 1) go(0, true);
  $("play").textContent = "❚❚ Pause";
  timer = setInterval(() => { if (at >= DATA.length - 1) return stop(); go(at + 1, true); }, 1100);
});
function fit() { const h = Math.ceil(document.querySelector(".wrap").getBoundingClientRect().height) + 4;
  try { window.frameElement.style.height = h + "px"; window.frameElement.height = h; } catch (e) {} }
new ResizeObserver(fit).observe(document.querySelector(".wrap"));
if (DATA.length) go(at, false); else $("side").innerHTML = '<div class="quiet">HANSEI has not run yet.</div>';
</script></body></html>
"""


def explorer_html(payload: list[dict]) -> str:
    """The explorer page for `components.html`, with the journal embedded."""
    actions = {"BUY": {"word": "Buy", "colour": T.UP}, "SELL": {"word": "Sell", "colour": T.DOWN},
               "REBALANCE": {"word": "Rebalance", "colour": T.TAN}, "HOLD": {"word": "Hold", "colour": T.MUTED}}
    advisors = ["trend", "volatility", "momentum", "news"]
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = _TEMPLATE
    for key, value in {
        "__DATA__": blob, "__ADVISORS__": json.dumps(advisors),
        "__LABELS__": json.dumps({a: T.ADVISOR_LABELS[a] for a in advisors}),
        "__ACOLOURS__": json.dumps({a: T.ADVISOR_COLOURS[a] for a in advisors}),
        "__ACTIONS__": json.dumps(actions), "__FONTS__": T.FONT_IMPORT, "__FONT__": T.FONT, "__MONO__": T.MONO,
        "__BG__": T.BG, "__PANEL__": T.PANEL, "__RAISED__": T.RAISED, "__LINE__": T.LINE, "__INK__": T.INK,
        "__SOFT__": T.SOFT, "__MUTED__": T.MUTED, "__UP__": T.UP, "__DOWN__": T.DOWN, "__FLAT__": T.FLAT,
        "__TAN__": T.TAN, "__HOLD__": T.HOLD,
    }.items():
        html = html.replace(key, value)
    return html
