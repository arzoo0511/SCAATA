"""The Now page's cockpit: one self-contained interactive view of where
HANSEI stands against simply holding.

Top: the verdict, the two portfolios, and how much of the money is in
stocks. Middle: the race, drawn live, with the gap shaded (blue where HANSEI
is ahead, red where it is behind), its trades marked, and three ways to look
at it. Below: the bridge from holding's return to HANSEI's, and one tile per
stock showing what it holds, what it wants, and why.

Every number comes from the paper book, the journal and the daily
evaluation as saved by the agent. Nothing is estimated here.
"""
from __future__ import annotations

import json

from scaata.dashboard.ui import theme as T

HEIGHT = 1180   # first paint only; the frame then sizes itself to its content

_TEMPLATE = r"""
<!doctype html><html><head><meta charset="utf-8">
<link href="__FONTS__" rel="stylesheet">
<style>
:root { --bg:__BG__; --panel:__PANEL__; --raised:__RAISED__; --line:__LINE__; --ink:__INK__; --soft:__SOFT__;
        --muted:__MUTED__; --up:__UP__; --down:__DOWN__; --tan:__TAN__; --hold:__HOLD__; --flat:__FLAT__; }
* { box-sizing: border-box; }
html, body { margin: 0; background: transparent; color: var(--ink); font-family: __FONT__; overflow: hidden; }
.mono { font-family: __MONO__; font-variant-numeric: tabular-nums; }
.hero { position: relative; border: 1px solid var(--line); border-radius: 18px; overflow: hidden;
        background: radial-gradient(900px 380px at 88% -8%, rgba(47,155,184,.22), transparent 62%),
                    radial-gradient(700px 300px at -5% 110%, rgba(196,130,43,.12), transparent 60%), var(--panel); }
.hero::before { content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .5;
        background-image: linear-gradient(var(--line) 1px, transparent 1px), linear-gradient(90deg, var(--line) 1px, transparent 1px);
        background-size: 44px 44px; mask-image: linear-gradient(180deg, rgba(0,0,0,.5), transparent 55%);
        -webkit-mask-image: linear-gradient(180deg, rgba(0,0,0,.5), transparent 55%); }
.head { position: relative; display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 24px; padding: 26px 28px 6px; }
.eyebrow { text-transform: uppercase; letter-spacing: .22em; font-size: .68rem; font-weight: 600; color: var(--tan); }
.verdict { font-size: clamp(2rem, 4.6vw, 3.5rem); font-weight: 700; letter-spacing: -.035em; line-height: 1.02; margin: 10px 0 12px; }
.verdict em { font-style: normal; }
.lede { color: var(--soft); font-size: 1rem; line-height: 1.55; max-width: 62ch; }
.lede b { color: var(--ink); font-weight: 600; }
.duo { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 20px; }
.pf { flex: 1 1 200px; border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; background: rgba(7,26,30,.55);
      border-top: 3px solid var(--c); }
.pf .l { font-size: .66rem; letter-spacing: .16em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.pf .v { font-family: __MONO__; font-size: clamp(1.7rem, 3.4vw, 2.5rem); font-weight: 500; line-height: 1.15; }
.pf .s { font-family: __MONO__; font-size: .78rem; color: var(--muted); }
.ring { position: relative; width: 190px; align-self: center; }
.ring svg { width: 100%; display: block; }
.ring .c { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; pointer-events: none; }
.ring .c b { font-family: __MONO__; font-size: 2rem; font-weight: 500; }
.ring .c span { font-size: .62rem; letter-spacing: .16em; text-transform: uppercase; color: var(--muted); }
.ring path { transition: opacity .15s; cursor: default; }
.race { position: relative; padding: 14px 28px 20px; }
.bar { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 6px; }
.tabs { display: inline-flex; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; background: var(--bg); }
.tabs button { background: none; border: 0; color: var(--muted); font-family: __FONT__; font-size: .78rem; font-weight: 600;
      padding: 8px 14px; cursor: pointer; }
.tabs button[aria-pressed=true] { background: var(--up); color: #04161A; }
.tabs button:focus-visible { outline: 2px solid var(--ink); outline-offset: -2px; }
.key { display: flex; gap: 16px; font-size: .74rem; color: var(--soft); flex-wrap: wrap; }
.key i { display: inline-block; width: 16px; height: 3px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
#chart { width: 100%; height: 320px; display: block; touch-action: pan-y; }
.axis { font-family: __MONO__; font-size: 11px; fill: var(--muted); }
.grid { stroke: #143238; stroke-width: 1; }
.draw { stroke-dasharray: var(--len); stroke-dashoffset: var(--len); animation: draw 1.6s ease-out forwards; }
@keyframes draw { to { stroke-dashoffset: 0; } }
.fade { opacity: 0; animation: fade .8s ease-out 1.1s forwards; }
@keyframes fade { to { opacity: 1; } }
@media (prefers-reduced-motion: reduce) { .draw, .fade { animation: none; stroke-dashoffset: 0; opacity: 1; } }
.note { color: var(--muted); font-size: .8rem; margin-top: 4px; }
.tip { position: absolute; pointer-events: none; background: var(--raised); border: 1px solid var(--line); border-radius: 10px;
       padding: 8px 11px; font-family: __MONO__; font-size: .74rem; line-height: 1.6; opacity: 0; transition: opacity .1s;
       white-space: nowrap; z-index: 4; box-shadow: 0 8px 24px rgba(0,0,0,.35); }
.tip b { font-weight: 500; } .tip .d { color: var(--muted); } .tip i { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 6px; }
h2 { font-size: 1.12rem; font-weight: 700; letter-spacing: -.01em; margin: 30px 0 4px; display: flex; align-items: baseline; gap: 12px; }
h2::after { content: ""; flex: 1; height: 1px; background: var(--line); }
h2 small { font-family: __MONO__; font-weight: 400; font-size: .68rem; color: var(--muted); letter-spacing: .06em; order: 2; }
.cap { color: var(--muted); font-size: .84rem; line-height: 1.5; max-width: 80ch; margin-bottom: 12px; }
.cap b { color: var(--soft); font-weight: 600; }
.bridge { border: 1px solid var(--line); border-radius: 14px; background: var(--panel); padding: 16px 18px; }
.br { display: grid; grid-template-columns: 170px 1fr 84px; align-items: center; gap: 12px; padding: 7px 0; }
.br .n { font-size: .86rem; color: var(--soft); } .br .n small { display: block; color: var(--muted); font-size: .72rem; }
.br .t { position: relative; height: 26px; }
.br .t::before { content: ""; position: absolute; left: var(--zero); top: -7px; bottom: -7px; width: 1px; background: var(--line); }
.br .t i { position: absolute; top: 0; height: 26px; border-radius: 5px; width: 0; transition: width 1s cubic-bezier(.2,.7,.2,1), left 1s cubic-bezier(.2,.7,.2,1); }
.br .v { font-family: __MONO__; font-size: .92rem; text-align: right; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
.tile { border: 1px solid var(--line); border-radius: 14px; background: var(--panel); padding: 14px; display: grid;
        grid-template-columns: 34px 1fr; gap: 12px; transition: border-color .2s, transform .2s; }
.tile:hover { border-color: var(--soft); transform: translateY(-2px); }
.tank { position: relative; border: 1.5px solid var(--line); border-radius: 8px; height: 132px; background: var(--bg); overflow: hidden; }
.tank i { position: absolute; left: 0; right: 0; bottom: 0; height: 0; background: linear-gradient(180deg, #58C2DE, var(--up));
          transition: height 1.1s cubic-bezier(.2,.7,.2,1); }
.tank b { position: absolute; left: -2px; right: -2px; height: 2px; background: var(--tan); box-shadow: 0 0 0 1px var(--bg); }
.tile .tk { font-family: __MONO__; font-size: .66rem; letter-spacing: .12em; color: var(--muted); }
.tile .nm { font-weight: 700; font-size: 1rem; margin: 1px 0 6px; }
.act { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: .6rem; font-weight: 700; letter-spacing: .12em; border: 1px solid; }
.tile .hw { font-family: __MONO__; font-size: .74rem; color: var(--soft); margin: 8px 0 4px; line-height: 1.5; }
.tile .hw span { color: var(--muted); }
.tile .why { font-size: .76rem; color: var(--muted); line-height: 1.4; grid-column: 1 / -1; border-top: 1px solid var(--line); padding-top: 9px; }
.tile svg { width: 100%; height: 26px; display: block; }
.queue { margin-top: 12px; border: 1px dashed var(--line); border-radius: 12px; padding: 10px 14px; font-size: .84rem; color: var(--soft); }
.queue b { color: var(--ink); } .queue span { color: var(--muted); font-size: .78rem; }
@media (max-width: 720px) { .head { grid-template-columns: 1fr; padding: 20px 18px 4px; } .ring { width: 150px; justify-self: start; }
  .race { padding: 12px 14px 16px; } .br { grid-template-columns: 110px 1fr 66px; gap: 8px; } #chart { height: 250px; } }
</style></head><body>
<div id="root">
  <section class="hero">
    <div class="head">
      <div>
        <div class="eyebrow" id="eyebrow"></div>
        <div class="verdict" id="verdict"></div>
        <div class="lede" id="lede"></div>
        <div class="duo" id="duo"></div>
      </div>
      <div class="ring" id="ring" role="img"></div>
    </div>
    <div class="race">
      <div class="bar">
        <div class="tabs" role="group" aria-label="Chart view">
          <button data-m="value" aria-pressed="true">Value</button>
          <button data-m="gap" aria-pressed="false">Gap to holding</button>
          <button data-m="dd" aria-pressed="false">Fall from peak</button>
        </div>
        <div class="key" id="key"></div>
      </div>
      <svg id="chart" role="img" aria-label="HANSEI against holding over time"></svg>
      <div class="note" id="note"></div>
      <div class="tip" id="tip"></div>
    </div>
  </section>
  <div id="bridgeWrap"></div>
  <h2>What it holds, and what it wants <small>tan line = wants · fill = holds</small></h2>
  <div class="cap">Each tank is one stock's equal slot of the book. It holds by default and trades only when the two are far apart.</div>
  <div class="tiles" id="tiles"></div>
  <div id="queue"></div>
</div>
<script>
const D = __DATA__, ACT = __ACTIONS__, SHADES = ["#2F9BB8", "#45A9C4", "#5DB7CF", "#78C4D9", "#95D1E2"];
const NS = "http://www.w3.org/2000/svg", $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const rupee = v => "₹" + Math.round(v).toLocaleString("en-IN");
const pc = (v, d = 2) => (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(d) + "%";
const day = s => new Date(s + "T00:00:00").toLocaleDateString("en-GB", {day: "numeric", month: "short"});
const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
function count(node, to, fmt) { if (still) { node.textContent = fmt(to); return; }
  const t0 = performance.now(), from = to * .97; (function step(t) { const k = Math.min(1, (t - t0) / 1100), e = 1 - Math.pow(1 - k, 3);
    node.textContent = fmt(from + (to - from) * e); if (k < 1) requestAnimationFrame(step); })(t0); }

// ---- header
const S = D.summary, ahead = S.gap >= 0;
$("eyebrow").textContent = `Paper book · ${S.sessions} sessions since ${day(D.series.dates[0])} · fake money`;
$("verdict").innerHTML = `${ahead ? "Ahead of" : "Behind"} holding by <em style="color:${ahead ? css("--up") : css("--down")}">${Math.abs(S.gap * 100).toFixed(2)} points</em>`;
$("lede").innerHTML = D.lede;
$("duo").innerHTML = `<div class="pf" style="--c:${css("--up")}"><div class="l">HANSEI</div><div class="v" id="vh"></div><div class="s">${pc(S.ret)} · worst fall ${S.dd == null ? "n/a" : pc(S.dd, 1)}</div></div>
  <div class="pf" style="--c:${css("--hold")}"><div class="l">Just holding the five</div><div class="v" id="vb"></div><div class="s">${pc(S.hold_ret)} · worst fall ${S.hold_dd == null ? "n/a" : pc(S.hold_dd, 1)}</div></div>`;
count($("vh"), S.equity, rupee); count($("vb"), S.hold, rupee);

// ---- allocation ring
(function ring() {
  const parts = D.allocation, total = parts.reduce((a, p) => a + p[1], 0) || 1, box = $("ring");
  const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", "0 0 200 200"); box.appendChild(svg);
  let a0 = -Math.PI / 2, k = 0; const R = 92, r = 66, gap = 0.03;
  const invested = parts.filter(p => p[0] !== "Liquid fund").reduce((a, p) => a + p[1], 0) / total;
  const c = document.createElement("div"); c.className = "c"; c.innerHTML = `<b>${Math.round(invested * 100)}%</b><span>in stocks</span>`; box.appendChild(c);
  box.setAttribute("aria-label", parts.map(p => `${p[0]} ${Math.round(p[1] / total * 100)}%`).join(", "));
  parts.forEach(p => { const frac = p[1] / total; if (frac <= 0) return; const a1 = a0 + frac * 2 * Math.PI, s = a0 + gap / 2, e = Math.max(s + .001, a1 - gap / 2);
    const big = e - s > Math.PI ? 1 : 0, P = (rad, ang) => `${100 + rad * Math.cos(ang)},${100 + rad * Math.sin(ang)}`;
    const cash = p[0] === "Liquid fund";
    const path = el("path", {d: `M${P(R, s)} A${R},${R} 0 ${big} 1 ${P(R, e)} L${P(r, e)} A${r},${r} 0 ${big} 0 ${P(r, s)} Z`,
      fill: cash ? "#3B5257" : SHADES[k++ % SHADES.length]}, svg);
    path.addEventListener("mouseenter", () => { c.innerHTML = `<b>${Math.round(frac * 100)}%</b><span>${esc(p[0])}</span>`; svg.querySelectorAll("path").forEach(q => q.style.opacity = q === path ? 1 : .35); });
    path.addEventListener("mouseleave", () => { c.innerHTML = `<b>${Math.round(invested * 100)}%</b><span>in stocks</span>`; svg.querySelectorAll("path").forEach(q => q.style.opacity = 1); });
    a0 = a1; });
})();

// ---- the race
let mode = "value";
const chart = $("chart"), tip = $("tip");
function draw(animate) {
  const W = chart.clientWidth || 900, H = chart.clientHeight || 320, m = {l: 58, r: 16, t: 14, b: 26};
  chart.setAttribute("viewBox", `0 0 ${W} ${H}`); chart.innerHTML = "";
  const n = D.series.dates.length, X = i => m.l + (n > 1 ? i * (W - m.l - m.r) / (n - 1) : 0);
  if (n < 2) { const t = el("text", {x: W / 2, y: H / 2, "text-anchor": "middle", class: "axis"}, chart); t.textContent = "The chart appears after the second session."; return; }
  const a = mode === "value" ? D.series.hansei : mode === "dd" ? D.series.hansei_dd.map(v => v * 100) : D.series.gap.map(v => v * 100);
  const b = mode === "value" ? D.series.hold : mode === "dd" ? D.series.hold_dd.map(v => v * 100) : null;
  let lo = Math.min(...a, ...(b || a), mode === "value" ? Infinity : 0), hi = Math.max(...a, ...(b || a), mode === "value" ? -Infinity : 0);
  const pad = (hi - lo || 1) * .12; lo -= pad; hi += pad;
  const Y = v => m.t + (hi - v) / (hi - lo) * (H - m.t - m.b);
  const fmt = v => mode === "value" ? "₹" + Math.round(v).toLocaleString("en-IN") : (v > 0 ? "+" : "") + v.toFixed(1) + (mode === "gap" ? " pts" : "%");
  for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4; el("line", {x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid"}, chart);
    const t = el("text", {x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", class: "axis"}, chart); t.textContent = fmt(v); }
  const every = Math.max(1, Math.ceil(n / Math.max(3, Math.floor(W / 110))));
  for (let i = 0; i < n; i += every) { const t = el("text", {x: X(i), y: H - 6, "text-anchor": "middle", class: "axis"}, chart); t.textContent = day(D.series.dates[i]); }
  const g = el("g", {class: animate ? "fade" : ""}, chart);
  if (mode === "gap" || mode === "dd") el("line", {x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), stroke: css("--muted"), "stroke-width": 1}, g);
  // shaded gap: blue where HANSEI is ahead, red where it is behind
  for (let i = 0; i < n - 1; i++) {
    const top0 = a[i], top1 = a[i + 1], base0 = b ? b[i] : 0, base1 = b ? b[i + 1] : 0;
    if (mode === "dd") continue;
    const sign = (top0 - base0) + (top1 - base1) >= 0;
    el("path", {d: `M${X(i)},${Y(top0)} L${X(i + 1)},${Y(top1)} L${X(i + 1)},${Y(base1)} L${X(i)},${Y(base0)} Z`, fill: sign ? css("--up") : css("--down"), opacity: .2}, g);
  }
  const line = arr => arr.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
  const stroke = (arr, colour, width) => { const p = el("path", {d: line(arr), fill: "none", stroke: colour, "stroke-width": width, "stroke-linejoin": "round", "stroke-linecap": "round"}, chart);
    if (animate && !still) { const len = p.getTotalLength(); p.style.setProperty("--len", len); p.classList.add("draw"); } return p; };
  if (b) stroke(b, css("--hold"), 2.4);
  stroke(a, mode === "gap" ? css("--ink") : css("--up"), 2.6);
  if (mode === "value") D.trades.forEach(tr => { const i = D.series.dates.indexOf(tr.date); if (i < 0) return;
    const x = X(i), y = Y(a[i]), up = tr.action === "BUY";
    el("path", {d: up ? `M${x},${y - 15} l6,10 h-12 z` : `M${x},${y + 15} l6,-10 h-12 z`, fill: up ? css("--up") : css("--down"), stroke: css("--bg"), "stroke-width": 1.5, class: animate ? "fade" : ""}, chart); });
  const last = n - 1;
  [[a, mode === "gap" ? css("--ink") : css("--up")], ...(b ? [[b, css("--hold")]] : [])].forEach(([arr, c]) =>
    el("circle", {cx: X(last), cy: Y(arr[last]), r: 4.5, fill: c, stroke: css("--bg"), "stroke-width": 2, class: animate ? "fade" : ""}, chart));
  const cross = el("line", {y1: m.t, y2: H - m.b, stroke: css("--muted"), "stroke-width": 1, "stroke-dasharray": "3 4", opacity: 0}, chart);
  const dots = [a, b].filter(Boolean).map((arr, k) => el("circle", {r: 5, fill: k ? css("--hold") : (mode === "gap" ? css("--ink") : css("--up")), stroke: css("--bg"), "stroke-width": 2, opacity: 0}, chart));
  const hit = el("rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent"}, chart);
  const move = ev => { const box = chart.getBoundingClientRect(), px = (ev.touches ? ev.touches[0].clientX : ev.clientX) - box.left;
    const i = Math.max(0, Math.min(n - 1, Math.round((px * (W / box.width) - m.l) / ((W - m.l - m.r) / (n - 1)))));
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("opacity", 1);
    [a, b].filter(Boolean).forEach((arr, k) => { dots[k].setAttribute("cx", X(i)); dots[k].setAttribute("cy", Y(arr[i])); dots[k].setAttribute("opacity", 1); });
    const tr = D.trades.filter(t => t.date === D.series.dates[i]).map(t => `<div><i style="background:${t.action === "BUY" ? css("--up") : css("--down")}"></i>${t.action.toLowerCase()} ${t.qty} × ${esc(t.name)} @ ₹${t.price.toFixed(2)}</div>`).join("");
    const g100 = D.series.gap[i] * 100;
    tip.innerHTML = `<div class="d">${new Date(D.series.dates[i] + "T00:00:00").toLocaleDateString("en-GB", {weekday: "short", day: "numeric", month: "short", year: "numeric"})}</div>` +
      (mode === "dd" ? `<div><i style="background:${css("--up")}"></i>HANSEI <b>${a[i].toFixed(2)}%</b> below its peak</div><div><i style="background:${css("--hold")}"></i>Holding <b>${b[i].toFixed(2)}%</b> below its peak</div>`
        : `<div><i style="background:${css("--up")}"></i>HANSEI <b>${rupee(D.series.hansei[i])}</b></div><div><i style="background:${css("--hold")}"></i>Holding <b>${rupee(D.series.hold[i])}</b></div><div class="d">gap ${g100 >= 0 ? "+" : "−"}${Math.abs(g100).toFixed(2)} pts · ${Math.round(D.series.invested[i] * 100)}% in stocks</div>`) + tr;
    tip.style.opacity = 1; const wrap = chart.parentElement.getBoundingClientRect(), tw = tip.offsetWidth, x = box.left - wrap.left + X(i) * (box.width / W);
    tip.style.left = Math.max(4, Math.min(wrap.width - tw - 4, x > wrap.width / 2 ? x - tw - 14 : x + 14)) + "px"; tip.style.top = (box.top - wrap.top + 18) + "px"; };
  const out = () => { tip.style.opacity = 0; cross.setAttribute("opacity", 0); dots.forEach(d => d.setAttribute("opacity", 0)); };
  hit.addEventListener("mousemove", move); hit.addEventListener("touchstart", move, {passive: true}); hit.addEventListener("touchmove", move, {passive: true});
  hit.addEventListener("mouseleave", out); hit.addEventListener("touchend", out);
  $("key").innerHTML = mode === "gap" ? `<span><i style="background:${css("--ink")}"></i>HANSEI minus holding</span><span><i style="background:${css("--up")};height:8px;opacity:.5"></i>ahead</span><span><i style="background:${css("--down")};height:8px;opacity:.5"></i>behind</span>`
    : `<span><i style="background:${css("--up")}"></i>HANSEI</span><span><i style="background:${css("--hold")}"></i>Just holding</span>` + (mode === "value" ? `<span>▲ bought · ▼ sold</span>` : "");
  $("note").textContent = {value: "Both start with the same ₹10,000. Orders fill at the next session's open with real NSE costs; idle cash earns a liquid-fund rate in both.",
    gap: "HANSEI minus holding, in percentage points of the starting ₹10,000. Above the line it is winning.",
    dd: "How far each has dropped from its own highest value. Smaller falls are the one edge the backtests support."}[mode];
}
document.querySelectorAll(".tabs button").forEach(btn => btn.addEventListener("click", () => { mode = btn.dataset.m;
  document.querySelectorAll(".tabs button").forEach(x => x.setAttribute("aria-pressed", x === btn)); draw(true); }));

// ---- the bridge
(function bridge() {
  const g = D.bridge; if (!g) return;
  const rows = [["Just holding", "the five stocks, untouched", 0, g.hold, css("--hold")],
                ["Not fully invested", `on average ${Math.round(g.average_invested * 100)}% in stocks`, g.hold, g.hold + g.exposure, g.exposure >= 0 ? css("--up") : css("--down")],
                ["Which stocks it held", "tilts between the five", g.hold + g.exposure, g.hold + g.exposure + g.selection, g.selection >= 0 ? css("--up") : css("--down")],
                ["HANSEI", "what it actually made", 0, g.hansei, css("--up")]];
  const all = rows.flatMap(r => [r[2], r[3]]), lo = Math.min(0, ...all), hi = Math.max(0, ...all), span = hi - lo || 1, P = v => (v - lo) / span * 100;
  const vals = [g.hold, g.exposure, g.selection, g.hansei];
  $("bridgeWrap").innerHTML = `<h2>Why the gap <small>from holding's return to HANSEI's</small></h2>
    <div class="cap">Holding made <b>${pc(g.hold)}</b>. Being under-invested moved HANSEI <b>${pc(g.exposure)}</b>; which stocks it held moved it <b>${pc(g.selection)}</b>.</div>
    <div class="bridge">${rows.map((r, i) => `<div class="br"><div class="n">${r[0]}<small>${r[1]}</small></div>
      <div class="t" style="--zero:${P(0)}%"><i data-l="${P(Math.min(r[2], r[3]))}" data-w="${Math.abs(P(r[3]) - P(r[2]))}" style="left:${P(r[2])}%;background:${r[4]}"></i></div>
      <div class="v mono" style="color:${i === 1 || i === 2 ? (vals[i] >= 0 ? css("--up") : css("--down")) : css("--ink")}">${pc(vals[i])}</div></div>`).join("")}</div>`;
  requestAnimationFrame(() => setTimeout(() => document.querySelectorAll(".br .t i").forEach(i => { i.style.left = i.dataset.l + "%"; i.style.width = Math.max(.6, +i.dataset.w) + "%"; }), 80));
})();

// ---- stock tiles
$("tiles").innerHTML = D.stocks.map((s, k) => { const act = ACT[s.action] || ACT.HOLD, cl = v => Math.max(0, Math.min(1, v)) * 100;
  const hist = s.history, n = hist.length, pts = hist.map((v, i) => `${n > 1 ? (i / (n - 1) * 100).toFixed(1) : 50},${(24 - Math.min(1.2, Math.max(0, v)) / 1.2 * 22).toFixed(1)}`).join(" ");
  return `<div class="tile"><div class="tank" title="holds ${Math.round(s.held * 100)}%, wants ${Math.round(s.target * 100)}%"><i data-h="${cl(s.held)}"></i><b style="bottom:calc(${cl(s.target)}% - 1px)"></b></div>
    <div><div class="tk">${esc(s.symbol)}</div><div class="nm">${esc(s.name)}</div>
      <span class="act" style="color:${act.colour};border-color:${act.colour}">${act.word.toUpperCase()}</span>
      <div class="hw">holds <b>${Math.round(s.held * 100)}%</b> <span>·</span> wants <b>${Math.round(s.target * 100)}%</b><br><span>${rupee(s.value)} in the book</span></div>
      <svg viewBox="0 0 100 26" preserveAspectRatio="none" aria-hidden="true"><polyline points="${pts}" fill="none" stroke="${css("--soft")}" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg></div>
    <div class="why">${esc(s.reason)}</div></div>`; }).join("") || '<div class="cap">No decisions recorded yet.</div>';
requestAnimationFrame(() => setTimeout(() => document.querySelectorAll(".tank i").forEach(i => i.style.height = i.dataset.h + "%"), 80));
$("queue").innerHTML = D.pending.length ? `<div class="queue"><b>Queued for the next open:</b> ` + D.pending.map(o => `${esc(o.name)} → ${rupee(o.target)} <span>(${esc(o.reason)})</span>`).join(" · ") + `</div>`
  : `<div class="queue">Nothing queued for the next open. It trades only when it has a reason to.</div>`;

// ---- size the frame to the content
function fit() { const h = Math.ceil($("root").getBoundingClientRect().height) + 6;
  try { window.frameElement.style.height = h + "px"; window.frameElement.height = h; } catch (e) { document.documentElement.style.overflowY = "auto"; document.body.style.overflow = "auto"; } }
draw(true); fit();
new ResizeObserver(fit).observe($("root"));
let rz; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => draw(false), 120); });
</script></body></html>
"""


def cockpit_html(payload: dict) -> str:
    """The cockpit page for `components.html`, with the saved state embedded."""
    actions = {"BUY": {"word": "Buy", "colour": T.UP}, "SELL": {"word": "Sell", "colour": T.DOWN},
               "REBALANCE": {"word": "Rebalance", "colour": T.TAN}, "HOLD": {"word": "Hold", "colour": T.MUTED}}
    html = _TEMPLATE
    for key, value in {
        "__DATA__": json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"),
        "__ACTIONS__": json.dumps(actions), "__FONTS__": T.FONT_IMPORT, "__FONT__": T.FONT, "__MONO__": T.MONO,
        "__BG__": T.BG, "__PANEL__": T.PANEL, "__RAISED__": T.RAISED, "__LINE__": T.LINE, "__INK__": T.INK,
        "__SOFT__": T.SOFT, "__MUTED__": T.MUTED, "__UP__": T.UP, "__DOWN__": T.DOWN, "__FLAT__": T.FLAT,
        "__TAN__": T.TAN, "__HOLD__": T.HOLD,
    }.items():
        html = html.replace(key, value)
    return html
