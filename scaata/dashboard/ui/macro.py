"""The Macroverse stage: one self-contained interactive view with two modes.

Replay history: pick a real episode and watch it unfold day by day -- the
market, crude and the rupee; each of the five stocks; how much HANSEI held
of each; its trades; and HANSEI against simply holding, with a running
score and a log of what happened in the world.

What if: pick a synthetic shock and see how it reaches each stock through
measured sensitivities (dashed where the data cannot tell the effect from
zero), and the spread of outcomes when the real brain trades each
simulated path.

The two are visibly different and labelled: REAL or SYNTHETIC. Everything
drawn was computed beforehand by `python -m scaata.macroverse all` and
saved under `macroverse/runs/`; this page computes nothing.
"""
from __future__ import annotations

import json

from scaata.dashboard.ui import theme as T

HEIGHT = 1250   # first paint only; the frame then sizes itself to its content

_TEMPLATE = r"""
<!doctype html><html><head><meta charset="utf-8">
<link href="__FONTS__" rel="stylesheet">
<style>
:root { --bg:__BG__; --panel:__PANEL__; --raised:__RAISED__; --line:__LINE__; --ink:__INK__; --soft:__SOFT__;
        --muted:__MUTED__; --up:__UP__; --down:__DOWN__; --tan:__TAN__; --hold:__HOLD__; --flat:__FLAT__; }
* { box-sizing: border-box; }
html, body { margin: 0; background: transparent; color: var(--ink); font-family: __FONT__; overflow: hidden; }
.mono { font-family: __MONO__; font-variant-numeric: tabular-nums; }
button { font-family: __FONT__; cursor: pointer; }
.modes { display: inline-flex; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: var(--bg); }
.modes button { background: none; border: 0; color: var(--soft); font-weight: 600; font-size: .92rem; padding: 10px 18px; }
.modes button[aria-pressed=true] { background: var(--up); color: #04161A; }
.picks { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: 14px 0; }
.pick { text-align: left; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 11px 13px;
        color: var(--ink); transition: border-color .15s, transform .15s; }
.pick:hover { border-color: var(--soft); transform: translateY(-2px); }
.pick[aria-pressed=true] { border-color: var(--up); box-shadow: 0 0 0 1px var(--up), 0 8px 22px rgba(47,155,184,.2); }
.pick b { display: block; font-size: .92rem; } .pick span { font-family: __MONO__; font-size: .7rem; color: var(--muted); }
.badge { display: inline-block; font-size: .58rem; font-weight: 700; letter-spacing: .14em; padding: 2px 7px; border-radius: 999px;
         border: 1px solid; margin-bottom: 6px; }
.badge.real { color: #7BD3A6; border-color: #2C6B4D; } .badge.syn { color: var(--tan); border-color: #6B5536; border-style: dashed; }
.stage { position: relative; border: 1px solid var(--line); border-radius: 18px; overflow: hidden; padding: 20px 22px;
         background: radial-gradient(800px 320px at 90% -10%, rgba(47,155,184,.18), transparent 60%), var(--panel); }
.stage.syn { background: repeating-linear-gradient(135deg, rgba(196,130,43,.05) 0 14px, transparent 14px 28px), var(--panel);
             border-style: dashed; border-color: #6B5536; }
.top { display: flex; justify-content: space-between; gap: 18px; flex-wrap: wrap; align-items: flex-end; }
.title { font-size: clamp(1.5rem, 3vw, 2.2rem); font-weight: 700; letter-spacing: -.03em; line-height: 1.05; }
.sum { color: var(--soft); font-size: .92rem; line-height: 1.5; max-width: 70ch; margin-top: 6px; }
.clock { text-align: right; } .clock .d { font-family: __MONO__; font-size: 1.5rem; } .clock .s { font-size: .7rem; color: var(--muted); letter-spacing: .12em; text-transform: uppercase; }
.grid { display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(260px, 1fr); gap: 18px; margin-top: 14px; }
svg { display: block; width: 100%; }
#chart { height: 300px; } #fan { height: 340px; }
.axis { font-family: __MONO__; font-size: 11px; fill: var(--muted); } .gridline { stroke: #143238; }
.key { display: flex; gap: 14px; font-size: .74rem; color: var(--soft); flex-wrap: wrap; margin: 2px 0 4px; }
.key i { display: inline-block; width: 16px; height: 3px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
.ctl { display: flex; gap: 8px; align-items: center; margin-top: 10px; }
.ctl button { background: var(--raised); color: var(--ink); border: 1px solid var(--line); border-radius: 9px; height: 36px; min-width: 40px;
              padding: 0 12px; font-weight: 600; font-size: .8rem; }
.ctl button:hover { border-color: var(--soft); } .ctl button.on { background: var(--up); color: #04161A; border-color: var(--up); }
.ctl input { flex: 1; accent-color: __UP__; }
.lab { font-size: .62rem; letter-spacing: .16em; text-transform: uppercase; color: var(--muted); font-weight: 600; margin: 0 0 7px; }
.world { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 14px; }
.fac { border: 1px solid var(--line); border-radius: 11px; padding: 9px 10px; background: var(--bg); }
.fac .n { font-size: .68rem; color: var(--muted); } .fac .v { font-family: __MONO__; font-size: 1.15rem; }
.fac svg { height: 24px; margin-top: 3px; }
.stk { display: grid; grid-template-columns: 1fr 64px 46px; gap: 8px; align-items: center; padding: 7px 9px; border-radius: 10px;
       margin-bottom: 6px; border: 1px solid var(--line); transition: background .25s; }
.stk .n { font-weight: 600; font-size: .86rem; } .stk .n small { display: block; font-family: __MONO__; font-size: .62rem; color: var(--muted); font-weight: 400; }
.stk .p { font-family: __MONO__; font-size: .86rem; text-align: right; }
.bar { position: relative; height: 26px; border: 1px solid var(--line); border-radius: 6px; background: var(--bg); overflow: hidden; }
.bar i { position: absolute; left: 0; right: 0; bottom: 0; background: linear-gradient(180deg, #58C2DE, var(--up)); }
.bar em { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-family: __MONO__;
          font-style: normal; font-size: .62rem; color: var(--ink); text-shadow: 0 0 3px var(--bg); }
.log { border-top: 1px solid var(--line); margin-top: 12px; padding-top: 10px; min-height: 92px; }
.ev { font-size: .8rem; color: var(--muted); padding: 3px 0 3px 12px; border-left: 2px solid var(--line); margin-bottom: 4px; line-height: 1.4; }
.ev.new { color: var(--ink); border-color: var(--tan); } .ev span { font-family: __MONO__; font-size: .68rem; color: var(--tan); margin-right: 6px; }
.score { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 16px; }
.sc { border: 1px solid var(--line); border-radius: 12px; padding: 11px 13px; background: var(--bg); border-top: 3px solid var(--c, var(--line)); }
.sc .l { font-size: .62rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.sc .v { font-family: __MONO__; font-size: 1.45rem; line-height: 1.2; } .sc .s { font-family: __MONO__; font-size: .72rem; color: var(--muted); }
.notes { color: var(--muted); font-size: .78rem; line-height: 1.55; margin-top: 14px; }
.notes b { color: var(--soft); }
.water { position: absolute; right: 18px; top: 14px; font-size: .62rem; letter-spacing: .2em; font-weight: 700; color: var(--tan);
         border: 1px dashed #6B5536; padding: 3px 9px; border-radius: 999px; }
.tip { position: fixed; pointer-events: none; background: var(--raised); border: 1px solid var(--line); border-radius: 9px; padding: 7px 10px;
       font-family: __MONO__; font-size: .72rem; line-height: 1.55; opacity: 0; transition: opacity .1s; z-index: 9; max-width: 280px; }
.na { border: 1px dashed var(--line); border-radius: 12px; padding: 10px 14px; margin-top: 8px; }
.na b { font-size: .88rem; } .na div { color: var(--muted); font-size: .8rem; line-height: 1.45; }
.net text { font-size: 12px; fill: var(--ink); } .net .f { font-weight: 700; } .net .m { font-family: __MONO__; font-size: 11px; }
.lk { fill: none; stroke-linecap: round; transition: opacity .15s; } .dim { opacity: .07 !important; }
.empty { color: var(--muted); padding: 30px 0; text-align: center; }
@media (max-width: 820px) { .grid { grid-template-columns: 1fr; } .clock { text-align: left; } .stage { padding: 16px 14px; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style></head><body>
<div id="root">
  <div class="modes" role="group" aria-label="Mode">
    <button data-mode="replay" aria-pressed="true">Replay history</button>
    <button data-mode="scenario" aria-pressed="false">What if</button>
  </div>
  <div class="picks" id="picks"></div>
  <div id="stage"></div>
</div>
<div class="tip" id="tip"></div>
<script>
const RUNS = __RUNS__, SENS = __SENS__;
const NS = "http://www.w3.org/2000/svg", $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const pc = (v, d = 1) => (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(d) + "%";
const dstr = (s, o) => new Date(s + "T00:00:00").toLocaleDateString("en-GB", o || {day: "numeric", month: "short", year: "numeric"});
const FLAB = (SENS && SENS.labels) || {market: "Nifty 50", crude: "Brent crude ($)", rupee: "USD/INR"};
const tip = $("tip");
function showTip(e, html) { tip.innerHTML = html; tip.style.opacity = 1; tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 290) + "px"; tip.style.top = (e.clientY + 14) + "px"; }
function hideTip() { tip.style.opacity = 0; }
function fit() { const h = Math.ceil($("root").getBoundingClientRect().height) + 6;
  try { window.frameElement.style.height = h + "px"; window.frameElement.height = h; } catch (e) { document.documentElement.style.overflowY = "auto"; document.body.style.overflow = "auto"; } }
function heat(change) { const k = Math.max(-1, Math.min(1, change / 0.3)), a = Math.abs(k) * .42; return k >= 0 ? `rgba(47,155,184,${a})` : `rgba(229,96,79,${a})`; }

let mode = "replay", pickId = null, at = 0, timer = null, speed = 1;
const byKind = k => RUNS.filter(r => r.kind === k);
function stop() { clearInterval(timer); timer = null; const b = $("play"); if (b) b.textContent = "▶ Play"; }

function picks() {
  const list = byKind(mode);
  if (!list.find(r => r.id === pickId)) pickId = list.length ? list[Math.min(mode === "replay" ? 4 : 0, list.length - 1)].id : null;
  $("picks").innerHTML = list.map(r => `<button class="pick" data-id="${r.id}" aria-pressed="${r.id === pickId}">
    <span class="badge ${mode === "replay" ? "real" : "syn"}">${mode === "replay" ? "REAL HISTORY" : "SYNTHETIC"}</span>
    <b>${esc(r.title)}</b><span>${mode === "replay" ? dstr(r.start, {month: "short", year: "numeric"}) + " – " + dstr(r.end, {month: "short", year: "numeric"})
      : Object.entries(r.shock).map(([f, v]) => `${FLAB[f].split(" ")[0]} ${pc(v, 0)}`).join(" · ")}</span></button>`).join("");
  document.querySelectorAll(".pick").forEach(b => b.addEventListener("click", () => { stop(); pickId = b.dataset.id; picks(); }));
  const run = RUNS.find(r => r.id === pickId);
  if (!run) { $("stage").innerHTML = `<div class="stage"><div class="empty">Nothing has been run yet. Run <span class="mono">python -m scaata.macroverse all</span>.</div></div>`; fit(); return; }
  mode === "replay" ? replayStage(run) : scenarioStage(run);
}

// ============================================================ replay
function replayStage(R) {
  at = 0;
  $("stage").innerHTML = `<div class="stage">
    <div class="top"><div><span class="badge real">REAL HISTORY · REAL BRAIN</span><div class="title">${esc(R.title)}</div><div class="sum">${esc(R.summary)}</div></div>
      <div class="clock"><div class="s" id="sess"></div><div class="d" id="date"></div></div></div>
    <div class="grid"><div>
        <div class="key"><span><i style="background:${css("--up")}"></i>HANSEI</span><span><i style="background:${css("--hold")}"></i>Just holding</span>
          <span><i style="background:${css("--tan")};width:2px;height:12px"></i>world event</span><span>▲▼ its trades</span><span>both start at 100</span></div>
        <svg id="chart" role="img" aria-label="HANSEI against holding through the episode"></svg>
        <div class="ctl"><button id="restart" aria-label="Back to the start">⏮</button><button id="play">▶ Play</button>
          <button class="sp on" data-s="1">1×</button><button class="sp" data-s="3">3×</button><button class="sp" data-s="8">8×</button>
          <input id="scrub" type="range" min="0" max="${R.sessions - 1}" value="0" aria-label="Day"></div>
        <div class="log"><div class="lab">What happened in the world</div><div id="log"></div></div>
      </div><div>
        <div class="lab">The world since day one</div><div class="world" id="world"></div>
        <div class="lab">The five stocks · price since day one · how much HANSEI holds</div><div id="stocks"></div>
      </div></div>
    <div class="score" id="score"></div>
    <div class="notes">${R.notes.map(n => "· " + esc(n)).join("<br>")}</div></div>`;
  $("world").innerHTML = Object.keys(R.factors).map(f => `<div class="fac"><div class="n">${FLAB[f]}</div><div class="v" id="fv_${f}"></div><svg id="fs_${f}" viewBox="0 0 100 24" preserveAspectRatio="none"></svg></div>`).join("")
    || `<div class="fac" style="grid-column:1/-1"><div class="n">No factor data for these dates</div></div>`;
  $("stocks").innerHTML = R.stocks.map((s, i) => `<div class="stk" id="st_${i}"><div class="n">${esc(s.name)}<small>${esc(s.symbol)}</small></div><div class="p" id="sp_${i}"></div>
    <div class="bar" title="share of its equal slot HANSEI holds"><i id="sb_${i}"></i><em id="se_${i}"></em></div></div>`).join("")
    + `<div class="stk" style="border-style:dashed"><div class="n">In stocks overall<small>the rest is in a liquid fund</small></div><div class="p"></div><div class="bar"><i id="inv_b"></i><em id="inv_e"></em></div></div>`;
  const chart = $("chart"), n = R.sessions;
  let geom = null;
  function base() {
    const W = chart.clientWidth || 800, H = chart.clientHeight || 300, m = {l: 44, r: 14, t: 16, b: 24};
    chart.setAttribute("viewBox", `0 0 ${W} ${H}`); chart.innerHTML = "";
    const all = R.hansei.concat(R.holding), lo0 = Math.min(...all), hi0 = Math.max(...all), pad = (hi0 - lo0 || 1) * .1, lo = lo0 - pad, hi = hi0 + pad;
    const X = i => m.l + i * (W - m.l - m.r) / (n - 1), Y = v => m.t + (hi - v) / (hi - lo) * (H - m.t - m.b);
    for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4; el("line", {x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "gridline"}, chart);
      const t = el("text", {x: m.l - 7, y: Y(v) + 4, "text-anchor": "end", class: "axis"}, chart); t.textContent = v.toFixed(0); }
    el("line", {x1: m.l, x2: W - m.r, y1: Y(100), y2: Y(100), stroke: css("--muted"), "stroke-dasharray": "2 5"}, chart);
    const step = Math.max(1, Math.ceil(n / Math.max(3, Math.floor(W / 120))));
    for (let i = 0; i < n; i += step) { const t = el("text", {x: X(i), y: H - 6, "text-anchor": "middle", class: "axis"}, chart); t.textContent = dstr(R.dates[i], {month: "short", year: "2-digit"}); }
    const line = (arr, upto) => arr.slice(0, upto + 1).map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    el("path", {d: line(R.holding, n - 1), fill: "none", stroke: css("--hold"), "stroke-width": 1, opacity: .22}, chart);
    el("path", {d: line(R.hansei, n - 1), fill: "none", stroke: css("--up"), "stroke-width": 1, opacity: .22}, chart);
    const flags = R.events.map(e => { const g = el("g", {opacity: .25}, chart);
      el("line", {x1: X(e.index), x2: X(e.index), y1: m.t, y2: H - m.b, stroke: css("--tan"), "stroke-width": 1.5, "stroke-dasharray": "4 3"}, g);
      el("circle", {cx: X(e.index), cy: m.t, r: 4, fill: css("--tan")}, g);
      const hit = el("rect", {x: X(e.index) - 8, y: m.t - 6, width: 16, height: H - m.t - m.b + 6, fill: "transparent"}, g);
      hit.addEventListener("mousemove", ev => showTip(ev, `<b>${dstr(e.date)}</b><br>${esc(e.text)}`)); hit.addEventListener("mouseleave", hideTip); return g; });
    const hold = el("path", {fill: "none", stroke: css("--hold"), "stroke-width": 2.4, "stroke-linejoin": "round"}, chart);
    const agent = el("path", {fill: "none", stroke: css("--up"), "stroke-width": 2.6, "stroke-linejoin": "round"}, chart);
    const marks = R.trades.map(t => { const up = t.rupees > 0, x = X(t.index), y = Y(R.hansei[t.index]);
      const p = el("path", {d: up ? `M${x},${y - 14} l5,9 h-10 z` : `M${x},${y + 14} l5,-9 h-10 z`, fill: up ? css("--up") : css("--down"), stroke: css("--bg"), "stroke-width": 1.2, opacity: 0}, chart);
      p.addEventListener("mousemove", ev => showTip(ev, `<b>${dstr(t.date)}</b><br>${up ? "bought" : "sold"} ${esc(t.name)}: ₹${Math.abs(t.rupees).toFixed(0)} of a ₹10,000 book`)); p.addEventListener("mouseleave", hideTip); return p; });
    const head = el("line", {y1: m.t, y2: H - m.b, stroke: css("--ink"), "stroke-width": 1}, chart);
    const da = el("circle", {r: 5, fill: css("--up"), stroke: css("--bg"), "stroke-width": 2}, chart), dh = el("circle", {r: 5, fill: css("--hold"), stroke: css("--bg"), "stroke-width": 2}, chart);
    const over = el("rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent", style: "cursor:ew-resize"}, chart);
    const seek = ev => { const box = chart.getBoundingClientRect(), px = ((ev.touches ? ev.touches[0].clientX : ev.clientX) - box.left) * (W / box.width);
      stop(); go(Math.round((px - m.l) / ((W - m.l - m.r) / (n - 1)))); };
    over.addEventListener("click", seek); over.addEventListener("mousemove", ev => { if (ev.buttons === 1) seek(ev); });
    flags.forEach(g => chart.appendChild(g)); marks.forEach(p => chart.appendChild(p));
    geom = {X, Y, line, hold, agent, marks, flags, head, da, dh};
  }
  const dd = (arr, i) => { let peak = -Infinity, worst = 0; for (let k = 0; k <= i; k++) { peak = Math.max(peak, arr[k]); worst = Math.min(worst, arr[k] / peak - 1); } return worst; };
  function go(i) {
    at = Math.max(0, Math.min(n - 1, i)); $("scrub").value = at;
    const g = geom; g.hold.setAttribute("d", g.line(R.holding, at)); g.agent.setAttribute("d", g.line(R.hansei, at));
    g.head.setAttribute("x1", g.X(at)); g.head.setAttribute("x2", g.X(at));
    g.da.setAttribute("cx", g.X(at)); g.da.setAttribute("cy", g.Y(R.hansei[at])); g.dh.setAttribute("cx", g.X(at)); g.dh.setAttribute("cy", g.Y(R.holding[at]));
    R.trades.forEach((t, k) => g.marks[k].setAttribute("opacity", t.index <= at ? 1 : 0));
    R.events.forEach((e, k) => g.flags[k].setAttribute("opacity", e.index <= at ? 1 : .25));
    $("date").textContent = dstr(R.dates[at], {weekday: "short", day: "numeric", month: "short", year: "numeric"});
    $("sess").textContent = `day ${at + 1} of ${n}`;
    Object.keys(R.factors).forEach(f => { const arr = R.factors[f], v = arr[at] / 100 - 1, node = $("fv_" + f);
      node.textContent = pc(v); node.style.color = Math.abs(v) < .005 ? css("--soft") : ((f === "market" ? v > 0 : v < 0) ? css("--up") : css("--down"));
      const lo = Math.min(...arr), hi = Math.max(...arr), s = $("fs_" + f); s.innerHTML = "";
      el("polyline", {points: arr.slice(0, at + 1).map((x, k) => `${(k / (n - 1) * 100).toFixed(1)},${(22 - (x - lo) / (hi - lo || 1) * 20).toFixed(1)}`).join(" "), fill: "none", stroke: css("--soft"), "stroke-width": 1.5, "vector-effect": "non-scaling-stroke"}, s); });
    R.stocks.forEach((s, k) => { const ch = s.price[at] / 100 - 1, ex = Math.max(0, s.exposure[at]);
      $("st_" + k).style.background = heat(ch); $("sp_" + k).textContent = pc(ch); $("sp_" + k).style.color = ch >= 0 ? css("--ink") : "#F0958A";
      $("sb_" + k).style.height = Math.min(100, ex * 100) + "%"; $("se_" + k).textContent = Math.round(ex * 100) + "%"; });
    $("inv_b").style.height = Math.min(100, R.invested[at] * 100) + "%"; $("inv_e").textContent = Math.round(R.invested[at] * 100) + "%";
    const seen = R.events.filter(e => e.index <= at);
    $("log").innerHTML = seen.length ? seen.slice().reverse().map((e, k) => `<div class="ev ${k === 0 ? "new" : ""}"><span>${dstr(e.date, {day: "numeric", month: "short"})}</span>${esc(e.text)}</div>`).join("")
      : `<div class="ev">Nothing yet. Press play.</div>`;
    const a = R.hansei[at] / 100 - 1, h = R.holding[at] / 100 - 1, da = dd(R.hansei, at), dh = dd(R.holding, at), done = at === n - 1, st = R.stats;
    const rec = v => v == null ? "not within the episode" : v + " sessions";
    $("score").innerHTML = `<div class="sc" style="--c:${css("--up")}"><div class="l">HANSEI so far</div><div class="v">${pc(a)}</div><div class="s">worst fall ${pc(da)}</div></div>
      <div class="sc" style="--c:${css("--hold")}"><div class="l">Holding so far</div><div class="v">${pc(h)}</div><div class="s">worst fall ${pc(dh)}</div></div>
      <div class="sc"><div class="l">HANSEI minus holding</div><div class="v" style="color:${a - h >= 0 ? css("--up") : css("--down")}">${(a - h >= 0 ? "+" : "−") + Math.abs((a - h) * 100).toFixed(1)} pts</div><div class="s">${da > dh ? "has fallen less" : "has fallen as much or more"}</div></div>
      <div class="sc"><div class="l">${done ? "Back to its old high" : "Trades so far"}</div><div class="v" style="font-size:${done ? "1rem" : "1.45rem"};padding-top:${done ? ".3rem" : "0"}">${done ? "HANSEI: " + rec(st.hansei.recovery_sessions) : R.trades.filter(t => t.index <= at).length}</div>
        <div class="s">${done ? "holding: " + rec(st.holding.recovery_sessions) + ` · ${st.trades} trades, ₹${st.costs.toFixed(0)} costs` : Math.round(R.invested[at] * 100) + "% in stocks now"}</div></div>`;
  }
  base(); go(0); fit();
  $("scrub").addEventListener("input", e => { stop(); go(+e.target.value); });
  $("restart").addEventListener("click", () => { stop(); go(0); });
  document.querySelectorAll(".sp").forEach(b => b.addEventListener("click", () => { speed = +b.dataset.s; document.querySelectorAll(".sp").forEach(x => x.classList.toggle("on", x === b)); if (timer) { stop(); $("play").click(); } }));
  $("play").addEventListener("click", () => { if (timer) return stop(); if (at >= n - 1) go(0); $("play").textContent = "❚❚ Pause";
    timer = setInterval(() => { if (at >= n - 1) return stop(); go(at + speed); }, 70); });
  let rz; window.onresize = () => { clearTimeout(rz); rz = setTimeout(() => { if (mode === "replay") { base(); go(at); } }, 150); };
}

// ============================================================ scenario
function scenarioStage(S) {
  const st = S.stats;
  $("stage").innerHTML = `<div class="stage syn"><span class="water">SYNTHETIC · NOT A FORECAST</span>
    <div class="top"><div><span class="badge syn">MODEL OUTPUT · REAL BRAIN ON SIMULATED PRICES</span><div class="title">${esc(S.title)}</div><div class="sum">${esc(S.summary)}</div></div></div>
    <div class="grid"><div>
        <div class="lab">${S.paths} simulated futures over ${S.horizon} sessions · both start at 100</div>
        <div class="key"><span><i style="background:${css("--up")}"></i>HANSEI, middle outcome</span><span><i style="background:${css("--hold")}"></i>Holding, middle outcome</span>
          <span><i style="background:${css("--up")};height:9px;opacity:.3"></i>8 in 10 of HANSEI's outcomes</span><span><i style="background:${css("--tan")};height:9px;opacity:.3"></i>shock arriving</span></div>
        <svg id="fan" role="img" aria-label="Range of simulated outcomes"></svg>
      </div><div>
        <div class="lab">How the shock reaches each stock · hover a line</div>
        <svg id="net" class="net" viewBox="0 0 420 330" role="img" aria-label="Sensitivity of each stock to each factor"></svg>
        <div class="key"><span><i style="background:${css("--up")}"></i>pushes the stock up</span><span><i style="background:${css("--down")}"></i>pushes it down</span><span><i style="background:${css("--flat")};height:2px"></i>dashed: cannot be told from zero</span></div>
      </div></div>
    <div class="score">
      <div class="sc" style="--c:${css("--up")}"><div class="l">HANSEI, middle outcome</div><div class="v">${pc(st.hansei.median_return)}</div><div class="s">typical worst fall ${pc(st.hansei.median_drawdown)}</div></div>
      <div class="sc" style="--c:${css("--hold")}"><div class="l">Holding, middle outcome</div><div class="v">${pc(st.holding.median_return)}</div><div class="s">typical worst fall ${pc(st.holding.median_drawdown)}</div></div>
      <div class="sc"><div class="l">HANSEI falls less</div><div class="v">${Math.round(st.share_hansei_falls_less * S.paths)} of ${S.paths}</div><div class="s">simulated futures</div></div>
      <div class="sc"><div class="l">HANSEI ends ahead</div><div class="v">${Math.round(st.share_hansei_ends_ahead * S.paths)} of ${S.paths}</div><div class="s">simulated futures</div></div>
    </div>
    <div class="notes">${S.notes.map(n => "· " + esc(n)).join("<br>")}</div>
    ${SENS && SENS.unavailable ? `<div class="lab" style="margin-top:18px">Shocks this cannot simulate, and why</div>` + SENS.unavailable.map(u => `<div class="na"><b>${esc(u.title)}</b><div>${esc(u.why)}</div></div>`).join("") : ""}</div>`;
  // fan
  const fan = $("fan"), W = fan.clientWidth || 700, H = fan.clientHeight || 340, m = {l: 44, r: 14, t: 14, b: 26}, n = S.days.length;
  fan.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const all = S.hansei.p10.concat(S.hansei.p90, S.holding.p10, S.holding.p90), lo0 = Math.min(...all), hi0 = Math.max(...all), pad = (hi0 - lo0 || 1) * .08, lo = lo0 - pad, hi = hi0 + pad;
  const X = i => m.l + i * (W - m.l - m.r) / (n - 1), Y = v => m.t + (hi - v) / (hi - lo) * (H - m.t - m.b);
  el("rect", {x: X(0), y: m.t, width: X(S.ramp) - X(0), height: H - m.t - m.b, fill: css("--tan"), opacity: .1}, fan);
  for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4; el("line", {x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "gridline"}, fan);
    const t = el("text", {x: m.l - 7, y: Y(v) + 4, "text-anchor": "end", class: "axis"}, fan); t.textContent = v.toFixed(0); }
  [0, 21, 42, 63, 84, 105, 126].filter(d => d < n).forEach(d => { const t = el("text", {x: X(d), y: H - 7, "text-anchor": "middle", class: "axis"}, fan); t.textContent = d === 0 ? "today" : "+" + d + " sessions"; });
  el("line", {x1: m.l, x2: W - m.r, y1: Y(100), y2: Y(100), stroke: css("--muted"), "stroke-dasharray": "2 5"}, fan);
  const line = arr => arr.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
  const band = (b, c) => el("path", {d: line(b.p90) + " " + b.p10.map((v, i) => [X(i), Y(v)]).reverse().map(p => `L${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ") + " Z", fill: c, opacity: .16}, fan);
  band(S.holding, css("--hold")); band(S.hansei, css("--up"));
  S.sample_paths.holding.forEach(p => el("path", {d: line(p), fill: "none", stroke: css("--hold"), "stroke-width": .7, opacity: .25}, fan));
  S.sample_paths.hansei.forEach(p => el("path", {d: line(p), fill: "none", stroke: css("--up"), "stroke-width": .7, opacity: .3}, fan));
  el("path", {d: line(S.holding.p50), fill: "none", stroke: css("--hold"), "stroke-width": 2.6}, fan);
  el("path", {d: line(S.hansei.p50), fill: "none", stroke: css("--up"), "stroke-width": 2.8}, fan);
  const cross = el("line", {y1: m.t, y2: H - m.b, stroke: css("--muted"), "stroke-dasharray": "3 4", opacity: 0}, fan);
  const hit = el("rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent"}, fan);
  hit.addEventListener("mousemove", ev => { const box = fan.getBoundingClientRect(), i = Math.max(0, Math.min(n - 1, Math.round(((ev.clientX - box.left) * (W / box.width) - m.l) / ((W - m.l - m.r) / (n - 1)))));
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("opacity", 1);
    showTip(ev, `<b>${i === 0 ? "today" : i + " sessions after the shock begins"}</b><br>HANSEI ${S.hansei.p50[i].toFixed(1)} (8 in 10 between ${S.hansei.p10[i].toFixed(1)} and ${S.hansei.p90[i].toFixed(1)})<br>Holding ${S.holding.p50[i].toFixed(1)} (${S.holding.p10[i].toFixed(1)} to ${S.holding.p90[i].toFixed(1)})`); });
  hit.addEventListener("mouseleave", () => { hideTip(); cross.setAttribute("opacity", 0); });
  // transmission network
  const net = $("net"), factors = ["market", "crude", "rupee"], stocks = S.stocks, fy = i => 70 + i * 95, sy = j => 36 + j * 64;
  if (SENS && SENS.table) {
    const links = el("g", {}, net);
    SENS.table.forEach(r => { const i = factors.indexOf(r.factor), j = stocks.findIndex(s => s.symbol === r.symbol); if (i < 0 || j < 0) return;
      const shocked = S.shock[r.factor] != null, d = `M118,${fy(i)} C210,${fy(i)} 210,${sy(j)} 300,${sy(j)}`;
      const p = el("path", {d, class: "lk", "data-f": r.factor, "data-s": r.symbol, stroke: !r.reliable ? css("--flat") : (r.beta > 0 ? css("--up") : css("--down")),
        "stroke-width": r.reliable ? Math.max(1.5, Math.min(14, Math.abs(r.beta) * 8)) : 1.2, "stroke-dasharray": r.reliable ? "none" : "4 4", opacity: shocked ? (r.reliable ? .8 : .55) : .14}, links);
      el("path", {d, fill: "none", stroke: "transparent", "stroke-width": 14, "data-f": r.factor, "data-s": r.symbol}, links).addEventListener("mousemove", ev => {
        net.querySelectorAll(".lk").forEach(q => q.classList.toggle("dim", q !== p));
        showTip(ev, `<b>${FLAB[r.factor]} → ${esc(r.name)}</b><br>a 10% rise moves it ${pc(r.beta * .1)}<br>plausible range ${pc(r.low * .1)} to ${pc(r.high * .1)}<br>${r.reliable ? "the data can tell this from zero" : "the data cannot tell this from zero"}`
          + (r.before != null ? `<br>before 2017: ${pc(r.before * .1)} · after: ${pc(r.after * .1)}` : "")); });
      links.lastChild.addEventListener("mouseleave", () => { hideTip(); net.querySelectorAll(".lk").forEach(q => q.classList.remove("dim")); }); });
  }
  factors.forEach((f, i) => { const shocked = S.shock[f] != null;
    el("rect", {x: 6, y: fy(i) - 24, width: 112, height: 48, rx: 10, fill: css("--bg"), stroke: shocked ? css("--tan") : css("--line"), "stroke-width": shocked ? 2 : 1}, net);
    const t = el("text", {x: 16, y: fy(i) - 4, class: "f"}, net); t.textContent = FLAB[f].replace(" ($)", "");
    const v = el("text", {x: 16, y: fy(i) + 13, class: "m", fill: shocked ? css("--tan") : css("--muted")}, net); v.textContent = shocked ? pc(S.shock[f], 0) : "held still"; });
  stocks.forEach((s, j) => { el("rect", {x: 300, y: sy(j) - 22, width: 114, height: 44, rx: 10, fill: css("--bg"), stroke: css("--line")}, net);
    const t = el("text", {x: 310, y: sy(j) - 3, class: "f"}, net); t.textContent = s.name;
    const v = el("text", {x: 310, y: sy(j) + 13, class: "m", fill: s.expected_move >= 0 ? css("--up") : css("--down")}, net); v.textContent = "expected " + pc(s.expected_move); });
  fit();
  window.onresize = () => { if (mode === "scenario") scenarioStage(S); };
}

document.querySelectorAll(".modes button").forEach(b => b.addEventListener("click", () => { stop(); mode = b.dataset.mode;
  document.querySelectorAll(".modes button").forEach(x => x.setAttribute("aria-pressed", x === b)); picks(); }));
new ResizeObserver(fit).observe($("root"));
picks();
</script></body></html>
"""


def stage_html(runs: list[dict], sensitivities: dict | None) -> str:
    """The Macroverse page for `components.html`, with every saved run embedded."""
    html = _TEMPLATE
    for key, value in {
        "__RUNS__": json.dumps(runs, ensure_ascii=False).replace("</", "<\\/"),
        "__SENS__": json.dumps(sensitivities, ensure_ascii=False).replace("</", "<\\/"),
        "__FONTS__": T.FONT_IMPORT, "__FONT__": T.FONT, "__MONO__": T.MONO, "__BG__": T.BG, "__PANEL__": T.PANEL,
        "__RAISED__": T.RAISED, "__LINE__": T.LINE, "__INK__": T.INK, "__SOFT__": T.SOFT, "__MUTED__": T.MUTED,
        "__UP__": T.UP, "__DOWN__": T.DOWN, "__FLAT__": T.FLAT, "__TAN__": T.TAN, "__HOLD__": T.HOLD,
    }.items():
        html = html.replace(key, value)
    return html
