"""The Research Lab page: every experiment on file, how it was judged, and
what the live record is asking next.

It only reads `research/experiments.jsonl` and `research/agenda.json`. It
cannot start, edit or delete an experiment.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from scaata.dashboard.ui import charts
from scaata.dashboard.ui import theme as T

CHIP = {"supported": "live", "not_supported": "off", "inconclusive": "part", "failed_review": "off",
        "ran": "plan", "registered": "plan"}
ORDER = ["supported", "inconclusive", "not_supported", "failed_review", "ran", "registered"]
SPANS = {"tune": "2007–2019", "holdout": "2020 onward"}


def _plot(fig, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, config=charts.CONFIG, key=key)


def metric(name: str, value) -> str:
    """A metric in the unit its name implies."""
    if value is None:
        return "n/a"
    if name.endswith(("sessions", "opinions", "settings", "settings_better")):
        return f"{value:,.0f}"
    if "prob" in name or "hit_rate" in name or "share" in name:
        return f"{value:.0%}"
    if "sharpe" in name:
        return f"{value:+.2f}"
    return f"{value:+.2%}"


def verdict_chip(verdict: str) -> str:
    return T.chip(charts.VERDICT_WORDS.get(verdict, verdict), CHIP.get(verdict, ""))


def pipeline_html(agenda: list[dict], lab: list[dict]) -> str:
    """The path every idea takes, how many are at each stage, and how the reviewed ones ended."""
    ran = [r for r in lab if r.get("results") is not None]
    reviewed = [r for r in lab if r.get("review")]
    stages = [("Questions raised", len(agenda), "by the live record"),
              ("Filed in advance", len(lab), "criteria written first"),
              ("Run", len(ran), "real brain · frozen data"),
              ("Reviewed", len(reviewed), "by the devil&rsquo;s advocate")]
    cells = []
    for i, (name, count, note) in enumerate(stages):
        cells.append(
            f'<div style="flex:1 1 150px;padding:14px 16px;border:1px solid {T.LINE};border-radius:12px;'
            f'background:{T.BG}"><div style="font-family:{T.MONO};font-size:2rem;color:{T.INK};line-height:1.1">'
            f'{count}</div><div style="font-weight:600;color:{T.INK};font-size:.9rem">{name}</div>'
            f'<div style="color:{T.MUTED};font-size:.74rem">{note}</div></div>')
        if i < len(stages) - 1:
            cells.append(f'<div style="align-self:center;color:{T.MUTED};font-size:1.2rem">→</div>')
    counts = {v: sum(1 for r in reviewed if r["review"]["verdict"] == v) for v in ORDER[:4]}
    bar = "".join(f'<div title="{charts.VERDICT_WORDS[v]}: {n}" style="flex:{n};background:'
                  f'{charts.VERDICT_COLOURS[v]};height:12px;border-radius:3px"></div>' for v, n in counts.items() if n)
    legend = " ".join(f'<span style="white-space:nowrap;margin-right:1rem"><span class="hs-sw" style="background:'
                      f'{charts.VERDICT_COLOURS[v]}"></span> {charts.VERDICT_WORDS[v]} <b style="color:{T.INK}">{n}'
                      f'</b></span>' for v, n in counts.items() if n)
    verdicts = (f'<div style="margin-top:16px"><div style="display:flex;gap:3px">{bar}</div>'
                f'<div style="margin-top:8px;color:{T.SOFT};font-size:.82rem">{legend}</div></div>' if reviewed else "")
    return (f'<div class="hs-panel"><div style="display:flex;gap:10px;flex-wrap:wrap;align-items:stretch">'
            f'{"".join(cells)}</div>{verdicts}</div>')


def _experiment(record: dict, status: str) -> None:
    results, verdict = record.get("results") or {}, record.get("review") or {}
    st.markdown(f'<div class="hs-lede" style="font-size:.95rem"><b>Question.</b> {record["question"]}<br>'
                f'<b>Hypothesis, written before the run.</b> {record["hypothesis"]}</div>'
                f'<div class="hs-note" style="margin:.8rem 0">{verdict_chip(status)} &nbsp;'
                f'{verdict.get("summary", "Not reviewed yet.")}</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="hs-eyebrow">What it promised</div>', unsafe_allow_html=True)
        rows = "".join(
            f'<div class="hs-row"><div><div class="a">{c.get("label") or c["metric"]}</div><div class="b">needed '
            f'{c["op"]} {metric(c["metric"], c["value"])} · got <b style="color:{T.INK}">'
            f'{metric(c["metric"], c.get("observed"))}</b></div></div>'
            + (T.chip("met", "live") if c.get("passed") else T.chip("missed", "off")) + "</div>"
            for c in verdict.get("criteria", []))
        st.markdown(f'<div class="hs-panel">{rows or "No criteria evaluated."}</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="hs-eyebrow">What the reviewer checked</div>', unsafe_allow_html=True)
        rows = "".join(
            f'<div class="hs-row"><div><div class="a">{c["name"].replace("_", " ").capitalize()}</div>'
            f'<div class="b">{c["detail"]}</div></div>'
            + (T.chip("pass", "live") if c["passed"] else T.chip("n/a", "plan") if c["passed"] is None
               else T.chip("fail", "off")) + "</div>" for c in verdict.get("checks", []))
        st.markdown(f'<div class="hs-panel">{rows or "No checks recorded."}</div>', unsafe_allow_html=True)

    if results.get("regimes"):
        _plot(charts.regime_bars(results["regimes"]), f"regimes_{record['id']}")
        T.caption("HANSEI&rsquo;s yearly return minus holding&rsquo;s inside each market condition. Faded bars are "
                  "2007–2019; solid bars are 2020 onward. Hover for the session counts.")
    if results.get("curves"):
        span = st.segmented_control("Span", list(SPANS), default="holdout", format_func=SPANS.get,
                                    label_visibility="collapsed", key=f"span_{record['id']}") or "holdout"
        _plot(charts.experiment_curves(results["curves"], span, record["kind"] == "baseline"),
              f"curve_{record['id']}")
    stats = results.get("periods") or {}
    if stats:
        table = pd.DataFrame([{
            "Span": SPANS.get(p, p), "This run / yr": r["variant"]["cagr"] * 100,
            "Live rule / yr": r["baseline"]["cagr"] * 100, "Holding / yr": r["holding"]["cagr"] * 100,
            "This run Sharpe": r["variant"]["sharpe"], "Live rule Sharpe": r["baseline"]["sharpe"],
            "Holding Sharpe": r["holding"]["sharpe"], "This run worst fall": r["variant"]["max_drawdown"] * 100,
            "Holding worst fall": r["holding"]["max_drawdown"] * 100, "Trades": r["trades"]}
            for p, r in stats.items()])
        formats = {c: st.column_config.NumberColumn(c, format="%.1f%%") for c in table if "/ yr" in c or "fall" in c}
        formats.update({c: st.column_config.NumberColumn(c, format="%.2f") for c in table if "Sharpe" in c})
        st.dataframe(table, hide_index=True, use_container_width=True, column_config=formats)

    snapshot, engine = results.get("data") or {}, results.get("engine") or {}
    reproduce = ""
    if snapshot.get("sha256"):
        reproduce = (f"<br><b>Reproduce.</b> Price snapshot <code>{snapshot['sha256'][:12]}</code> through "
                     f"{snapshot.get('last')}; code at <code>{(engine.get('commit') or 'unknown')[:10]}</code>"
                     + (" plus uncommitted changes" if engine.get("dirty") else "")
                     + f"; filed {record['registered_at'][:16]}Z, ran {(record.get('ran_at') or '')[:16]}Z.")
    T.foot("<b>Method.</b> " + record["method"] + "<br><b>Limits.</b> " + " ".join(record.get("limitations", []))
           + reproduce)


def page() -> None:
    from scaata.research import director, registry

    records, agenda = registry.load(), director.load_agenda()
    lab = [r for r in records if r.get("origin") == "lab"]
    legacy = [r for r in records if r.get("origin") == "legacy"]
    T.intro("Research Lab", "Every idea has to survive a hostile reviewer.",
            "An idea becomes an experiment only when its pass/fail criteria are written down first. It is then "
            "run through HANSEI&rsquo;s real brain on a frozen copy of the price history, and handed to a "
            "devil&rsquo;s advocate made of code: did it peek at the future, pay its costs, hold up in years it "
            "was never tuned on, beat noise? <b>Nothing here can change how HANSEI trades.</b>")
    if not records:
        st.markdown('<div class="hs-note"><b>No experiments on file yet.</b> Run '
                    "<code>python -m scaata.research seed</code>, then "
                    "<code>python -m scaata.research run --proposed</code>.</div>", unsafe_allow_html=True)
        return
    T.heading("The pipeline", "counts are experiments")
    st.markdown(pipeline_html(agenda, lab), unsafe_allow_html=True)

    if lab:
        T.heading("The evidence map", "hover a dot")
        _plot(charts.evidence_map(lab), "evidence_map")
        T.caption("Each dot is one experiment. Across: how much it changed Sharpe in 2007–2019, the years the "
                  "rule was tuned on. Up: the same in 2020 onward. A real effect points the same way in both "
                  "(the shaded corners); a dot elsewhere worked in one span only. The baseline is measured "
                  "against holding, the others against the live rule.")

        T.heading("Experiments", f"{len(lab)} filed through the Lab")
        present = [v for v in ORDER if any(registry.status(r) == v for r in lab)]
        chosen = st.pills("Verdict", present, selection_mode="multi", default=present,
                          format_func=charts.VERDICT_WORDS.get, label_visibility="collapsed",
                          key="lab_verdicts") or present
        for record in sorted(lab, key=lambda r: ORDER.index(registry.status(r))):
            status = registry.status(record)
            if status in chosen:
                with st.expander(f"{charts.VERDICT_WORDS[status].upper()}  ·  {record['title']}"):
                    _experiment(record, status)

    if agenda:
        T.heading("What the live record is asking", "raised by rules over the daily report card")
        words = {"answered": ("answered", "live"), "proposed": ("proposed · not run", "part"),
                 "needs engine work": ("cannot be tested yet", "plan")}
        rows = ""
        for a in agenda:
            answer = (f' <b>Verdict: {charts.VERDICT_WORDS.get(a["verdict"], a["verdict"]).lower()}.</b>'
                      if a.get("verdict") else "")
            rows += (f'<div class="hs-row"><div><div class="a">{a["question"]}</div><div class="b">{a["trigger"]}'
                     f'{" " + a["note"] if a.get("note") else ""}{answer}</div></div>'
                     f'{T.chip(*words.get(a["status"], (a["status"], "")))}</div>')
        st.markdown(f'<div class="hs-panel">{rows}</div>', unsafe_allow_html=True)

    if legacy:
        T.heading("Before the Lab", "entered so they are never repeated")
        T.caption("These were run before experiments were filed in advance, so they did not go through the "
                  "reviewer. Their numbers are read from the original result files.")
        for record in legacy:
            status = registry.status(record)
            with st.expander(f"{charts.VERDICT_WORDS[status].upper()}  ·  {record['title']}"):
                st.markdown(f'<div class="hs-lede" style="font-size:.95rem"><b>Question.</b> {record["question"]}'
                            f'<br><b>Method.</b> {record["method"]}</div>'
                            f'<div class="hs-note" style="margin:.8rem 0">{verdict_chip(status)} &nbsp;'
                            f'{record["review"]["summary"]}</div>', unsafe_allow_html=True)
                T.foot(f'Source: <code>{record.get("source", "")}</code> · dated {record["registered_at"][:10]}')

    T.heading("What the Lab cannot do")
    rules = [("Change the live agent", "A supported finding is a proposal. It reaches the trading code only as a "
              "reviewed change that also passes the backtest gate."),
             ("Rewrite a result", "The registry is append-only. A re-run is a new experiment with a new id."),
             ("Move its own goalposts", "Criteria are filed before the run and cannot be edited afterwards."),
             ("Forget a failure", "Failed experiments stay on file, and each new look at the holdout years "
              "raises the bar for the next one.")]
    st.markdown('<div class="hs-panel">' + "".join(
        f'<div class="hs-row"><div><div class="a">{a}</div><div class="b">{b}</div></div>{T.chip("never", "off")}</div>'
        for a, b in rules) + "</div>", unsafe_allow_html=True)
