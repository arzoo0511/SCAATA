"""The Research Lab page: every experiment on file, how it was judged, and
what the live record is asking next.

It only reads `research/experiments.jsonl` and `research/agenda.json`. It
cannot start, edit or delete an experiment.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from scaata.config import ROOT_DIR
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


def _how_much_to_trust(records: list[dict], lab: list[dict]) -> None:
    from scaata.research import audit, review, stats

    T.heading("How much to trust these numbers", "the discount for trying many things")
    T.caption("Try forty variants and the best of them will look good by luck alone. These tables discount for "
              "that. They are cautious: the variants tried here are mostly small tweaks of one rule, so the "
              "number of truly independent tries is much smaller than the number of tries.")
    rows = audit.dsr_table(records)
    span = st.segmented_control("Span", list(audit.SPAN_NAMES), default="holdout", format_func=audit.SPAN_NAMES.get,
                                label_visibility="collapsed", key="dsr_span") or "holdout"
    _plot(charts.dsr_bars(rows, span), f"dsr_{span}")
    T.caption("Grey: the chance each variant&rsquo;s true Sharpe really exceeds holding&rsquo;s, taken alone. "
              "Blue: the same after discounting for every variant tried in this span (the Deflated Sharpe "
              "Ratio). The dashed line is 95%. <b>None reaches it.</b> The rule is probably a little better than "
              "holding on Sharpe, and nothing here is strong enough to be sure.")

    left, right = st.columns(2, gap="large")
    with left:
        pbo = {r["span"]: r for r in audit.pbo_table(records)}
        cells = "".join(
            f'<div class="hs-row"><div><div class="a">{audit.SPAN_NAMES[s]}</div><div class="b">'
            + (f'{pbo[s]["variants"]} variants, {pbo[s]["splits"]} train/test splits' if pbo[s].get("pbo") is not None
               else pbo[s].get("reason", "n/a")) + "</div></div>"
            + (f'<div class="mono" style="font-size:1.3rem;color:{T.INK}">{pbo[s]["pbo"]:.0%}</div>'
               if pbo[s].get("pbo") is not None else "") + "</div>" for s in audit.SPANS if s in pbo)
        st.markdown('<div class="hs-eyebrow">Chance that picking the past winner fails</div>'
                    f'<div class="hs-panel">{cells}</div>', unsafe_allow_html=True)
        T.caption("Cut the history in pieces, pick whichever variant did best on some of them, and see where it "
                  "ranks on the rest (Bailey et al.). Near 50% means the past winner is no better than a coin "
                  "flip. With so few, so similar variants this is a smoke alarm, not a measurement.")
    with right:
        ledger = review.budget_ledger(lab)
        wealth = ledger[-1]["wealth_after"] if ledger else stats.INITIAL_WEALTH
        share = wealth / stats.INITIAL_WEALTH
        st.markdown('<div class="hs-eyebrow">Evidence budget</div><div class="hs-panel">'
                    f'<div class="hs-row"><div><div class="a">Left to spend</div><div class="b">of {stats.INITIAL_WEALTH:.2f} '
                    f'false-alarm risk at the start; each test uses half of what is left, and a pass earns '
                    f'{stats.PAYOUT:.2f} back</div></div><div class="mono" style="font-size:1.3rem;color:'
                    f'{T.INK if share > .25 else T.DOWN}">{wealth:.4f}</div></div>'
                    f'<div class="hs-row"><div><div class="a">Next test may use</div><div class="b">a false-alarm '
                    f'level of {stats.offered_level(wealth):.4f}, so it needs {1 - stats.offered_level(wealth):.2%} '
                    f'confidence</div></div></div></div>', unsafe_allow_html=True)
        T.caption("Every experiment spends from one budget, so testing more ideas makes each later pass harder. "
                  "<b>It is nearly empty on the 2020-onward years, which have been looked at many times.</b> "
                  "Further work on them cannot confirm anything; it has to be confirmed on data that arrives "
                  "after it is filed.")


def _self_critique() -> None:
    """Phase A: does the system's own critique work? From research/phase_a.json and contribution.json."""
    from scaata.dashboard.ui.data import read_json

    study = read_json(ROOT_DIR / "research" / "phase_a.json")
    if not study:
        return
    T.heading("Does its self-critique work?", f"{study['judged_days']:,} judged days since {study['first'][:4]}")
    T.caption("The live record has 14 judged days. This replays every advisor's view on every stock since 2007 "
              "and what each stock then did over the next 20 sessions against cash: about 4,800 days. A learner's "
              "forecasts are out of sample by construction (the trust used on a day was learned only from "
              "outcomes at least 20 sessions old), so the whole history is a fair test of a learner whose "
              "settings were fixed in advance. " + study["note"])

    poison = study["poison"]["live"]
    st.markdown('<div class="hs-eyebrow">The poison test: can it spot a bad advisor?</div>', unsafe_allow_html=True)
    order = ["adversary_demoted", "weak_adversary_below_equal", "random_not_promoted", "informative_top2"]
    rows = "".join(
        f'<div class="hs-row"><div><div class="a">{poison["rules"].get(key.replace("_demoted", "").replace("_below_equal", "").replace("_not_promoted", "").replace("_top2", ""), key)}'
        f'</div></div><div style="display:flex;gap:10px;align-items:center"><span class="mono" style="font-size:1.15rem;'
        f'color:{T.INK}">{poison["rates"][key]:.0%}</span>'
        + (T.chip("passes", "live") if poison["passes"][key] else T.chip("fails", "off")) + "</div></div>"
        for key in order if key in poison["rates"])
    st.markdown(f'<div class="hs-panel">{rows}</div>', unsafe_allow_html=True)
    T.caption(f"Four advisors were planted in the real history, built with knowledge of the future so their "
              f"quality is known: one always wrong, one wrong 60% of the time, one pure noise, one right 60% of the "
              f"time. The live learner was run over {poison['seeds']} different draws and had to get each right in "
              f"at least {poison['required']:.0%}. <b>It did, in every draw.</b> This is a standing check: a "
              "failure should block any change to the learner. (Mean final trust: "
              + ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in poison["mean_trust"].items()) + ".)")

    st.markdown('<div class="hs-eyebrow" style="margin-top:1.4rem">Does learning trust beat equal trust?</div>',
                unsafe_allow_html=True)
    metric = st.segmented_control("Score", ["brier", "payoff"], default="payoff", key="learner_metric",
                                  format_func={"brier": "Forecast score (Brier)", "payoff": "Following the sign"}.get,
                                  label_visibility="collapsed") or "payoff"
    _plot(charts.learner_scoreboard(study, metric), f"learners_{metric}")
    if metric == "brier":
        T.caption("The proper score of the combined forecast, treating a view of +1 as 75% that the stock beats "
                  "cash. <b>Every learner is significantly worse than equal trust.</b> The reason is not that they "
                  "pick worse directions: they concentrate trust on the advisor with the biggest views, which "
                  "makes the blend more confident than its record entitles it to be, and this score punishes "
                  "that. Switch to &ldquo;Following the sign&rdquo; to see direction alone.")
    else:
        T.caption("What following the combined view&rsquo;s sign would have earned, ignoring how large the view is. "
                  "<b>No learner can be told apart from equal trust</b>: every interval includes zero, and AdaHedge "
                  "is slightly worse. Toggle the spans in the legend; the sign of the difference changes from "
                  "one span to the other.")

    left, right = st.columns(2, gap="large")
    with left:
        span = st.segmented_control("Years", ["tune", "holdout"], default="tune", key="reliab_span",
                                    format_func={"tune": "2007–2019", "holdout": "2020 on"}.get,
                                    label_visibility="collapsed") or "tune"
        _plot(charts.reliability_diagram(study, span), f"reliability_{span}")
    with right:
        cal = study["calibration"]["equal blend"]
        sw, rc = cal["strong_vs_weak"], cal.get("recalibration")
        cells = "".join(
            f'<div class="hs-row"><div><div class="a">{label}</div><div class="b">weaker half of views right '
            f'{sw[s]["weak_right"]:.1%}, stronger half {sw[s]["strong_right"]:.1%}; gap {sw[s]["gap"]:+.1%} '
            f'(95%: {sw[s]["low"]:+.1%} to {sw[s]["high"]:+.1%})</div></div></div>'
            for s, label in (("tune", "2007–2019"), ("holdout", "2020 on")))
        st.markdown('<div class="hs-eyebrow">Does a stronger view mean a more reliable one?</div>'
                    f'<div class="hs-panel">{cells}</div>', unsafe_allow_html=True)
        T.caption("In the first span the stronger views were slightly <i>less</i> reliable; in the second, more. "
                  "Neither gap can be told from zero for the blend. <b>The inverted confidence seen in the live "
                  "record is not a stable pattern.</b>"
                  + (f" Re-fitting the mapping on 2007–2019 and scoring it on 2020 onward improves the Brier score "
                     f"by {rc['improvement']*1000:.1f} thousandths, to {rc['brier_recalibrated']:.4f}: "
                     "a coin flip scores 0.2500. Recalibrating does not find skill; it removes the over-confidence."
                     if rc else ""))

    skill = [r for r in study["skill"] if r["span"] in ("tune", "holdout") and r.get("skill") is not None]
    if skill:
        st.markdown('<div class="hs-eyebrow" style="margin-top:1.2rem">How good is each advisor on its own?</div>',
                    unsafe_allow_html=True)
        table = pd.DataFrame([{"Advisor": r["source"], "Years": {"tune": "2007–2019", "holdout": "2020 on"}[r["span"]],
                               "Views": r["opinions"], "Right": r["hit_rate"] * 100,
                               "Range": f"{r['hit_low']:.0%}–{r['hit_high']:.0%}", "Brier skill": r["skill"]}
                              for r in skill])
        st.dataframe(table, hide_index=True, use_container_width=True, column_config={
            "Right": st.column_config.NumberColumn("Right", format="%.1f%%"),
            "Brier skill": st.column_config.NumberColumn("Brier skill vs a coin flip", format="%+.3f"),
            "Views": st.column_config.NumberColumn("Views", format="%d")})
        T.caption("Each advisor is right on 46–53% of its opinions, with ranges that include 50%, and its views are "
                  "far more confident than that: negative skill means worse than always saying 50%. "
                  "<b>On 19 years of data no advisor has forecast skill that can be told from a coin flip.</b> "
                  "Their value, where they have any, is in the risk rules built on them, not in prediction.")

    cont = read_json(ROOT_DIR / "research" / "contribution.json")
    if cont:
        _contribution(cont)


def _rebound() -> None:
    """Phase B: the one published fix for returning late after a fall, tested and described."""
    from scaata.dashboard.ui.data import read_json

    study = read_json(ROOT_DIR / "research" / "rebound.json")
    if not study:
        return
    s = study["summary"]
    T.heading("Does it come back sooner after a fall?", f"{s['episodes']} falls, one rule")
    T.caption("HANSEI saves money as a fall begins and gives some back by returning late. The published fix "
              "(Goulding, Harvey and Mazzoleni 2023) is to re-enter when a stock&rsquo;s last month has turned up "
              "while its slow 200-day trend is still down. It was tried as one rule with settings taken from the "
              "paper and the live agent; <b>nothing was searched</b>. Each bar is the share of holding&rsquo;s gain "
              "from its own low that HANSEI made.")
    _plot(charts.rebound_capture(study), "rebound_capture")
    table = pd.DataFrame([{
        "Fall": r["title"], "Live rule": r["live_rule"]["return"] * 100, "Rebound rule": r["rebound"]["return"] * 100,
        "Holding": r["holding"]["return"] * 100, "Live worst fall": r["live_rule"]["worst_fall"] * 100,
        "Rebound worst fall": r["rebound"]["worst_fall"] * 100} for r in study["episodes"]])
    st.dataframe(table, hide_index=True, use_container_width=True, column_config={
        c: st.column_config.NumberColumn(c, format="%.1f%%") for c in table.columns[1:]})
    T.caption(f"It captured more of the rebound in <b>{s['captures_more']} of {s['with_a_rebound']}</b> falls, by "
              f"small amounts (on average {s['mean_capture_live']:.0%} to {s['mean_capture_rebound']:.0%}), but its "
              f"total return is mixed and in <b>{s['episodes'] - s['falls_no_worse']}</b> of {s['episodes']} the worst "
              "fall was more than 2 points worse. Over the two long spans the Lab found it <b>not supported</b>: "
              "Sharpe &minus;0.05 in 2007&ndash;2019 and +0.02 since, with 15&ndash;25% more trades. Re-entering "
              "sooner also means re-entering into bear-market rallies. It is <b>not switched on</b>. It is "
              "recorded every day as a shadow advisor, scored but never acted on, so forward data can say more.")


def _universe() -> None:
    """Phase C: a model that learns from scratch over ~200 stocks, tested, and the wide paper book that follows it."""
    from scaata.dashboard.ui.data import read_json

    study = read_json(ROOT_DIR / "research" / "universe_study.json")
    if not study:
        return
    T.heading("A model that learns from scratch", f"{study['stocks']} stocks")
    T.caption("Everything above uses five stocks. Here the same idea meets about two hundred: each week a model "
              "ranks them using eight price signals, learning how to weigh the signals only from outcomes that had "
              "already finished (it is never told which signals are good). It is judged by rank correlation with "
              "what the stocks then did. <b>The stock list is today&rsquo;s large and mid caps, so past results are "
              "flattered by survivors.</b>")
    _plot(charts.universe_ic(study), "universe_ic")
    ic, vs = study["ic"]["all"], study["versus"]["all"]
    v = vs.get("ridge_minus_equal", {})
    T.caption(f"Average weekly rank correlation, 2010 on: the learned model {ic['ridge']['mean']:+.3f}, a plain average "
              f"of the signals {ic['equal']['mean']:+.3f}, plain 12-1 month momentum {ic['mom_12_1']['mean']:+.3f}. "
              f"There is a small real signal. <b>The learning added nothing over the plain average</b> "
              f"({v.get('mean', 0):+.4f}, p={v.get('p')}), so the model has not earned trust beyond simple momentum.")
    state = read_json(ROOT_DIR / "paper_book_wide.json")
    if not state or not state["sleeves"]["universe"]["history"]:
        return
    T.heading("The wide paper book", f"fake money, since {state['started']}")
    _plot(charts.wide_book(state), "wide_book")
    fi = pd.DataFrame(state.get("forward_ic", []))
    live = (" Live rank correlation so far, " + ", ".join(f"{m} {fi[m].mean():+.3f}" for m in ("learned", "features", "momentum")
            if m in fi) + f" over {len(fi)} finished weeks.") if len(fi) else " No weekly forecast has finished yet."
    T.caption("Top fifth of the ranking, equal-weighted, rebalanced monthly with Indian costs, judged against owning "
              "every stock. It starts on its first day and is never back-filled; it shares nothing with the "
              "five-stock book." + live)


def _generated() -> None:
    """Phase E: the system's own batch of experiments, judged as a family with the holdout sealed."""
    from scaata.dashboard.ui.data import read_json

    g = read_json(ROOT_DIR / "research" / "generated.json")
    if not g:
        return
    rc, b = g["reality_check"], g["baseline"]
    T.heading("Experiments it proposed itself", f"{g['tried']} tried together, {g['batch']}")
    T.caption("Besides the questions the live record raises, the system tries changes to the brain&rsquo;s own settings. "
              "Each carries a written reason and cost before it runs. They run on 2007&ndash;2019 only: the 2020-on "
              "years are <b>sealed</b>, because every earlier experiment has already looked at them and the evidence "
              "budget is spent. Because twelve were tried, the best one is judged by White&rsquo;s Reality Check: how "
              "often the best of twelve duds would look this good.")
    _plot(charts.generated_family(g), "generated_family")
    win = g["champion"]
    T.caption(f"Against the live rule (Sharpe {b['sharpe']:.2f}), the best by Sharpe was <b>{g['best_by_sharpe']}</b>, by a hair. "
              f"Reality-check p (on daily return gains, best of all {g['tried']}) = <b>{rc['p']:.2f}</b>: luck alone gives a best of {rc['luck_only_median']:.4f} "
              f"(95% of the time under {rc['luck_only_95']:.4f}) against the {rc['observed']:.4f} seen. "
              + (f"<b>Champion: {win}</b>, proposed for forward testing only; nothing is switched on."
                 if win else "<b>No champion.</b> Nothing in the family beats chance, so no setting change is proposed."))
    with st.expander("Each candidate's reason and cost"):
        for r in g["candidates"]:
            st.markdown(f"**{r['id']}** &mdash; {r['reason']} *Cost:* {r['cost']}")


def _contribution(cont: dict) -> None:
    st.markdown('<div class="hs-eyebrow" style="margin-top:1.2rem">What does each advisor add to the whole '
                'system?</div>', unsafe_allow_html=True)
    span = st.segmented_control("Years ", ["tune", "holdout"], default="holdout", key="contrib_span",
                                format_func={"tune": "2007–2019", "holdout": "2020 on"}.get,
                                label_visibility="collapsed") or "holdout"
    d = cont["spans"][span]
    rows = "".join(
        f'<div class="hs-row"><div><div class="a">{a.capitalize()}</div><div class="b">Sharpe {d["shapley"]["sharpe"][a]:+.3f} · '
        f'return {d["shapley"]["cagr"][a]:+.2%} a year · worst fall {d["shapley"]["max_drawdown"][a]:+.1%}</div></div></div>'
        for a in cont["players"])
    st.markdown(f'<div class="hs-panel">{rows}</div>', unsafe_allow_html=True)
    T.caption(f"Each advisor&rsquo;s Shapley value: its average effect over every order in which advisors could "
              f"have been added to the real brain, risk rules included. With nobody speaking the brain just holds "
              f"(Sharpe {d['baseline']['sharpe']:.2f}, worst fall {d['baseline']['max_drawdown']:.0%}); with all "
              f"three it is the live rule (Sharpe {d['full']['sharpe']:.2f}, worst fall {d['full']['max_drawdown']:.0%}). "
              "The values add up to exactly that difference. News is left out: it never speaks in this history.")


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

    from scaata.research.proposals import build as build_proposals

    proposals = build_proposals(records)
    if proposals:
        T.heading("From finding to change", "the only road to the live agent")
        T.caption("A proposal is a possible change to how HANSEI trades, tied to the experiment that tests "
                  "exactly that change. <b>Nothing here is ever applied automatically.</b> &ldquo;Ready&rdquo; "
                  "means a person may now decide; the change is still made by hand.")
        words = {"ready": ("ready for a decision", "live"), "inconclusive": ("inconclusive", "part"),
                 "not supported": ("not supported", "off"), "needs a test": ("needs a test", "plan")}
        rows = ""
        for item in proposals:
            effect = ""
            if item.get("effect"):
                effect = (" " + (item.get("effect_label") or "Effect") + ": "
                          + ", ".join(f"{SPANS.get(k, k)} {v:+.3f}" for k, v in item["effect"].items()) + ".")
            rows += (f'<div class="hs-row"><div><div class="a">{item["change"]}</div><div class="b">'
                     f'<b>Why it was raised.</b> {item["why"]}<br><b>Tested by.</b> '
                     f'{item.get("experiment_title") or "an experiment not yet run"}.{effect}'
                     + (f'<br><b>How to read it.</b> {item["reading"]}' if item.get("reading") else "")
                     + f'</div></div><div style="display:flex;flex-direction:column;gap:6px;align-items:flex-end">'
                     + T.chip(*words[item["status"]]) + (T.chip("live now", "live") if item["applied"] else "")
                     + "</div></div>")
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

    if lab:
        _how_much_to_trust(records, lab)
    _self_critique()
    _rebound()
    _universe()
    _generated()

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
