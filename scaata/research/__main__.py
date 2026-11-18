"""The Research Lab's command line.

    python -m scaata.research refresh          download prices and freeze a new snapshot
    python -m scaata.research seed             enter the experiments run before the Lab existed
    python -m scaata.research agenda           what the live record says is worth testing
    python -m scaata.research run ID [ID ...]  register, run and review proposed experiments
    python -m scaata.research run --proposed   ... every proposed experiment not yet run
    python -m scaata.research list             every experiment on file and its verdict
    python -m scaata.research events           what each stock did after each kind of news
    python -m scaata.research rebound          the rebound rule, episode by episode
    python -m scaata.research contribution     each advisor's Shapley value in the whole system (~10 min)
    python -m scaata.research generate         the system's own batch of experiments, judged as a family (~15 min)
    python -m scaata.research race             the promotion race: frozen baseline, live rule and challengers
    python -m scaata.research phase-a          skill ledger, calibration, learners and the poison test

An experiment is registered (criteria and all) before it runs, then run on
the frozen snapshot, then reviewed. Nothing here touches the live paper
book, journal or memory.
"""
from __future__ import annotations

import sys

from scaata.research import director, legacy, prices, registry
from scaata.research.review import review


def _agenda() -> list[dict]:
    from scaata.agent.daily import load_journal
    from scaata.dashboard.ui.data import EVALUATION_PATH, read_json

    agenda = director.build_agenda(read_json(EVALUATION_PATH), load_journal(), registry.load(),
                                   (prices.meta() or {}).get("last"))
    director.save_agenda(agenda)
    return agenda


def run(ids: list[str]) -> None:
    from scaata.research.engine import Engine

    specs, engine = director.specs((prices.meta() or {}).get("last")), None
    for experiment_id in ids:
        if experiment_id not in specs:
            sys.exit(f"unknown experiment '{experiment_id}'. Known: {', '.join(specs)}")
        record = registry.get(experiment_id)
        if record and record.get("results") is not None:
            print(f"{experiment_id}: already run ({registry.status(record)}); results are never overwritten")
            continue
        if record is None:
            registry.register(specs[experiment_id])
            print(f"{experiment_id}: registered")
        engine = engine or Engine()
        registry.record_result(experiment_id, engine.run(specs[experiment_id]))
        record = registry.get(experiment_id)
        verdict = review(record, registry.load())
        registry.record_review(experiment_id, verdict)
        print(f"{experiment_id}: {verdict['verdict'].upper()} -- {verdict['summary']}")
        for c in verdict["checks"]:
            mark = "ok  " if c["passed"] else ("n/a " if c["passed"] is None else "FAIL")
            print(f"    {mark} {c['name']}: {c['detail']}")
    _agenda()


def main(argv: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    command = argv[0] if argv else "list"
    if command == "refresh":
        info = prices.refresh()
        print(f"snapshot: {info['sessions']} sessions, {info['first']} to {info['last']}, sha256 {info['sha256'][:12]}")
    elif command == "seed":
        added = legacy.seed()
        print(f"entered {len(added)} legacy experiment(s): {', '.join(added) or 'none new'}")
        _agenda()
    elif command == "agenda":
        for item in _agenda():
            print(f"[{item['status']}] {item['question']}\n    because: {item['trigger']}"
                  + (f"\n    experiment: {item['experiment']}" if item["experiment"] else ""))
    elif command == "race":
        from scaata.research import race

        out = race.run()
        for name in ("forward", "rehearsal"):
            r = out[name]
            if not r:
                print(f"{name}: not enough sessions yet")
                continue
            print(f"{name}: {r['start']} to {r['end']} ({len(r['days']) - 1} sessions)")
            for e in r["entrants"]:
                tail = f"  e-up {e['e_up']:.2f} e-down {e['e_down']:.2f}  -> {e['promotion']['verdict']}" if "promotion" in e else ""
                print(f"  {e['id']:<12} return {e['return']:+.1%}  Sharpe {e['sharpe']:.2f}  worst fall {e['max_drawdown']:.1%}{tail}")
    elif command == "generate":
        from scaata.research import generator
        from scaata.research.engine import Engine

        if generator.already_run(generator.quarter()):
            sys.exit(f"batch {generator.quarter()} was already run; one batch per quarter")
        out = generator.run(Engine(), registry.load())
        rc, b = out["reality_check"], out["baseline"]
        print(f"{out['tried']} candidates on {out['span'][0]} to {out['span'][1]} (holdout sealed). Live rule: Sharpe {b['sharpe']:.3f}")
        for r in sorted(out["candidates"], key=lambda r: -r["sharpe_vs_baseline"]):
            print(f"  {r['id']:<15} Sharpe {r['sharpe_vs_baseline']:+.3f}  return {r['cagr_vs_baseline']:+.2%}  worst fall "
                  f"{r['drawdown_vs_baseline']:+.1%}  trades {r['extra_trades']:+.0%}  {'PASSES' if r['passes'] else ''}")
        print(f"Reality check: best of {rc['candidates']} has p={rc['p']:.3f}; luck alone gives a best of {rc['luck_only_median']:.4f} "
              f"(95%: {rc['luck_only_95']:.4f}) vs observed {rc['observed']:.4f}. Champion: {out['champion'] or 'none'}")
    elif command == "run":
        ids = argv[1:]
        if ids == ["--proposed"]:
            ids = [i["experiment"] for i in _agenda() if i["status"] == "proposed"]
        if not ids:
            print("nothing to run: every proposed experiment already has a result")
            return
        run(ids)
    elif command == "events":
        from scaata.research import event_study

        study = event_study.run()
        print(f"{study['events']} events from {study['headlines_scored']} scored headlines, {study['first']} to "
              f"{study['last']}")
        for r in study["table"]:
            if r["span"] == "all":
                h = r["h20"]
                print(f"  {r['label']:<26} {r['tone']:<5} n={r['events']:<5} 20 sessions: {h['mean']:+.2%} "
                      f"[{h['low']:+.2%}, {h['high']:+.2%}]{' *' if h['reliable'] else ''}")
    elif command == "phase-a":
        from scaata.research import phase_a

        study = phase_a.run()
        print(f"{study['judged_days']} judged days since {study['first']}; {study['runtime_seconds']}s")
        for row in study["learners"]:
            h = row["all"]
            print(f"  {row['kind']:<9} Brier improvement over equal trust {h['improvement']:+.5f} "
                  f"[{h['low']:+.5f}, {h['high']:+.5f}]  corrected p {h['p_corrected']}")
        for kind, test in study["poison"].items():
            print(f"  poison ({kind}): " + ", ".join(f"{k} {v:.0%}" for k, v in test["rates"].items()))
    elif command == "contribution":
        from scaata.research import contribution

        out = contribution.run()
        for span, d in out["spans"].items():
            print(f"{span}: hold-by-default Sharpe {d['baseline']['sharpe']:.3f} -> all three {d['full']['sharpe']:.3f}")
            for a, v in d["shapley"]["sharpe"].items():
                print(f"   {a:<11} Shapley (Sharpe) {v:+.4f}   (return {d['shapley']['cagr'][a]:+.4f}, worst fall "
                      f"{d['shapley']['max_drawdown'][a]:+.4f})")
    elif command == "rebound":
        from scaata.research import rebound

        out = rebound.run()
        s = out["summary"]
        print(f"{s['with_a_rebound']} of {s['episodes']} episodes had a rebound to recapture; the rule captured more in "
              f"{s['captures_more']}; the worst fall was no more than 2 points worse in {s['falls_no_worse']}")
        for r in out["episodes"]:
            cl, cr = r["capture_live"], r["capture_rebound"]
            print(f"  {r['title']:<34} live {r['live_rule']['return']:+.1%} ({r['live_rule']['worst_fall']:.1%})  rebound "
                  f"{r['rebound']['return']:+.1%} ({r['rebound']['worst_fall']:.1%})  holding {r['holding']['return']:+.1%} "
                  f"({r['holding']['worst_fall']:.1%})  captured {('%.0f%%' % (cl*100)) if cl is not None else '-'} -> "
                  f"{('%.0f%%' % (cr*100)) if cr is not None else '-'}")
    elif command == "list":
        for r in registry.load():
            print(f"{registry.status(r):<14} {r['id']:<36} {r['title']}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
