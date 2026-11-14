"""Macroverse's command line.

    python -m scaata.macroverse refresh             freeze a new factor snapshot
    python -m scaata.macroverse replay [ID ...]     replay historical episodes (all by default)
    python -m scaata.macroverse scenario [ID ...]   run synthetic scenarios (all by default)
    python -m scaata.macroverse all                 both, plus the sensitivity table

Results are written to `macroverse/runs/`. Prices come from the research
snapshot (`python -m scaata.research refresh`). Nothing here reads or
writes the live paper book, journal or memory.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

from scaata.config import ROOT_DIR
from scaata.macroverse import factors, replay, shock
from scaata.research import prices

RUNS = ROOT_DIR / "macroverse" / "runs"
PATHS = 30


def _save(name: str, payload: dict, snapshot: dict, factor_meta: dict | None) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    payload["provenance"] = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "prices_sha256": snapshot.get("sha256"), "prices_through": snapshot.get("last"),
                             "factors_sha256": (factor_meta or {}).get("sha256")}
    (RUNS / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def run_replays(ids: list[str]) -> None:
    opens, closes, snapshot = prices.load()
    aligned, factor_meta = factors.aligned(closes.index), factors.meta()
    for episode in replay.EPISODES:
        if ids and episode["id"] not in ids:
            continue
        started = time.time()
        out = replay.run_episode(episode, opens, closes, aligned)
        _save(episode["id"], out, snapshot, factor_meta)
        a, h = out["stats"]["hansei"], out["stats"]["holding"]
        print(f"{episode['id']:<22} HANSEI {a['return']:+.1%} (worst fall {a['max_drawdown']:.1%})   holding "
              f"{h['return']:+.1%} ({h['max_drawdown']:.1%})   {out['stats']['trades']} trades   "
              f"{time.time() - started:.0f}s", flush=True)


def run_scenarios(ids: list[str]) -> None:
    opens, closes, snapshot = prices.load()
    aligned, factor_meta = factors.aligned(closes.index), factors.meta()
    sens = shock.sensitivities(closes, aligned)
    _save("sensitivities", {"kind": "estimate", "table": sens["table"], "weeks": sens["weeks"],
                            "first": sens["first"], "last": sens["last"], "split": sens["split"],
                            "labels": factors.LABELS,
                            "unavailable": [{"title": t, "why": w} for t, w in shock.UNAVAILABLE]},
          snapshot, factor_meta)
    print(f"sensitivities: {sens['weeks']} weeks, {sum(r['reliable'] for r in sens['table'])} of "
          f"{len(sens['table'])} intervals exclude zero", flush=True)
    for scenario in shock.SCENARIOS:
        if ids and scenario["id"] not in ids:
            continue
        started = time.time()
        out = shock.run_scenario(scenario, opens, closes, aligned, paths=PATHS, sens=sens)
        _save(scenario["id"], out, snapshot, factor_meta)
        s = out["stats"]
        print(f"{scenario['id']:<16} median HANSEI {s['hansei']['median_return']:+.1%} vs holding "
              f"{s['holding']['median_return']:+.1%}; falls less in {s['share_hansei_falls_less']:.0%} of paths   "
              f"{time.time() - started:.0f}s", flush=True)


def main(argv: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    command, ids = (argv[0] if argv else ""), argv[1:]
    if command == "refresh":
        info = factors.refresh()
        print(f"factor snapshot: {info['rows']} rows through {info['last']}, sha256 {info['sha256'][:12]}")
    elif command == "replay":
        run_replays(ids)
    elif command == "scenario":
        run_scenarios(ids)
    elif command == "all":
        run_replays([])
        run_scenarios([])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
