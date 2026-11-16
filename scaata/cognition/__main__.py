"""Cognitive HANSEI's command line.

    python -m scaata.cognition            print what it currently believes
    python -m scaata.cognition snapshot   record today's beliefs in the history

The history (`cognition/history.jsonl`) is append-only: each snapshot adds one
line per belief that is new or whose status changed, so what the system
believed on an earlier date is never rewritten.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from scaata.cognition.beliefs import build
from scaata.config import ROOT_DIR

HISTORY_PATH = ROOT_DIR / "cognition" / "history.jsonl"


def load_history(path: Path | None = None) -> list[dict]:
    path = path or HISTORY_PATH
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def snapshot(state: dict | None = None, path: Path | None = None, at: str | None = None) -> list[dict]:
    """Appends every belief that is new or has changed status since the last snapshot. Returns what was added."""
    path = path or HISTORY_PATH
    state = state or build()
    last: dict[str, dict] = {}
    for event in load_history(path):
        last[event["key"]] = event
    at = at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    added = []
    for b in state["beliefs"]:
        before = last.get(b["key"])
        if before and before["status"] == b["status"]:
            continue
        added.append({"at": at, "key": b["key"], "kind": b["kind"], "status": b["status"],
                      "was": before["status"] if before else None, "statement": b["statement"],
                      "evidence": b["evidence"], "source": b["source"]})
    if added:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for event in added:
                handle.write(json.dumps(event, ensure_ascii=False) + chr(10))
    return added


def main(argv: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if argv and argv[0] == "snapshot":
        added = snapshot()
        print(f"{len(added)} belief(s) new or changed")
        for event in added:
            change = f"{event['was']} -> {event['status']}" if event["was"] else event["status"]
            print(f"  [{change}] {event['statement']}")
        return
    state = build()
    for b in state["beliefs"]:
        print(f"[{b['kind']:<12} {b['status']:<12}] {b['statement']}")
    print()
    for source, s in state["scorecard"].items():
        if s["opinions"]:
            print(f"{source:<11} right {s['hit_rate']:.0%} of {s['days']} days ({s['low']:.0%}-{s['high']:.0%}); "
                  f"weak views {s['weak']:.0%}, strong views {s['strong']:.0%}")


if __name__ == "__main__":
    main(sys.argv[1:])
