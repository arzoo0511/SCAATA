"""The experiment registry: an append-only log of what was asked, what was
promised before the run, what came out, and what the reviewer made of it.

One JSON event per line in `research/experiments.jsonl`:

    registered -> the question, hypothesis, method and pass/fail criteria
    result     -> what the run produced
    review     -> the reviewer's checks and verdict

Events are only ever appended, never edited, so the order on disk proves the
criteria were written before the result existed. `load()` folds the events
into one record per experiment. A failed or rejected experiment stays on
file: that is the point.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scaata.config import ROOT_DIR

REGISTRY_PATH = ROOT_DIR / "research" / "experiments.jsonl"
VERDICTS = ("supported", "not_supported", "inconclusive", "failed_review")
REQUIRED = ("id", "title", "question", "hypothesis", "kind", "method", "criteria", "limitations")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _events(path: Path | None = None) -> list[dict]:
    path = path or REGISTRY_PATH
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(event: dict, path: Path | None = None) -> None:
    path = path or REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def register(spec: dict, path: Path | None = None, at: str | None = None) -> str:
    """Files an experiment before it is run. The spec must state its criteria up front."""
    missing = [k for k in REQUIRED if not spec.get(k)]
    if missing:
        raise ValueError(f"experiment spec is missing: {', '.join(missing)}")
    if any(e["id"] == spec["id"] for e in _events(path)):
        raise ValueError(f"experiment '{spec['id']}' is already registered; criteria cannot be rewritten")
    _append({"event": "registered", "id": spec["id"], "at": at or _now(),
             "spec": {"origin": "lab", "params": {}, **spec}}, path)
    return spec["id"]


def record_result(experiment_id: str, results: dict, path: Path | None = None, at: str | None = None) -> None:
    record = get(experiment_id, path)
    if record is None:
        raise KeyError(f"experiment '{experiment_id}' is not registered; register it before running it")
    if record.get("results") is not None:
        raise ValueError(f"experiment '{experiment_id}' already has a result; register a new experiment to re-run")
    _append({"event": "result", "id": experiment_id, "at": at or _now(), "results": results}, path)


def record_review(experiment_id: str, review: dict, path: Path | None = None, at: str | None = None) -> None:
    if review.get("verdict") not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    if get(experiment_id, path) is None:
        raise KeyError(f"experiment '{experiment_id}' is not registered")
    _append({"event": "review", "id": experiment_id, "at": at or _now(), "review": review}, path)


def load(path: Path | None = None) -> list[dict]:
    """One record per experiment, in the order they were registered. The latest review wins."""
    records: dict[str, dict] = {}
    for event in _events(path):
        if event["event"] == "registered":
            records[event["id"]] = {**event["spec"], "registered_at": event["at"], "results": None,
                                    "ran_at": None, "review": None, "reviewed_at": None}
        elif event["id"] in records and event["event"] == "result":
            records[event["id"]].update(results=event["results"], ran_at=event["at"])
        elif event["id"] in records and event["event"] == "review":
            records[event["id"]].update(review=event["review"], reviewed_at=event["at"])
    return list(records.values())


def get(experiment_id: str, path: Path | None = None) -> dict | None:
    return next((r for r in load(path) if r["id"] == experiment_id), None)


def status(record: dict) -> str:
    """Where a record stands: its verdict if reviewed, else how far it got."""
    if record.get("review"):
        return record["review"]["verdict"]
    return "ran" if record.get("results") is not None else "registered"
