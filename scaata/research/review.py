"""The devil's advocate: a reviewer made of code, not opinion.

It reads a finished experiment and tries to knock it down. Integrity checks
come first -- an experiment that peeked at the future, skipped costs, or
cannot say what data it ran on fails review outright, whatever its numbers.
Then the criteria that were written before the run are applied as written.
Then robustness: does the effect point the same way in the years the rule
was tuned on and the years it was not, is it larger than block-bootstrap
noise, and how many times has this holdout already been looked at.

Verdicts:

    failed_review   an integrity check failed; the numbers cannot be trusted
    not_supported   a pre-registered criterion was not met
    inconclusive    the criteria were met but a robustness check was not
    supported       criteria met and every check passed
"""
from __future__ import annotations

import operator

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
INTEGRITY = ("criteria_written_first", "data_snapshot_recorded", "no_look_ahead", "costs_charged")
MIN_SESSIONS = 750             # about three years in each span
MIN_TRADES = 30
NOISE_LEVEL = 0.10             # chance of a fluke we accept, before sharing it across looks


def _check(name: str, passed: bool | None, detail: str) -> dict:
    return {"name": name, "passed": passed, "detail": detail}


def apply_criteria(record: dict) -> list[dict]:
    metrics = (record.get("results") or {}).get("metrics", {})
    out = []
    for criterion in record.get("criteria", []):
        value = metrics.get(criterion["metric"])
        passed = None if value is None else bool(OPS[criterion["op"]](value, criterion["value"]))
        out.append({**criterion, "observed": value, "passed": passed})
    return out


def holdout_looks(record: dict, earlier: list[dict]) -> int:
    """How many lab experiments, this one included, have now been judged on the holdout years."""
    mine = record.get("registered_at") or ""
    return 1 + sum(1 for r in earlier if r.get("origin") == "lab" and r["id"] != record["id"]
                   and r.get("results") is not None and (r.get("registered_at") or "") <= mine)


def review(record: dict, earlier: list[dict] | None = None) -> dict:
    """Checks one experiment. `earlier` is the rest of the registry, for the multiple-testing count."""
    results = record.get("results") or {}
    periods, integrity = results.get("periods", {}), results.get("integrity", {})
    checks = []

    first = bool(record.get("registered_at") and record.get("ran_at") and record["registered_at"] <= record["ran_at"]
                 and record.get("origin") == "lab")
    checks.append(_check("criteria_written_first", first,
                         f"registered {record.get('registered_at')}, ran {record.get('ran_at')}" if first else
                         "the criteria were not on file before the result existed"))

    snapshot = results.get("data") or {}
    sha = snapshot.get("sha256")
    checks.append(_check("data_snapshot_recorded", bool(sha),
                         f"price snapshot {sha[:12]}, through {snapshot.get('last')}" if sha else
                         "the result does not say which data it ran on"))

    if integrity.get("checked"):
        same = bool(integrity.get("invariant"))
        leaked = integrity.get("days_with_leaked_views") or []
        if same:
            detail = (f"views recomputed without the future on {integrity.get('days_sampled')} days matched, and "
                      f"replaying {integrity.get('sessions')} sessions with later prices deleted gave the "
                      "identical equity curve")
        elif leaked:
            detail = (f"on {len(leaked)} of {integrity.get('days_sampled')} sampled days an advisor's view changed "
                      f"when later prices were deleted (first: {leaked[0]}): it used the future")
        else:
            detail = "replaying a stretch with later prices deleted gave a DIFFERENT equity curve"
    else:
        same, detail = False, "the look-ahead test was not run"
    checks.append(_check("no_look_ahead", same, detail))

    uncharged = [p for p, r in periods.items() if r.get("trades", 0) > 0 and not r.get("costs", 0) > 0]
    costs_ok = bool(periods) and not uncharged
    checks.append(_check("costs_charged", costs_ok, "every span with trades paid NSE costs" if costs_ok else
                         "trades without costs in: " + (", ".join(uncharged) or "no spans reported")))

    short = [f"{p} ({r.get('sessions', 0)} sessions)" for p, r in periods.items() if r.get("sessions", 0) < MIN_SESSIONS]
    thin = [f"{p} ({r.get('trades', 0)} trades)" for p, r in periods.items() if r.get("trades", 0) < MIN_TRADES]
    checks.append(_check("enough_data", not short and not thin,
                         f"at least {MIN_SESSIONS} sessions and {MIN_TRADES} trades in each span"
                         if not short and not thin else "too little to judge: " + ", ".join(short + thin)))

    primary = results.get("primary", {})
    values = primary.get("values", {})
    agree = len(values) >= 2 and (all(v > 0 for v in values.values()) or all(v < 0 for v in values.values()))
    checks.append(_check("same_direction_in_both_spans", agree, f"{primary.get('label', 'effect')}: "
                         + ", ".join(f"{p} {v:+.3f}" for p, v in values.items())))

    looks = holdout_looks(record, earlier or [])
    needed = 1 - NOISE_LEVEL / looks
    prob = primary.get("prob_better", {}).get("holdout")
    if prob is None:
        checks.append(_check("beats_noise", None, "no bootstrap for this kind of experiment; read it as descriptive"))
    else:
        # the effect may be claimed in either direction; noise is judged on the side it points
        confidence = prob if values.get("holdout", 0) >= 0 else 1 - prob
        plural = "" if looks == 1 else "s"
        checks.append(_check("beats_noise", confidence >= needed,
                             f"{confidence:.0%} of block-bootstrap resamples agree with the holdout effect; "
                             f"{needed:.1%} needed (this holdout has now been looked at {looks} time{plural})"))

    criteria = apply_criteria(record)
    integrity_ok = all(c["passed"] for c in checks if c["name"] in INTEGRITY)
    criteria_ok = bool(criteria) and all(c["passed"] for c in criteria)
    robust_ok = all(c["passed"] is not False for c in checks if c["name"] not in INTEGRITY)
    if not integrity_ok:
        verdict = "failed_review"
    elif not criteria_ok:
        verdict = "not_supported"
    else:
        verdict = "supported" if robust_ok else "inconclusive"
    failed = [c["name"].replace("_", " ") for c in checks if c["passed"] is False]
    missed = [c.get("label") or c["metric"] for c in criteria if not c["passed"]]
    summary = {"failed_review": "Integrity check failed: " + ", ".join(failed) + ". The numbers are not trusted.",
               "not_supported": "Did not meet what it promised before the run: " + "; ".join(missed) + ".",
               "inconclusive": "Met its own criteria but not the reviewer's: " + ", ".join(failed) + ".",
               "supported": "Met every criterion written before the run and passed every check."}[verdict]
    return {"verdict": verdict, "summary": summary, "checks": checks, "criteria": criteria, "holdout_looks": looks}
