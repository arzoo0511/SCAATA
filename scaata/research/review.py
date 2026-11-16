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

from scaata.research import stats as rstats
from scaata.research.registry import LEGACY_REPLAY

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
INTEGRITY = ("criteria_written_first", "data_snapshot_recorded", "no_look_ahead", "costs_charged",
             "change_took_effect", "baseline_reproduces")
LEGACY_LEARNER = {"fixed_share": 0.0, "learning_rate": 0.05}     # records filed before the learner was recorded
MIN_SESSIONS = 750             # about three years in each span
MIN_TRADES = 30


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


def p_value(record: dict) -> float | None:
    """One-sided p-value of an experiment's holdout effect, in the direction it points: one minus the share
    of block-bootstrap resamples that agree with it. None for experiments with no bootstrap."""
    primary = (record.get("results") or {}).get("primary") or {}
    prob = (primary.get("prob_better") or {}).get("holdout")
    if prob is None:
        return None
    value = (primary.get("values") or {}).get("holdout", 0.0)
    return 1 - (prob if value >= 0 else 1 - prob)


def budget_ledger(records: list[dict]) -> list[dict]:
    """The evidence budget replayed over lab experiments in the order they were filed. Experiments that
    failed review spent nothing: their numbers were never trusted."""
    ordered = sorted((r for r in records if r.get("origin") == "lab" and r.get("results") is not None
                      and (r.get("review") or {}).get("verdict") != "failed_review"),
                     key=lambda r: r.get("registered_at") or "")
    tests, asked = [], False
    for r in ordered:
        p = p_value(r)
        if r.get("kind") == "baseline":       # the standing question is re-asked on new data: only the first
            p, asked = (None if asked else p), True      # asking is a new hypothesis that spends budget
        tests.append((r["id"], p))
    return rstats.ledger(tests)


def budget_before(record: dict, earlier: list[dict]) -> tuple[float, float]:
    """(wealth left, level this experiment is offered) given the lab experiments filed before it."""
    mine = record.get("registered_at") or ""
    prior = [r for r in earlier if r["id"] != record["id"] and (r.get("registered_at") or "") <= mine]
    rows = budget_ledger(prior)
    wealth = rows[-1]["wealth_after"] if rows else rstats.INITIAL_WEALTH
    return wealth, rstats.offered_level(wealth)


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

    # A variant whose every number equals the baseline's did not change anything: the experiment is
    # mis-wired, whatever it claims to test. (This caught a real bug on 16 Nov 2026.)
    versus = [v for k, v in results.get("metrics", {}).items() if k.endswith("_vs_baseline")]
    if versus:
        took = any(abs(v) > 1e-12 for v in versus)
        checks.append(_check("change_took_effect", took, "the variant's results differ from the live rule's" if took
                             else "the variant is identical to the live rule to the last digit: the change was "
                                  "never applied"))

    # The live rule replayed on the same data with the same learner must give the same answer every time.
    learner = (results.get("engine") or {}).get("live_learner") or LEGACY_LEARNER
    replay_mode = (results.get("engine") or {}).get("replay") or LEGACY_REPLAY
    same = [r for r in (earlier or []) if r["id"] != record["id"] and r.get("origin") == "lab"
            and r.get("kind") == "baseline" and r.get("results")
            and (r["results"].get("data") or {}).get("sha256") == sha and sha
            and ((r["results"].get("engine") or {}).get("live_learner") or LEGACY_LEARNER) == learner
            and ((r["results"].get("engine") or {}).get("replay") or LEGACY_REPLAY) == replay_mode
            and (r.get("registered_at") or "") <= (record.get("registered_at") or "")]
    if same and periods:
        reference = same[-1]["results"]["periods"]
        drift = [p for p, r in periods.items() if p in reference
                 and abs(r["baseline"]["final"] / reference[p]["variant"]["final"] - 1) > 1e-6]
        checks.append(_check("baseline_reproduces", not drift,
                             f"the live rule gives the same result as in '{same[-1]['id']}' on the same data"
                             if not drift else f"the live rule no longer matches '{same[-1]['id']}' on the same "
                                               f"data ({', '.join(drift)}): the engine has changed underneath"))

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
    wealth, offered = budget_before(record, earlier or [])
    needed = 1 - offered
    prob = primary.get("prob_better", {}).get("holdout")
    if prob is None:
        checks.append(_check("beats_noise", None, "no bootstrap for this kind of experiment; read it as descriptive"))
    else:
        # the effect may be claimed in either direction; noise is judged on the side it points
        confidence = prob if values.get("holdout", 0) >= 0 else 1 - prob
        plural = "" if looks == 1 else "s"
        checks.append(_check("beats_noise", confidence >= needed,
                             f"{confidence:.0%} of block-bootstrap resamples agree with the holdout effect; "
                             f"{needed:.1%} needed: the evidence budget has {wealth:.3f} left and offers this "
                             f"test a false-alarm level of {offered:.3f} (this holdout has now been looked at "
                             f"{looks} time{plural})"))

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
    return {"verdict": verdict, "summary": summary, "checks": checks, "criteria": criteria, "holdout_looks": looks,
            "budget": {"wealth": wealth, "level": offered}}
