"""From a finding to a change: the only road from the Lab to the live agent.

A proposal is a suggested change to how HANSEI trades, with the evidence
that suggests it. This module works out, from the registry alone, where each
proposal stands:

    needs a test   something suggests it, but no experiment has tested the change itself
    not supported  the experiment that tested it did not hold up; nothing to do
    inconclusive   tested, but the result is within noise or holds in one span only
    ready          tested and supported: waiting for a person to approve

Nothing here applies a change. `ready` means "a person may now decide"; the
change itself is still made by hand, as a reviewed edit to the agent. There
is deliberately no function in this module, or anywhere else, that writes to
the agent's settings. `applied` only reports what the agent's code says.
"""
from __future__ import annotations

STATUS = {"supported": "ready", "not_supported": "not supported", "inconclusive": "inconclusive",
          "failed_review": "needs a test"}

# Each proposal names the change, the experiment that tests exactly that change, and why it was raised.
CATALOGUE = [
    {"id": "fixed-share", "change": "Replace plain Hedge with Fixed-Share",
     "why": "Which advisor is best changes with the market; plain Hedge is built to find one best advisor for "
            "the whole record, Fixed-Share to follow a best advisor that changes.",
     "tested_by": "fixed-share-corrected", "wants": "supported",
     "reading": "Decided by the owner on 16 Nov 2026 on the strength of the theory, and switched on that day "
                "whatever the experiment showed. The experiment is the record of what the replay made of it."},
    {"id": "drop-trend", "change": "Remove the trend advisor",
     "why": "Live it has been right on fewer than half its days, and silencing it in the replay did not "
            "make HANSEI measurably worse.",
     "tested_by": "without-trend", "wants": "not_supported",
     "reading": "The test asked whether HANSEI is worse without trend. 'Not supported' there means trend is "
                "not shown to earn its place; it does not show HANSEI is better without it."},
    {"id": "freeze-trust", "change": "Stop shifting trust between advisors (fixed equal trust)",
     "why": "Learning is the system's namesake, but switching it off changed nothing measurable.",
     "tested_by": "learning-switched-off", "wants": "not_supported",
     "reading": "Not shown to help is not the same as shown to hurt. Fixed trust would be simpler, not better."},
    {"id": "shorter-wait", "change": "Wait 5 sessions between trades instead of 10",
     "why": "HANSEI trails holding while the stocks are below their six-month-ago level: it sits out rebounds.",
     "tested_by": "wait-five-sessions", "wants": "supported", "reading": ""},
    {"id": "fractional-book", "change": "Let the paper book hold fractional shares, or make it larger",
     "why": "Live it has wanted far more invested than whole shares of a ₹10,000 book allow.",
     "tested_by": "live-like-book", "wants": "supported",
     "reading": "If a whole-share book does worse than the rupee replay, the live book is handicapped by its "
                "size, not by its rules. Changing it restarts the live comparison."},
    {"id": "replay-no-borrowing", "change": "Stop the replay engine from borrowing",
     "why": "The replay was up to 109% invested in 2008; the live book cannot borrow, so they disagree.",
     "tested_by": "no-borrowing", "wants": "any",
     "reading": "This is a correction to the measuring tool, not to the agent. Whatever the result, the replay "
                "should match the live book; applying it changes every recorded backtest number."},
]


def build(records: list[dict]) -> list[dict]:
    """Every proposal with its current standing, worked out from the experiments on file."""
    by_id = {r["id"]: r for r in records}
    out = []
    for item in CATALOGUE:
        record = by_id.get(item["tested_by"])
        review = (record or {}).get("review")
        verdict = review["verdict"] if review else None
        if verdict is None:
            status = "needs a test"
        elif item["wants"] == "any":
            status = "ready" if verdict != "failed_review" else "needs a test"
        elif item["wants"] == "supported":
            status = STATUS[verdict]
        else:
            # the change is the opposite of what the experiment hypothesised
            status = {"not_supported": "inconclusive", "supported": "not supported",
                      "inconclusive": "inconclusive", "failed_review": "needs a test"}[verdict]
        primary = ((record or {}).get("results") or {}).get("primary", {})
        out.append({**item, "status": status, "verdict": verdict, "experiment_title": (record or {}).get("title"),
                    "effect": primary.get("values"), "effect_label": primary.get("label"),
                    "summary": (review or {}).get("summary"), "applied": _applied(item["id"])})
    return out


def _applied(proposal_id: str) -> bool:
    """Whether the change is live, read from the agent's own code. Reported, never set, from here."""
    if proposal_id == "fixed-share":
        from scaata.agent import memory

        return memory.SHARE > 0
    return False
