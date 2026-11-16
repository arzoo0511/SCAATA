"""Builds the belief store and scores HANSEI's own predictions.

A belief is a plain record:

    kind       fact | relationship | finding | hypothesis | assumption
    status     established | leaning | cannot tell | contradicted | untested
    statement  one sentence
    evidence   the numbers, and where they came from
    strength   0..1, how far the evidence goes (for drawing only)

The status is computed from the evidence by fixed rules, so a belief cannot
be promoted by wording:

- a measured relationship is `established` only if its 95% interval excludes zero;
- a Lab finding takes the reviewer's verdict;
- a live hit rate is judged with a Wilson interval over *days*, not opinions,
  because five stocks on one day are not five independent observations, and
  never on fewer than 20 judged days;
- a design assumption is `contradicted` only when a measured number says so,
  `leaning` when it has been tested and held, and `untested` otherwise.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scaata.config import ROOT_DIR

MEMORY_PATH = ROOT_DIR / "forward_test" / "hansei_memory.json"
JOURNAL_PATH = ROOT_DIR / "forward_test" / "hansei_journal.json"
EVALUATION_PATH = ROOT_DIR / "forward_test" / "hansei_evaluation.json"
BOOK_PATH = ROOT_DIR / "paper_book_india.json"
REGISTRY_PATH = ROOT_DIR / "research" / "experiments.jsonl"
AGENDA_PATH = ROOT_DIR / "research" / "agenda.json"
SENSITIVITIES_PATH = ROOT_DIR / "macroverse" / "runs" / "sensitivities.json"
MACRO_RUNS = ROOT_DIR / "macroverse" / "runs"
PHASE_A_PATH = ROOT_DIR / "research" / "phase_a.json"
CONTRIBUTION_PATH = ROOT_DIR / "research" / "contribution.json"

ADVISORS = ("trend", "volatility", "momentum", "news")
ADVISOR_NAMES = {"trend": "Trend", "volatility": "The volatility brake", "momentum": "Momentum", "news": "News"}
FACTOR_NAMES = {"market": "the market", "crude": "crude oil", "rupee": "the dollar against the rupee"}
SILENT = 0.05                  # a view or score smaller than this is no opinion
MIN_DAYS = 20                  # judged days needed before a hit rate can settle anything
STATUSES = ("established", "leaning", "cannot tell", "contradicted", "untested")
KINDS = ("fact", "relationship", "finding", "hypothesis", "assumption")


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _registry(path: Path) -> list[dict]:
    """The Lab's records, folded from its event log. Kept local so this module needs nothing but JSON."""
    records: dict[str, dict] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        if event["event"] == "registered":
            records[event["id"]] = {**event["spec"], "registered_at": event["at"], "results": None, "review": None}
        elif event["id"] in records and event["event"] == "result":
            records[event["id"]]["results"] = event["results"]
        elif event["id"] in records and event["event"] == "review":
            records[event["id"]]["review"] = event["review"]
    return list(records.values())


def wilson(hits: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a share, well-behaved at small n."""
    if n <= 0:
        return 0.0, 1.0
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


# ------------------------------------------------------------ predictions
def predictions(memory: dict | None, journal: list[dict] | None, keyword_days: set[str] | None = None) -> list[dict]:
    """Every judged prediction on file. Each advisor's view, and the brain's weighted score, is a
    prediction of a stock's next 20 sessions against a liquid fund; this pairs each with what happened.

    On days the news was scored by the offline keyword fallback rather than the language model, the news
    advisor's view is left out of its own scorecard: it was a different, cruder instrument that day."""
    keyword_days = keyword_days or set()
    scores = {(e["session"], d["symbol"]): d["score"] for e in journal or [] for d in e.get("decisions", [])}
    rows = []
    for obs in (memory or {}).get("observations", []):
        if not obs.get("judged") or obs.get("forward_return") is None:
            continue
        outcome = obs["forward_return"]
        sources = {a: obs.get("views", {}).get(a, 0.0) for a in ADVISORS}
        if (obs["date"], obs["symbol"]) in scores:
            sources["brain"] = scores[(obs["date"], obs["symbol"])]
        for source, view in sources.items():
            if abs(view) < SILENT or outcome == 0 or (source == "news" and obs["date"] in keyword_days):
                continue
            rows.append({"date": obs["date"], "symbol": obs["symbol"], "source": source, "view": view,
                         "outcome": outcome, "right": (view > 0) == (outcome > 0)})
    return rows


def scorecard(rows: list[dict]) -> dict[str, dict]:
    """Per source: how often it pointed the right way, judged over days, and whether stronger views
    were right more often than weaker ones (which is what confidence should mean)."""
    out = {}
    for source in (*ADVISORS, "brain"):
        mine = [r for r in rows if r["source"] == source]
        if not mine:
            out[source] = {"opinions": 0, "days": 0, "hit_rate": None, "low": None, "high": None,
                           "status": "untested", "strong": None, "weak": None}
            continue
        by_day: dict[str, list[bool]] = {}
        for r in mine:
            by_day.setdefault(r["date"], []).append(r["right"])
        daily = [sum(v) / len(v) for v in by_day.values()]
        rate = sum(daily) / len(daily)
        low, high = wilson(rate * len(daily), len(daily))
        status = "established" if low > 0.5 else "contradicted" if high < 0.5 else "cannot tell"
        if len(daily) < MIN_DAYS:      # a perfect week is still a week
            status = "cannot tell"
        ranked = sorted(mine, key=lambda r: abs(r["view"]))
        half = len(ranked) // 2
        share = lambda part: (sum(r["right"] for r in part) / len(part)) if part else None   # noqa: E731
        out[source] = {"opinions": len(mine), "days": len(daily), "hit_rate": rate, "low": low, "high": high,
                       "status": status, "weak": share(ranked[:half]), "strong": share(ranked[half:]),
                       "weak_n": half, "strong_n": len(ranked) - half}
    return out


def calibration(rows: list[dict], source: str = "brain", buckets: int = 3) -> list[dict]:
    """Hit rate by how strong the view was: equal-sized groups from weakest to strongest."""
    mine = sorted((r for r in rows if r["source"] == source), key=lambda r: abs(r["view"]))
    if len(mine) < buckets * 4:
        return []
    size = len(mine) / buckets
    out = []
    for k in range(buckets):
        part = mine[round(k * size):round((k + 1) * size)]
        out.append({"bucket": ("weakest", "middle", "strongest")[k] if buckets == 3 else str(k + 1),
                    "n": len(part), "hit_rate": sum(r["right"] for r in part) / len(part),
                    "mean_strength": sum(abs(r["view"]) for r in part) / len(part)})
    return out


# ---------------------------------------------------------------- beliefs
def _belief(kind: str, status: str, statement: str, evidence: str, source: str, strength: float,
            topic: str, detail: dict | None = None) -> dict:
    return {"kind": kind, "status": status, "statement": statement, "evidence": evidence, "source": source,
            "strength": round(max(0.0, min(1.0, strength)), 3), "topic": topic, "detail": detail or {}}


def _relationships(sens: dict | None) -> list[dict]:
    out = []
    for r in (sens or {}).get("table", []):
        ten = r["beta"] * 0.10
        width = r["high"] - r["low"]
        if r["reliable"]:
            statement = (f"{r['name']} moves about {abs(ten):.1%} {'with' if r['beta'] > 0 else 'against'} a 10% "
                         f"move in {FACTOR_NAMES[r['factor']]}.")
            status = "established"
        else:
            statement = f"{r['name']}'s link to {FACTOR_NAMES[r['factor']]} cannot be told apart from zero."
            status = "cannot tell"
        stable = ""
        if r.get("before") is not None:
            stable = f" Before {sens.get('split', '2017')[:4]}: {r['before']:+.2f}; after: {r['after']:+.2f}."
            if r.get("held_out_ok") is not None:
                stable += (" The later years fall inside what the earlier years alone predicted."
                           if r["held_out_ok"] else
                           " The later years fall OUTSIDE what the earlier years alone predicted, so it has shifted.")
        if r["reliable"] and r.get("held_out_ok") is False:
            status = "leaning"       # measurable, but not stable across the two halves of history
        out.append(_belief(
            "relationship", status, statement,
            f"Sensitivity {r['beta']:+.2f}, 95% interval {r['low']:+.2f} to {r['high']:+.2f}, from "
            f"{sens.get('weeks')} weekly returns ({sens.get('first')} to {sens.get('last')}).{stable}",
            "Macroverse sensitivities", 1 - min(1.0, width / max(abs(r["beta"]) * 2, 0.5)) if r["reliable"] else 0.2,
            r["factor"], {"symbol": r["symbol"], "factor": r["factor"], "beta": r["beta"], "low": r["low"],
                          "high": r["high"]}))
    return out


VERDICT_STATUS = {"supported": "established", "inconclusive": "leaning", "not_supported": "contradicted",
                  "failed_review": "untested"}


def _findings(records: list[dict]) -> list[dict]:
    out = []
    answered = {r["hypothesis"] for r in records if (r.get("review") or {}).get("verdict") not in (None, "failed_review")}
    for r in records:
        review = r.get("review")
        if not review:
            continue
        if review["verdict"] == "failed_review" and r["hypothesis"] in answered:
            continue        # an invalid run that a corrected one replaced is a record, not a second belief
        legacy = r.get("origin") == "legacy"
        status = VERDICT_STATUS.get(review["verdict"], "untested")
        kind = "finding" if review["verdict"] in ("supported", "not_supported") else "hypothesis"
        met = [c for c in review.get("criteria", []) if c.get("passed")]
        evidence = review.get("summary", "")
        if review.get("criteria"):
            evidence += f" Met {len(met)} of {len(review['criteria'])} criteria written before the run."
        out.append(_belief(
            kind, status, r["hypothesis"], evidence,
            ("Before the Lab: " if legacy else "Lab experiment: ") + r["title"],
            {"established": 0.9, "leaning": 0.55, "contradicted": 0.85, "untested": 0.1}[status] * (0.7 if legacy else 1),
            "research", {"experiment": r["id"], "verdict": review["verdict"], "legacy": legacy}))
    return out


def _advisors(card: dict, trust: dict) -> list[dict]:
    out = []
    for advisor in ADVISORS:
        s = card.get(advisor, {})
        weight = trust.get(advisor)
        if not s.get("opinions"):
            out.append(_belief("hypothesis", "untested", f"{ADVISOR_NAMES[advisor]} points the right way more often "
                               "than not.", "No judged opinions yet.", "Live prediction record", 0.05, "advisors",
                               {"advisor": advisor, "trust": weight}))
            continue
        words = {"established": "has pointed the right way more often than chance",
                 "contradicted": "has pointed the wrong way more often than chance",
                 "cannot tell": "has not yet shown whether it beats a coin flip"}[s["status"]]
        out.append(_belief(
            "hypothesis" if s["status"] == "cannot tell" else "finding", s["status"],
            f"{ADVISOR_NAMES[advisor]} {words}.",
            f"Right on {s['hit_rate']:.0%} of days (95% interval {s['low']:.0%} to {s['high']:.0%}) over {s['days']} "
            f"judged days and {s['opinions']} opinions, 20 sessions ahead. Trusted {weight:.0%} of the vote "
            "(starts at 25%)." if weight is not None else "",
            "Live prediction record", abs(s["hit_rate"] - 0.5) * 2 if s["status"] != "cannot tell" else 0.25, "advisors",
            {"advisor": advisor, "trust": weight, **{k: s[k] for k in ("hit_rate", "low", "high", "days", "opinions")}}))
    return out


def _phase_a(study: dict | None, contribution: dict | None) -> list[dict]:
    """Beliefs about its own self-critique, from the 19-year replay (research/phase_a.json) and the
    contribution ledger (research/contribution.json). Statuses follow the same rules as everywhere else."""
    out = []
    if study:
        for row in study.get("skill", []):
            if row["span"] != "all" or row["source"] == "equal blend" or row.get("skill") is None:
                continue
            low, high = row["hit_low"], row["hit_high"]
            status = "established" if low > 0.5 else "contradicted" if high < 0.5 else "cannot tell"
            out.append(_belief(
                "finding", status,
                f"The {row['source']} advisor forecasts a stock's direction better than a coin flip.",
                f"Right on {row['hit_rate']:.1%} of {row['opinions']:,} opinions over 19 years (95% range {low:.1%} to "
                f"{high:.1%}). As a probability its views are over-confident: Brier skill {row['skill']:+.3f} "
                "(negative is worse than always saying 50%).", "Phase A replay", abs(row["hit_rate"] - 0.5) * 4,
                "advisors", {"advisor": row["source"], "key": f"skill-{row['source']}"}))
        for row in study.get("learners", []):
            if row["kind"] not in ("live", "adahedge"):
                continue
            for metric, words in (("payoff", "picks better directions"), ("brier", "gives a better-scored forecast")):
                cell = row[metric]["all"]
                if cell["improvement"] is None:
                    continue
                status = ("established" if cell["p_corrected"] is not None and cell["p_corrected"] < 0.05
                          and cell["improvement"] > 0 else "contradicted" if cell["high"] < 0 else "cannot tell")
                out.append(_belief(
                    "finding", status, f"{row['label']} {words} than equal trust.",
                    f"Over 19 years the difference was {cell['improvement']:+.5f} (95% range {cell['low']:+.5f} to "
                    f"{cell['high']:+.5f}); corrected p-value {cell['p_corrected']}.", "Phase A replay", 0.8
                    if status != "cannot tell" else 0.2, "learning",
                    {"key": f"learner-{row['kind']}-{metric}"}))
        for key, test in (study.get("poison") or {}).items():
            if key != "live":
                continue
            for rule, rate in test["rates"].items():
                status = "established" if test["passes"][rule] else "contradicted"
                out.append(_belief(
                    "finding", status, f"The live learner passes the poison test: {rule.replace('_', ' ')}.",
                    f"{rate:.0%} of {test['seeds']} draws (needs {test['required']:.0%}).", "Phase A poison test",
                    rate, "learning", {"key": f"poison-{rule}"}))
    if contribution:
        for advisor in contribution.get("players", []):
            values = {span: d["shapley"]["max_drawdown"][advisor] for span, d in contribution["spans"].items()}
            both = all(v > 0 for v in values.values())
            status = "established" if both and len(values) >= 2 else "leaning" if any(v > 0 for v in values.values()) \
                else "contradicted"
            out.append(_belief(
                "finding", status, f"The {advisor} advisor makes the worst fall smaller in the whole system.",
                "Shapley value of its effect on the worst fall: " + ", ".join(f"{span} {v:+.1%}" for span, v in values.items())
                + ". Replays the real brain with every subset of advisors allowed to speak.", "Contribution ledger",
                min(1.0, abs(sum(values.values())) * 5), "advisors", {"advisor": advisor, "key": f"contribution-{advisor}"}))
    return out


def _facts(book: dict | None, evaluation: dict | None, replays: list[dict]) -> list[dict]:
    out = []
    perf = (evaluation or {}).get("performance")
    if perf:
        ahead = perf["gap"] >= 0
        out.append(_belief(
            "fact", "established",
            f"Live, HANSEI is {'ahead of' if ahead else 'behind'} holding by {abs(perf['gap']):.1%} after "
            f"{perf['sessions']} sessions.",
            f"HANSEI {perf['hansei_return']:+.2%}, holding {perf['hold_return']:+.2%}. {perf['gap_from_exposure']:+.2%} "
            f"of the gap is from being on average {perf['average_invested']:.0%} invested; "
            f"{perf['gap_from_selection']:+.2%} from which stocks it held.", "Paper book and daily report card", 1.0,
            "live", {"gap": perf["gap"]}))
        out.append(_belief(
            "fact", "established",
            f"Live, its worst fall so far is {perf['hansei_max_drawdown']:.1%} against holding's "
            f"{perf['hold_max_drawdown']:.1%}.",
            f"Measured over {perf['sessions']} sessions; far too short to generalise.", "Daily report card", 1.0, "live"))
    if replays:
        less = sum(r["stats"]["hansei"]["max_drawdown"] > r["stats"]["holding"]["max_drawdown"] + 0.005 for r in replays)
        ahead = sum(r["stats"]["hansei"]["return"] > r["stats"]["holding"]["return"] for r in replays)
        out.append(_belief(
            "fact", "established",
            f"In {len(replays)} replayed crises its worst fall was clearly smaller than holding's in {less}, and it "
            f"finished ahead of holding in {ahead}.",
            "; ".join(f"{r['title']}: {r['stats']['hansei']['max_drawdown']:.0%} vs "
                      f"{r['stats']['holding']['max_drawdown']:.0%}" for r in replays) + ". Replays use real prices "
            "and the real brain, with the news advisor silent.", "Macroverse replays", 1.0, "history",
            {"falls_less": less, "ends_ahead": ahead, "episodes": len(replays)}))
    return out


def _assumptions(evaluation: dict | None, journal: list[dict], replays: list[dict], backtest: dict | None) -> list[dict]:
    """The brain's design choices, each checked against whatever number can check it."""
    out = []

    def add(key: str, statement: str, status: str, evidence: str) -> None:
        out.append(_belief("assumption", status, statement, evidence, "Design of the brain",
                           {"contradicted": 0.8, "leaning": 0.5, "untested": 0.1, "established": 0.9}[status],
                           "design", {"key": key}))

    peak = max((r["stats"].get("peak_invested", 0.0) for r in replays), default=None)
    if peak is None:
        add("no-leverage", "It holds by default and never borrows.", "untested", "No replay has been run to check.")
    elif peak > 1.005:
        add("no-leverage", "It holds by default and never borrows.", "contradicted",
            f"In replays the engine was up to {peak:.0%} invested: cash dips below zero when holdings outgrow "
            "their slots. The live paper book cannot do this, so replays are slightly more aggressive than live.")
    else:
        add("no-leverage", "It holds by default and never borrows.", "leaning",
            f"The most invested any replay became was {peak:.0%}.")

    grid = (backtest or {}).get("grid") or []
    if grid:
        positive = sum(g["sharpe_diff"] > 0 for g in grid)
        add("act-threshold", "A trade is worth making only when wanted and held exposure differ by a third of a slot.",
            "leaning" if positive == len(grid) else "untested",
            f"The Sharpe edge over holding was positive in {positive} of {len(grid)} nearby settings on 2020 to "
            "2026. That shows the result is not one lucky setting, not that this setting is best.")
    else:
        add("act-threshold", "A trade is worth making only when wanted and held exposure differ by a third of a slot.",
            "untested", "No settings grid on file.")

    add("horizon", "A view should be judged 20 sessions later, against a liquid fund.", "untested",
        "A design choice. No experiment has compared it with other horizons.")
    add("learning-rate", "Trust should move slowly (learning rate 0.05).", "untested",
        "Chosen because faster rates traded more in backtests. Whether learning helps at all is a separate, open "
        "question in the Lab.")

    held, wanted = [], []
    for entry in journal or []:
        for d in entry.get("decisions", []):
            held.append(d["current_exposure"]); wanted.append(d["target_exposure"])
    unfilled = len(((evaluation or {}).get("health") or {}).get("orders_that_filled_nothing") or [])
    if held:
        gap = sum(wanted) / len(wanted) - sum(held) / len(held)
        add("whole-shares", "₹10,000 in five equal slots, bought in whole shares, represents the strategy fairly.",
            "contradicted" if gap > 0.10 else "leaning",
            f"Live it has wanted {sum(wanted) / len(wanted):.0%} invested on average and held "
            f"{sum(held) / len(held):.0%}; {unfilled} queued orders filled nothing. One share of the dearest stock "
            "is most of a slot. The replays trade rupee amounts and do not have this drag.")
    return out


def build(paths: dict | None = None) -> dict:
    """The whole cognitive picture, from the files on disk. `paths` overrides any source file (for tests)."""
    p = {"memory": MEMORY_PATH, "journal": JOURNAL_PATH, "evaluation": EVALUATION_PATH, "book": BOOK_PATH,
         "registry": REGISTRY_PATH, "agenda": AGENDA_PATH, "sensitivities": SENSITIVITIES_PATH,
         "macro_runs": MACRO_RUNS, "phase_a": PHASE_A_PATH, "contribution": CONTRIBUTION_PATH, "backtest": ROOT_DIR / "results" / "hansei_backtest.json", **(paths or {})}
    memory, journal = _read(p["memory"]) or {}, _read(p["journal"]) or []
    evaluation, book, sens = _read(p["evaluation"]), _read(p["book"]), _read(p["sensitivities"])
    records = _registry(p["registry"])
    replays = []
    if Path(p["macro_runs"]).exists():
        replays = sorted((r for r in (_read(f) for f in Path(p["macro_runs"]).glob("*.json"))
                          if r and r.get("kind") == "replay"), key=lambda r: r["start"])
    keyword_days = set(((evaluation or {}).get("health") or {}).get("news_days_scored_by_keywords") or [])
    rows = predictions(memory, journal, keyword_days)
    card = scorecard(rows)
    trust = memory.get("weights", {})
    beliefs = (_facts(book, evaluation, replays) + _relationships(sens) + _findings(records)
               + _advisors(card, trust) + _phase_a(_read(p["phase_a"]), _read(p["contribution"]))
               + _assumptions(evaluation, journal, replays, _read(p["backtest"])))
    for i, b in enumerate(beliefs):
        b["id"] = f"b{i}"
        d = b["detail"]
        b["key"] = (d.get("key") or d.get("experiment") or d.get("advisor")
                    or (f"{d['symbol']}:{d['factor']}" if "symbol" in d else f"{b['topic']}:{i}"))
    latest = journal[-1] if journal else None
    working = None
    if latest:
        working = {
            "session": latest["session"],
            "stocks": [{"symbol": d["symbol"].replace(".NS", ""), "action": d["action"], "score": d["score"],
                        "wants": d["target_exposure"], "holds": d["current_exposure"],
                        "why": (d.get("reasons") or [""])[0]} for d in latest.get("decisions", [])],
            "material_news": sum(len(v.get("top", [])) for v in latest.get("news", {}).values()),
            "queued": len((book or {}).get("pending", [])),
            "pending_beliefs": sum(1 for o in memory.get("observations", []) if not o.get("judged")),
        }
    agenda = _read(p["agenda"]) or []
    return {
        "beliefs": beliefs, "scorecard": card, "calibration": {s: calibration(rows, s) for s in ("brain", *ADVISORS)},
        "predictions": {"judged": len({(r["date"], r["symbol"]) for r in rows}),
                        "days": len({r["date"] for r in rows}), "rows": rows},
        "trust": trust, "working": working,
        "open_questions": [a for a in agenda if a.get("status") != "answered"],
        "counts": {s: sum(1 for b in beliefs if b["status"] == s) for s in STATUSES},
        "memory": {
            "episodic": {"records": len(journal), "what": "journal sessions, each as it was on the day"},
            "procedural": {"records": len(memory.get("lessons", [])), "what": "lessons that moved trust"},
            "semantic": {"records": len((sens or {}).get("table", [])), "what": "measured stock-to-factor links"},
            "research": {"records": len(records), "what": "experiments on file"},
            "working": {"records": len(working["stocks"]) if working else 0, "what": "stocks with a view today"},
        },
    }
