"""The research director: turns what the live agent is doing into questions
worth testing, and says which are still open.

It is rules over the daily report card and the journal, not a model. Each
rule names its trigger, shows the number that fired it, and proposes an
experiment with its pass/fail criteria already written. The director can
only propose. It cannot run anything against the live book, and it cannot
change the agent.
"""
from __future__ import annotations

import json
from pathlib import Path

from scaata.config import ROOT_DIR

AGENDA_PATH = ROOT_DIR / "research" / "agenda.json"
COIN_FLIP, MIN_JUDGED = 0.50, 50
ADVISOR_NAMES = {"trend": "trend", "volatility": "the volatility brake", "momentum": "momentum"}
COMMON_LIMITS = [
    "Five stocks chosen because the owner already held them; they move together.",
    "The news advisor is silent in this replay: there is no scored news history before 2012.",
    "The replay trades rupee amounts; the live ₹10,000 book trades whole shares and is less invested.",
    "The holdout years (2020 onward) have been looked at before, during earlier tuning.",
]
METHOD = ("The real brain and memory replayed day by day on the frozen price snapshot: decide after each close, "
          "fill at the next open, pay NSE costs, idle cash in a liquid fund. Two spans: 2007-2019 (the years the "
          "rule was tuned on) and 2020 onward (the years it was not).")


def baseline_spec(month: str | None = None) -> dict:
    """The standing question. `month` (YYYY-MM) names a re-run on a newer price snapshot: results are
    never overwritten, so each re-run is its own experiment."""
    return {
        "id": "live-rule-vs-holding" + (f"-{month}" if month else ""),
        "title": "The live rule against holding" + (f" ({month})" if month else ""), "kind": "baseline", "params": {},
        "question": "Does the rule HANSEI trades with today beat simply holding the five stocks?",
        "hypothesis": "It falls less than holding and has a higher Sharpe, in both spans. It is not expected "
                      "to earn more.",
        "method": METHOD,
        "criteria": [
            {"metric": "tune.drawdown_vs_holding", "op": ">", "value": 0.0, "label": "smaller worst fall, 2007-2019"},
            {"metric": "holdout.drawdown_vs_holding", "op": ">", "value": 0.0, "label": "smaller worst fall, 2020 on"},
            {"metric": "tune.sharpe_vs_holding", "op": ">", "value": 0.0, "label": "higher Sharpe, 2007-2019"},
            {"metric": "holdout.sharpe_vs_holding", "op": ">", "value": 0.0, "label": "higher Sharpe, 2020 on"}],
        "limitations": COMMON_LIMITS,
    }


def ablation_spec(advisor: str) -> dict:
    name = ADVISOR_NAMES[advisor]
    return {
        "id": f"without-{advisor}", "title": f"HANSEI without {name}", "kind": "advisor_ablation",
        "params": {"drop": advisor},
        "question": f"Is {name} earning its place? What happens if it never speaks?",
        "hypothesis": f"Silencing {name} makes HANSEI worse: a lower Sharpe than the live rule in both spans.",
        "method": METHOD + f" The {advisor} advisor's view is set to 'no opinion' every day; its share of trust "
                           "sits idle, exactly as a silent advisor's does live.",
        "criteria": [
            {"metric": "tune.sharpe_vs_baseline", "op": "<", "value": 0.0, "label": "worse without it, 2007-2019"},
            {"metric": "holdout.sharpe_vs_baseline", "op": "<", "value": 0.0, "label": "worse without it, 2020 on"}],
        "limitations": COMMON_LIMITS,
    }


def learning_spec() -> dict:
    return {
        "id": "learning-switched-off", "title": "HANSEI with learning switched off", "kind": "brain_variant",
        "params": {"learning_rate": 0.0},
        "question": "Does shifting trust between advisors help, or would fixed equal trust do as well?",
        "hypothesis": "Learning helps: with it switched off, Sharpe is lower than the live rule in both spans.",
        "method": METHOD + " Trust stays at 25% each for the whole replay.",
        "criteria": [
            {"metric": "tune.sharpe_vs_baseline", "op": "<", "value": 0.0, "label": "worse without learning, 2007-2019"},
            {"metric": "holdout.sharpe_vs_baseline", "op": "<", "value": 0.0, "label": "worse without learning, 2020 on"}],
        "limitations": COMMON_LIMITS,
    }


FIXED_SHARE = round(1 / 252, 5)     # one expected change of best advisor a year, in trading sessions


def fixed_share_spec() -> dict:
    return {
        "id": "fixed-share-corrected", "title": "Fixed-Share in place of plain Hedge", "kind": "brain_variant",
        "params": {"fixed_share": FIXED_SHARE},
        "question": "Which advisor is best changes with the market. Plain Hedge is only built to find the single "
                    "best advisor over the whole record. Does Fixed-Share, which is built to follow a best "
                    "advisor that changes, do better?",
        "hypothesis": "With Fixed-Share, Sharpe is higher than with plain Hedge in both spans.",
        "method": METHOD + f" After every trust update, a fraction {FIXED_SHARE} of all trust is pooled and "
                           "returned equally to the four advisors (Herbster and Warmuth, 1998). The fraction was "
                           "fixed in advance from theory, as one expected change of best advisor a year; it was "
                           "not tuned on any result. The learning rate and the 5% trust floor are unchanged.",
        "criteria": [
            {"metric": "tune.sharpe_vs_baseline", "op": ">", "value": 0.0, "label": "better than plain Hedge, 2007-2019"},
            {"metric": "holdout.sharpe_vs_baseline", "op": ">", "value": 0.0, "label": "better than plain Hedge, 2020 on"}],
        "limitations": COMMON_LIMITS + [
            "The learner already keeps every advisor above 5% trust, so the anti-collapse benefit of Fixed-Share "
            "is partly present already; what is new is the steady pull back toward equal trust.",
            "An earlier experiment found learning itself (plain Hedge against fixed equal trust) made no "
            "measurable difference, which limits how much any change to the learner can show here."],
    }


def live_like_spec() -> dict:
    return {
        "id": "live-like-book", "title": "A book that behaves like the live one", "kind": "brain_variant",
        "params": {"whole_shares": True, "no_borrowing": True},
        "question": "The replay trades rupee amounts and can dip below zero cash. The live book buys whole "
                    "shares and cannot borrow. How much does that flatter the replay?",
        "hypothesis": "A book with the live book's share granularity and no borrowing does worse than the "
                      "rupee replay: a lower Sharpe in both spans.",
        "method": METHOD + " Each stock can be held only in whole lots, one lot being the share of a slot that "
                           "one share takes up in the live ₹10,000 book at the latest price; a buy is capped at "
                           "the cash on hand; an order that fills nothing does not restart the waiting period.",
        "criteria": [
            {"metric": "tune.sharpe_vs_baseline", "op": "<", "value": 0.0, "label": "worse than the rupee replay, 2007-2019"},
            {"metric": "holdout.sharpe_vs_baseline", "op": "<", "value": 0.0, "label": "worse than the rupee replay, 2020 on"}],
        "limitations": COMMON_LIMITS[:2] + [
            "Granularity is today's, held constant through history; the real book's would have changed with "
            "prices and with its own size.", COMMON_LIMITS[3]],
    }


def no_borrowing_spec() -> dict:
    return {
        "id": "no-borrowing", "title": "The replay without borrowing", "kind": "brain_variant",
        "params": {"no_borrowing": True},
        "question": "The replay engine lets cash go below zero when holdings outgrow their slots. Does that "
                    "borrowing account for any of its result?",
        "hypothesis": "Borrowing adds return: with buys capped at the cash on hand, yearly return is lower than "
                      "the live rule's in both spans.",
        "method": METHOD + " A buy is capped at the cash on hand; sells fill before buys.",
        "criteria": [
            {"metric": "tune.cagr_vs_baseline", "op": "<", "value": 0.0, "label": "lower return without borrowing, 2007-2019"},
            {"metric": "holdout.cagr_vs_baseline", "op": "<", "value": 0.0, "label": "lower return without borrowing, 2020 on"}],
        "limitations": COMMON_LIMITS,
    }


def shorter_wait_spec() -> dict:
    return {
        "id": "wait-five-sessions", "title": "Waiting 5 sessions between trades instead of 10",
        "kind": "brain_variant", "params": {"min_hold_days": 5},
        "question": "HANSEI trails holding while the stocks are below their six-month-ago level, by sitting out "
                    "the rebound. Would a shorter wait between trades get it back in sooner?",
        "hypothesis": "Waiting 5 sessions instead of 10 raises Sharpe against the live rule in both spans.",
        "method": METHOD + " The only change is the minimum number of sessions between trades in one stock.",
        "criteria": [
            {"metric": "tune.sharpe_vs_baseline", "op": ">", "value": 0.0, "label": "better than waiting 10, 2007-2019"},
            {"metric": "holdout.sharpe_vs_baseline", "op": ">", "value": 0.0, "label": "better than waiting 10, 2020 on"}],
        "limitations": COMMON_LIMITS + ["This setting was one of the nine in the earlier settings grid, so it is "
                                        "not a first look."],
    }


def regime_spec() -> dict:
    return {
        "id": "edge-by-market-condition", "title": "Where the edge comes from", "kind": "regime_study", "params": {},
        "question": "Under which market conditions does HANSEI beat holding, and under which does it lag?",
        "hypothesis": "It beats holding while the five-stock basket is in a downtrend and lags while it is in "
                      "an uptrend, in both spans.",
        "method": METHOD + " Each session is labelled from the previous close: is the equal-weight basket above "
                           "where it was 126 sessions ago (up or down), and is its 20-session volatility above its "
                           "one-year median (choppy or calm). Returns are annualised within each label.",
        "criteria": [
            {"metric": "tune.down.hansei_minus_holding", "op": ">", "value": 0.0, "label": "ahead in downtrends, 2007-2019"},
            {"metric": "holdout.down.hansei_minus_holding", "op": ">", "value": 0.0, "label": "ahead in downtrends, 2020 on"},
            {"metric": "tune.up.hansei_minus_holding", "op": "<", "value": 0.0, "label": "behind in uptrends, 2007-2019"},
            {"metric": "holdout.up.hansei_minus_holding", "op": "<", "value": 0.0, "label": "behind in uptrends, 2020 on"}],
        "limitations": COMMON_LIMITS + ["Descriptive: it splits one run by condition and has no bootstrap. Labels "
                                        "use the five stocks themselves, not the wider market."],
    }


def _exposure_gap(journal: list[dict]) -> tuple[float, float] | None:
    held, wanted = [], []
    for entry in journal or []:
        for d in entry.get("decisions", []):
            held.append(d["current_exposure"]); wanted.append(d["target_exposure"])
    return (sum(wanted) / len(wanted), sum(held) / len(held)) if held else None


def _baseline_due(records: list[dict], snapshot_last: str | None) -> str | None:
    """The month to re-ask the standing question for, if the price snapshot is newer than the last answer."""
    answered = [((r.get("results") or {}).get("data") or {}).get("last") for r in records
                if r["id"].startswith("live-rule-vs-holding") and r.get("results") is not None]
    answered = [a for a in answered if a]
    if snapshot_last and answered and snapshot_last[:7] > max(answered)[:7]:
        return snapshot_last[:7]
    # The live learner changed (Fixed-Share, 16 Nov 2026): the rule being traded is no longer the rule
    # the last answer was about, so the question is open again.
    from scaata.agent import memory

    learners = [((r.get("results") or {}).get("engine") or {}).get("live_learner", {}).get("fixed_share", 0.0)
                for r in records if r["id"].startswith("live-rule-vs-holding") and r.get("results") is not None]
    if learners and memory.SHARE not in learners:
        return "fixed-share"
    return None


def build_agenda(evaluation: dict | None, journal: list[dict], records: list[dict],
                 snapshot_last: str | None = None) -> list[dict]:
    """The questions the live record raises today, each with the evidence that raised it."""
    done = {r["id"]: r for r in records}
    agenda = []

    def add(question: str, trigger: str, spec: dict | None, note: str = "") -> None:
        record = done.get(spec["id"]) if spec else None
        verdict = (record.get("review") or {}).get("verdict") if record else None
        agenda.append({"question": question, "trigger": trigger, "experiment": spec["id"] if spec else None,
                       "status": "answered" if verdict else ("proposed" if spec else "needs engine work"),
                       "verdict": verdict, "note": note})

    due = _baseline_due(records, snapshot_last)
    add("Does the live rule still beat holding on the latest data?",
        ("Standing question: the live learner is now Fixed-Share, and the last answer was about plain Hedge."
         if due == "fixed-share" else
         f"Standing question: the price snapshot now runs to {snapshot_last}, newer than the last answer.")
        if due else "Standing question: re-asked whenever the price snapshot moves into a new month.",
        baseline_spec(due))
    add("Does Fixed-Share follow a changing best advisor better than plain Hedge?",
        "Chosen by the owner on 16 Nov 2026 as the one change to the learner with theory behind it for this "
        "problem: which advisor is best changes with the market.", fixed_share_spec())
    perf = (evaluation or {}).get("performance") or {}
    if perf and abs(perf.get("gap_from_exposure", 0)) > 3 * abs(perf.get("gap_from_selection", 0)):
        add("Is HANSEI's caution worth what it costs in a rally?",
            f"Live, {perf['gap_from_exposure']:+.2%} of the {perf['gap']:+.2%} gap to holding comes from being "
            f"on average {perf['average_invested']:.0%} invested; stock choice moved it {perf['gap_from_selection']:+.2%}.",
            regime_spec())
    scores = ((evaluation or {}).get("advisors") or {}).get("20_sessions", {})
    for advisor in ADVISOR_NAMES:
        score = scores.get(advisor) or {}
        if score.get("hit_rate") is not None and score.get("judged", 0) >= MIN_JUDGED and score["hit_rate"] < COIN_FLIP:
            add(f"Is {ADVISOR_NAMES[advisor]} earning its place?",
                f"Live, it was right {score['hit_rate']:.0%} of the time over {score['judged']} judged opinions "
                f"(20 sessions ahead): worse than a coin flip.", ablation_spec(advisor))
    trust = (evaluation or {}).get("trust") or {}
    if trust and max(trust.values()) - min(trust.values()) > 0.05:
        top, low = max(trust, key=trust.get), min(trust, key=trust.get)
        add("Does learning which advisors to trust actually help?",
            f"Live, trust has moved from equal to {top} {trust[top]:.0%} and {low} {trust[low]:.0%}.", learning_spec())
    gap = _exposure_gap(journal)
    if gap and gap[0] - gap[1] > 0.10:
        add("How much does whole-share rounding cost on a ₹10,000 book?",
            f"Live, the brain wanted {gap[0]:.0%} invested on average and held {gap[1]:.0%}.", live_like_spec())
        add("Does the replay's ability to borrow account for any of its result?",
            "Macroverse found the replay up to 109% invested in 2008; the live book cannot borrow.",
            no_borrowing_spec())
    regime = done.get("edge-by-market-condition")
    if regime and (regime.get("results") or {}).get("metrics", {}).get("holdout.down.hansei_minus_holding", 0) < 0:
        m = regime["results"]["metrics"]
        add("Would a shorter wait get it back in sooner after a fall?",
            f"The Lab found HANSEI {m['tune.down.hansei_minus_holding']:+.1%} and "
            f"{m['holdout.down.hansei_minus_holding']:+.1%} a year behind holding while the stocks are below "
            "their six-month-ago level.", shorter_wait_spec())
    health = (evaluation or {}).get("health") or {}
    if health.get("news_days_scored_by_keywords"):
        add("Do keyword-scored news days behave differently from model-scored ones?",
            f"{len(health['news_days_scored_by_keywords'])} live sessions fell back to keyword scoring.", None,
            "Too few days to test. Listed so it is not forgotten.")
    return agenda


def save_agenda(agenda: list[dict], path: Path | None = None) -> None:
    path = path or AGENDA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(agenda, indent=2, ensure_ascii=False), encoding="utf-8")


def load_agenda(path: Path | None = None) -> list[dict]:
    path = path or AGENDA_PATH
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def specs(snapshot_last: str | None = None) -> dict[str, dict]:
    """Every experiment the director knows how to propose, by id."""
    dated = ([baseline_spec(snapshot_last[:7])] if snapshot_last else []) + [baseline_spec("fixed-share")]
    all_specs = [baseline_spec(), *dated, regime_spec(), learning_spec(), fixed_share_spec(), live_like_spec(),
                 no_borrowing_spec(),
                 shorter_wait_spec(), *(ablation_spec(a) for a in ADVISOR_NAMES)]
    return {s["id"]: s for s in all_specs}
