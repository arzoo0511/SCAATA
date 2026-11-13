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


def baseline_spec() -> dict:
    return {
        "id": "live-rule-vs-holding", "title": "The live rule against holding", "kind": "baseline", "params": {},
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


def build_agenda(evaluation: dict | None, journal: list[dict], records: list[dict]) -> list[dict]:
    """The questions the live record raises today, each with the evidence that raised it."""
    done = {r["id"]: r for r in records}
    agenda = []

    def add(question: str, trigger: str, spec: dict | None, note: str = "") -> None:
        record = done.get(spec["id"]) if spec else None
        verdict = (record.get("review") or {}).get("verdict") if record else None
        agenda.append({"question": question, "trigger": trigger, "experiment": spec["id"] if spec else None,
                       "status": "answered" if verdict else ("proposed" if spec else "needs engine work"),
                       "verdict": verdict, "note": note})

    add("Does the live rule still beat holding on the latest data?",
        "Standing question: re-asked whenever the price snapshot is refreshed.", baseline_spec())
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
            f"Live, the brain wanted {gap[0]:.0%} invested on average and held {gap[1]:.0%}.", None,
            "The replay trades rupee amounts, so it cannot answer this yet. It needs a whole-share mode in the "
            "replay engine.")
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


def specs() -> dict[str, dict]:
    """Every experiment the director knows how to propose, by id."""
    all_specs = [baseline_spec(), regime_spec(), learning_spec(), *(ablation_spec(a) for a in ADVISOR_NAMES)]
    return {s["id"]: s for s in all_specs}
