"""3rd Eye narrative generation: turns the correlator's numeric summary
into a short explanation of what changed between the 2020-era and
2026-era slices — grounded in already-computed numbers (not asking the LLM
to "discover" the correlation itself), which keeps this auditable and
reduces hallucination risk. Falls back to a deterministic template (no LLM
call) when GROQ_API_KEY isn't configured, so this stays testable offline.

`faithfulness_check` is the concrete guard against the LLM (or the
template) overclaiming beyond what the numbers actually show — it checks
the narrative's directional claims (stronger/weaker relationship, more/less
coverage) against the report's own sign and magnitude comparisons.
"""
from __future__ import annotations

import os
import re

NARRATIVE_PROMPT_TEMPLATE = """
You are a quantitative research assistant. Given the following numeric summary comparing
an early period ("2020-era") and a later period ("2026-era") of sentiment-vs-market-behavior
correlation, write a short (3-4 sentence) factual narrative describing what changed.

Only state what is directly supported by these numbers. Do not speculate about causes not
present in the data. If coverage volume differs substantially between eras, mention that as
a possible confound.

2020-era: {era_early}
2026-era: {era_late}
Granger causality (sentiment -> volatility): {granger}
"""


def _groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    from groq import Groq

    return Groq(api_key=api_key)


def _template_narrative(report: dict, granger: dict) -> str:
    early, late = report["era_early"], report["era_late"]
    corr_change = "increased" if (late["corr_vol"] or 0) > (early["corr_vol"] or 0) else "decreased"
    coverage_note = (
        f"Coverage volume changed from {early['mean_coverage_volume']:.1f} to {late['mean_coverage_volume']:.1f} "
        "mentions/day on average, which may partly explain any correlation shift rather than a genuine "
        "change in the news-market relationship."
    )
    granger_note = (
        f"A Granger causality test found sentiment had its strongest predictive lead at lag "
        f"{granger.get('best_lag', 'N/A')} (p={granger.get('best_p_value', float('nan'))})."
        if "error" not in granger else "Granger causality could not be computed (insufficient data)."
    )
    return (
        f"The sentiment-volatility correlation {corr_change} from {early['corr_vol']:.3f} in the 2020-era "
        f"slice to {late['corr_vol']:.3f} in the 2026-era slice. {coverage_note} {granger_note}"
    )


def generate_narrative(report: dict, granger: dict, model: str = "llama-3.1-8b-instant") -> str:
    client = _groq_client()
    if client is None:
        return _template_narrative(report, granger)

    prompt = NARRATIVE_PROMPT_TEMPLATE.format(era_early=report["era_early"], era_late=report["era_late"], granger=granger)
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model, temperature=0.2, max_tokens=300,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM narrative generation failed, falling back to template: {e}")
        return _template_narrative(report, granger)


def faithfulness_check(narrative: str, report: dict) -> dict:
    """Rule-based faithfulness check: confirms the narrative's directional
    claim ("increased"/"decreased"/"stronger"/"weaker") about the
    sentiment-volatility correlation matches the report's actual sign
    comparison, and that it doesn't claim a specific correlation number
    that contradicts the report. This is intentionally a simple, auditable
    check rather than a second LLM-judge call, so it has no dependency on
    live API access.
    """
    early_corr = report["era_early"]["corr_vol"]
    late_corr = report["era_late"]["corr_vol"]
    if early_corr is None or late_corr is None or (early_corr != early_corr) or (late_corr != late_corr):
        return {"checked": False, "reason": "insufficient data in report to check against"}

    actual_direction = "increased" if late_corr > early_corr else "decreased"
    text = narrative.lower()

    claims_increase = bool(re.search(r"\b(increased|stronger|rose|grew)\b", text))
    claims_decrease = bool(re.search(r"\b(decreased|weaker|fell|declined)\b", text))

    if claims_increase and claims_decrease:
        return {"checked": True, "faithful": False, "reason": "narrative makes contradictory directional claims"}
    if not claims_increase and not claims_decrease:
        return {"checked": True, "faithful": None, "reason": "no directional claim detected to verify"}

    claimed_direction = "increased" if claims_increase else "decreased"
    faithful = claimed_direction == actual_direction
    return {
        "checked": True,
        "faithful": faithful,
        "actual_direction": actual_direction,
        "claimed_direction": claimed_direction,
    }
