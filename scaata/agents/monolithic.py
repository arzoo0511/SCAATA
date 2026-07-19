"""Monolithic-agent baseline for Phase 5's research question: does explicit
multi-agent decomposition (Scraper -> Normalizer -> Regime-Classifier ->
Meta-Selector <-> Critique) actually beat an equally-capable single LLM
call doing the same job in one shot?

Scope note: the decomposed pipeline's meta-selector/critique steps are
numeric ML training and a rule-based penalty computation, not LLM calls —
there's no monolithic-LLM equivalent for "iteratively retrain a classifier"
to compare against. The scoped, honest comparison this module enables is:
does a single LLM's one-shot self-assessed confidence in a strategy
correlate with that strategy's *actual* historical profitability, versus
the decomposed pipeline's meta-selector/critique-derived weight (which is
explicitly grounded in realized returns and self-critique penalties, not
a holistic LLM impression)? See `scaata/evaluation/agent_comparison.py`.
"""
from __future__ import annotations

import os

import numpy as np

from scaata.strategies.normalizer import _strip_markdown_fence

MONOLITHIC_PROMPT_TEMPLATE = """
You are an expert quantitative developer and risk reviewer. Given the raw Python trading
strategy below, do two things in one response:
1. Rewrite its core logic into a Python function `strategy(df)` returning integer signals
   in {{-1, 0, 1}} (sell/hold/buy), using columns Open/High/Low/Close/Volume.
2. Assign it an initial confidence weight in [0.05, 1.0] reflecting how sound/well-specified
   the logic appears (lower for vague/overfit-looking/momentum-chasing logic, higher for
   clear, well-reasoned rules).

Output EXACTLY two lines: first the weight as a bare float, then the code (no markdown
fences, no explanation, no other text).

Raw code:
{raw_code}
"""


def _groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    from groq import Groq

    return Groq(api_key=api_key)


def monolithic_normalize_and_weight(raw_code: str, model: str = "llama-3.1-8b-instant", client=None) -> dict | None:
    """One LLM call producing both a normalized strategy and its initial
    pool weight — the monolithic equivalent of normalizer_node plus the
    first meta_selector_node weight assignment, combined into a single call.
    Returns None (caller should fall back to `mock_monolithic_pass`) if no
    GROQ_API_KEY is configured or the call/parse fails.
    """
    client = client or _groq_client()
    if client is None:
        return None

    prompt = MONOLITHIC_PROMPT_TEMPLATE.format(raw_code=raw_code[:1500])
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model, temperature=0.0, max_tokens=1000,
        )
        text = completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Monolithic LLM call failed: {e}")
        return None

    lines = text.split("\n", 1)
    if len(lines) < 2:
        return None
    try:
        weight = float(lines[0].strip())
    except ValueError:
        return None

    code = _strip_markdown_fence(lines[1].strip())
    return {"clean_code": code, "weight": max(0.05, min(1.0, weight))}


def mock_monolithic_pass(raw_strategies: list[dict], seed: int = 0) -> list[dict]:
    """Offline fallback when GROQ_API_KEY isn't set: assigns a fixed random
    weight per strategy since there's no LLM self-assessment to draw on.
    Keeps the comparison harness runnable/testable without live API access
    — but the actual research question (does an LLM's self-assessed
    confidence track real profitability?) can only be answered using the
    live LLM path; the mock path only proves the harness runs end-to-end.
    """
    rng = np.random.default_rng(seed)
    return [
        {"source": strat["source"], "clean_code": strat["code"], "weight": float(rng.uniform(0.4, 0.8))}
        for strat in raw_strategies
    ]
