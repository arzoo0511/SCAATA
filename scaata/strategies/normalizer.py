"""LLM strategy normalizer, ported from the v1 notebook + one real fix:
the original notebook accepted any LLM output containing the literal
substring "def strategy(" into the pool with no execution check, so a
subtly broken rewrite could silently enter the strategy pool. This version
adds an execute-to-validate step (run the candidate on a small sample
DataFrame) before acceptance.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from scaata.strategies.pool import run_strategy_safely

load_dotenv()

NORMALIZE_PROMPT_TEMPLATE = """
You are an expert quantitative developer. I will provide you with a raw Python trading strategy scraped from GitHub.
Your task is to extract the core trading logic and rewrite it into a Python function named `strategy(df)` that takes a pandas DataFrame (with columns: Open, High, Low, Close, Volume) and returns a numpy integer array of signals: 1 (buy/long), -1 (sell/short), or 0 (hold).

If the script uses an external library like backtrader, convert its logic into pure pandas vectorization!

ONLY output the Python code. Do not output markdown code block formatting like ```python, just the raw code. Do not output any explanations. Your output must strictly contain the function `strategy(df)`.

Raw code:
{raw_code}
"""


def _groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    from groq import Groq

    return Groq(api_key=api_key)


def normalize_strategy_llm(raw_code: str, model: str = "llama-3.1-8b-instant", client=None) -> str | None:
    client = client or _groq_client()
    if client is None:
        print("Warning: GROQ_API_KEY not set — cannot normalize via LLM.")
        return None

    prompt = NORMALIZE_PROMPT_TEMPLATE.format(raw_code=raw_code[:1500])
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=1000,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM conversion failed: {e}")
        return None


def _strip_markdown_fence(code: str) -> str:
    if "```python" in code:
        return code.split("```python")[1].split("```")[0].strip()
    if "```" in code:
        parts = code.split("```")
        if len(parts) >= 3:
            return parts[1].strip()
    return code


def _make_validation_sample(n: int = 60) -> pd.DataFrame:
    """A small synthetic OHLCV sample used only to check that a normalized
    strategy executes and returns the right shape/dtype before it's
    accepted into the pool."""
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n),
    })


def validate_strategy_code(code: str) -> bool:
    """Execute-to-validate: confirm the candidate actually runs and returns
    a same-length integer signal array, before it's trusted enough to enter
    the strategy pool. This is the concrete fix for the v1 gap where any
    LLM output containing "def strategy(" was accepted unchecked."""
    sample = _make_validation_sample()
    signals = run_strategy_safely(code, sample)
    return signals is not None and len(signals) == len(sample)


def normalize_and_validate(raw_strategies: list[dict], model: str = "llama-3.1-8b-instant") -> list[dict]:
    """Normalize each raw strategy via the LLM, keep only those that both
    contain `def strategy(` and pass execution validation."""
    client = _groq_client()
    normalized = []
    for strat in raw_strategies:
        clean_code = normalize_strategy_llm(strat["code"], model=model, client=client)
        if not clean_code:
            continue
        clean_code = _strip_markdown_fence(clean_code)
        if "def strategy(" not in clean_code:
            continue
        if not validate_strategy_code(clean_code):
            print(f"Rejected (failed execute-to-validate check): {strat['source']}")
            continue
        normalized.append({"source": strat["source"], "clean_code": clean_code})
    return normalized


def validate_mock_strategies(mock_strategies: list[dict]) -> list[dict]:
    """For offline/mock mode: the mock strategies are already in
    `strategy(df)` form (no LLM call needed), so just run them through the
    same execute-to-validate check for a consistent code path."""
    normalized = []
    for strat in mock_strategies:
        if validate_strategy_code(strat["code"]):
            normalized.append({"source": strat["source"], "clean_code": strat["code"]})
    return normalized
