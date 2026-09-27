"""LLM-agent-only baseline: no RL training at all, just prompts an LLM
directly for a buy/hold/sell decision at each trading day. A genuinely
different comparison point from buy-and-hold/rule-based/PPO — tests
whether the structured RL + self-critique + meta-selection system in this
rebuild actually beats a naive "ask an LLM every day" agent, a directly
relevant question given the project's multi-agent framing.

Falls back to a clearly-labeled placeholder decision (not silently
reusing the rule-based baseline) when no `GROQ_API_KEY` is configured, so
the mock path can never be mistaken for a real LLM decision downstream —
every decision function returns its `source` ("llm" or "mock") alongside
the action.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from scaata.config import INITIAL_CASH
from scaata.rl.env import BUY, HOLD, SELL

LLM_DECISION_PROMPT_TEMPLATE = """
You are a trading agent. Given the following market indicators for one stock on a single day,
decide whether to BUY, SELL, or HOLD. Respond with exactly one word: BUY, SELL, or HOLD.

Log return: {returns:.4f}
10-day MA: {ma_10:.4f}
50-day MA: {ma_50:.4f}
10-day volatility: {volatility:.4f}
10-day momentum: {momentum:.4f}
14-day RSI: {rsi:.4f}
30-day volume MA: {volume_ma_30:.4f}
Current position: {position_desc}
"""


def _groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    from groq import Groq

    return Groq(api_key=api_key)


def _mock_decide(row: dict, position: int) -> int:
    """Deterministic placeholder used only when no GROQ_API_KEY is
    configured — NOT a real LLM decision. Uses a simple momentum rule so
    the mock path still produces a plausible-looking equity curve for
    harness testing, without ever being confused for a live call (callers
    get "mock" back from `llm_decide`'s second return value, never "llm")."""
    momentum = row.get("momentum", 0.0)
    if position == 0 and momentum > 0:
        return BUY
    if position == 1 and momentum < 0:
        return SELL
    return HOLD


def _parse_action(text: str) -> int:
    text = text.strip().upper()
    if "BUY" in text:
        return BUY
    if "SELL" in text:
        return SELL
    return HOLD


def llm_decide(
    row: dict, position: int, model: str = "llama-3.1-8b-instant", client=None
) -> tuple[int, str]:
    """Returns (action, source) where source is "llm" or "mock" so callers
    can never mistake a placeholder decision for a real one."""
    client = client if client is not None else _groq_client()
    if client is None:
        return _mock_decide(row, position), "mock"

    position_desc = "holding shares" if position == 1 else "no position (cash)"
    prompt = LLM_DECISION_PROMPT_TEMPLATE.format(position_desc=position_desc, **row)
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model, temperature=0.0, max_tokens=5,
        )
        text = completion.choices[0].message.content
    except Exception as e:
        print(f"LLM baseline call failed, falling back to mock decision: {e}")
        return _mock_decide(row, position), "mock"

    return _parse_action(text), "llm"


def backtest_llm_agent(
    test_df: pd.DataFrame,
    feature_columns: list[str],
    ticker: str,
    model: str = "llama-3.1-8b-instant",
    client=None,
    fee: float = 0.0,
    use_llm: bool = True,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Backtests the LLM-agent-only baseline over `test_df` for `ticker`.
    Returns (equity_curve, actions, source) where `source` is "llm" only if
    every single decision came from a live call, else "mock" — transparent
    about which regime actually produced the result, not just at the
    per-decision level.

    `use_llm=False` runs the deterministic momentum fallback only, never
    contacting Groq even when a key is configured. `fee` is charged per
    trade side (default 0, the original fee-free baseline).
    """
    ticker_df = test_df[test_df["Ticker"] == ticker].reset_index(drop=True)
    if use_llm:
        client = client if client is not None else _groq_client()

    cash, shares, position = INITIAL_CASH, 0.0, 0
    equity = [cash]
    actions = []
    sources_used = set()

    for i in range(len(ticker_df) - 1):
        row = {col: ticker_df[col].iloc[i] for col in feature_columns}
        if use_llm:
            action, source = llm_decide(row, position, model=model, client=client)
        else:
            action, source = _mock_decide(row, position), "mock"
        sources_used.add(source)

        price = ticker_df["Close"].iloc[i]
        if action == BUY and position == 0:
            shares = cash * (1 - fee) / price
            cash = 0.0
            position = 1
        elif action == SELL and position == 1:
            cash = shares * price * (1 - fee)
            shares = 0.0
            position = 0

        equity.append(cash + shares * price)
        actions.append(action)

    overall_source = "llm" if sources_used == {"llm"} else "mock"
    return np.array(equity), np.array(actions), overall_source
