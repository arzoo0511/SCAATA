"""Tests for the LLM-agent-only baseline: the mock fallback must run
end-to-end without a live key, and must never be mislabeled as a real
LLM decision — the whole point of tagging `source` is to make that
distinction impossible to lose downstream.
"""
import numpy as np
import pandas as pd
import pytest

from scaata.config import FEATURE_COLUMNS
from scaata.rl.env import BUY, HOLD, SELL
from scaata.rl.llm_baseline import backtest_llm_agent, llm_decide, _parse_action


def _make_synthetic_test_df(n=60, seed=0, ticker="TEST"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}, index=dates)
    df["Close"] = close
    df["Ticker"] = ticker
    return df


def test_llm_decide_uses_mock_when_no_client_configured(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    row = {c: 0.0 for c in FEATURE_COLUMNS}
    action, source = llm_decide(row, position=0, client=None)
    assert source == "mock"
    assert action in (HOLD, BUY, SELL)


def test_parse_action_extracts_the_right_signal():
    assert _parse_action("BUY") == BUY
    assert _parse_action("I recommend SELL here") == SELL
    assert _parse_action("HOLD for now") == HOLD
    assert _parse_action("unrelated gibberish") == HOLD


class _FakeGroqResponse:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})]


class _FakeGroqClient:
    """A test double standing in for a real Groq client, so the "llm" code
    path (source labeling, prompt formatting) is exercised without any
    live network access."""

    def __init__(self, reply="BUY"):
        self.reply = reply
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        return _FakeGroqResponse(self.reply)


def test_llm_decide_labels_source_as_llm_when_client_provided():
    row = {c: 0.0 for c in FEATURE_COLUMNS}
    action, source = llm_decide(row, position=0, client=_FakeGroqClient(reply="BUY"))
    assert source == "llm"
    assert action == BUY


def test_backtest_llm_agent_runs_end_to_end_in_mock_mode(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    test_df = _make_synthetic_test_df()

    equity, actions, source = backtest_llm_agent(test_df, FEATURE_COLUMNS, "TEST", client=None)

    assert source == "mock"
    assert len(equity) == len(test_df)
    assert len(actions) == len(test_df) - 1
    assert equity[0] > 0
    assert all(a in (HOLD, BUY, SELL) for a in actions)


def test_backtest_llm_agent_reports_llm_source_when_fake_client_used():
    test_df = _make_synthetic_test_df(seed=1)
    equity, actions, source = backtest_llm_agent(
        test_df, FEATURE_COLUMNS, "TEST", client=_FakeGroqClient(reply="HOLD")
    )
    assert source == "llm"
    assert len(equity) == len(test_df)
