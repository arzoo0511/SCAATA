"""Tests for fair-comparison options on the baselines: the same fee PPO
pays, full-history moving averages for the rule-based baseline, and a
momentum fallback that never calls an LLM. Defaults must reproduce the
original fee-free behaviour exactly."""
import numpy as np
import pandas as pd
import pytest

from scaata.rl import llm_baseline
from scaata.rl.env import BUY, HOLD
from scaata.rl.train import buy_and_hold_equity, rule_based_equity, rule_based_signals


def _df(n=120, ticker="TST"):
    close = np.r_[np.linspace(100, 130, n // 2), np.linspace(130, 95, n - n // 2)]
    ma_10 = np.where(np.arange(n) < n // 2, 120.0, 100.0)  # full-history MAs: bullish, then bearish
    ma_50 = np.full(n, 110.0)
    return pd.DataFrame({
        "Close": close, "Ticker": ticker, "ma_10": ma_10, "ma_50": ma_50,
        "momentum": np.where(np.arange(n) < n // 2, 0.5, -0.5),
    }, index=pd.bdate_range("2024-01-01", periods=n))


def test_buy_and_hold_default_is_the_original_fee_free_curve():
    df = _df()
    prices = df["Close"].values
    np.testing.assert_allclose(buy_and_hold_equity(df, "TST"), 10_000.0 / prices[0] * prices)


def test_buy_and_hold_fee_is_charged_once_on_entry():
    df = _df()
    free = buy_and_hold_equity(df, "TST")
    paid = buy_and_hold_equity(df, "TST", fee=0.001)
    assert paid[0] == 10_000.0
    assert paid[-1] == pytest.approx(free[-1] * 0.999)


def test_rule_based_default_still_idles_for_warmup():
    assert (rule_based_signals(_df())[:50] == HOLD).all()


def test_rule_based_precomputed_mas_trade_from_day_one():
    signals = rule_based_signals(_df(), use_precomputed_mas=True)
    assert signals[0] == BUY


def test_rule_based_fee_lowers_equity_when_it_trades():
    df = _df()
    free, actions = rule_based_equity(df, "TST", use_precomputed_mas=True)
    paid, _ = rule_based_equity(df, "TST", use_precomputed_mas=True, fee=0.001)
    assert (actions != HOLD).any()
    assert paid[-1] < free[-1]


def test_momentum_fallback_never_calls_an_llm_even_with_a_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "would-be-real")

    def _explode():
        raise AssertionError("use_llm=False must not construct an LLM client")

    monkeypatch.setattr(llm_baseline, "_groq_client", _explode)

    equity, actions, source = llm_baseline.backtest_llm_agent(_df(), ["momentum"], "TST", use_llm=False)

    assert source == "mock"
    assert len(equity) == len(_df())


def test_momentum_fallback_fee_lowers_equity(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    free, _, _ = llm_baseline.backtest_llm_agent(_df(), ["momentum"], "TST", use_llm=False)
    paid, _, _ = llm_baseline.backtest_llm_agent(_df(), ["momentum"], "TST", use_llm=False, fee=0.001)
    assert paid[-1] < free[-1]
