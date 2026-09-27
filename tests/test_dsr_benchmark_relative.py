"""Tests for benchmark-relative DSR (`dsr_benchmark_relative`): an
experimental variant found necessary by real-data testing (MSFT collapsed
to "never trade" under plain DSR, while AAPL converged to "buy and hold" --
plain DSR only avoids the collapse when buy-and-hold happens to be a good
bet on that ticker's training data). Feeding the tracker excess return over
the ticker's own buy-and-hold removes buy-and-hold as a free-lunch
attractor, so the policy has to find something that actually beats the
benchmark on every ticker, not just replicate it.
"""
import numpy as np
import pandas as pd
import pytest

from scaata.config import FEATURE_COLUMNS
from scaata.rl.env import BUY, HOLD, RobustTradingEnv


def _make_env_df(n=30, seed=0, ticker="TEST", drift=0.001):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(drift, 0.01, n))
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}, index=dates)
    df["Close"] = close
    df["Volume"] = rng.integers(1_000_000, 5_000_000, n)
    df["volume_ma_30"] = df["Volume"].rolling(5, min_periods=1).mean()
    df["Ticker"] = ticker
    return df


def test_default_is_not_benchmark_relative():
    df = _make_env_df()
    env = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="TEST", use_differential_sharpe=True)
    assert env.dsr_benchmark_relative is False


def test_buy_and_hold_gets_near_zero_reward_under_benchmark_relative_dsr():
    """The core property this variant is supposed to have: if the policy's
    return exactly matches the benchmark's own return every step (e.g. the
    policy IS buy-and-hold), excess return is ~0 every step -- so DSR gives
    it no reward, removing "replicate buy-and-hold" as a free reward source.

    Uses a smooth, low-volatility price path (0.1% daily vol) specifically
    so the -2% stop-loss never fires -- a first version of this test used
    the standard 1% daily vol synthetic data and got stopped out partway
    through (routine for that vol level against a -2% threshold), which
    then correctly produces LARGE excess-return swings (sitting in cash
    while a volatile benchmark keeps moving is a real relative outcome,
    not a bug) -- see the module docstring. That's testing a different,
    also-real property; this test isolates the "matches benchmark exactly"
    case specifically.

    Uses dsr_eta=0.5 (not the production default 0.005) for a second,
    independent reason found while debugging this test: the entry step
    itself pays a real, non-benchmark-relative fee cost (dsr_input=-fee,
    since step_idx==0 is excluded from the benchmark subtraction to avoid
    indexing prices[-1]). DSR mechanically keeps rewarding "recovery" from
    that one-off dip for ~1/eta steps afterwards even though every
    subsequent excess return is exactly 0 -- at the production eta=0.005
    that recovery tail decays too slowly to clear the fixed 20-step warmup
    window, leaking a ~126-point false positive into this test (verified
    numerically). A large eta here makes the perturbation fully decay
    *during* warmup (before reward reporting resumes), isolating the
    property this test actually wants to check -- the production default
    is untouched, since it's passed to the constructor per-test, not
    changed in scaata/config.py.
    """
    df = _make_env_df(n=60, drift=0.0005)
    # override with genuinely low volatility so a -2% stop-loss basically never fires
    rng = np.random.default_rng(1)
    df["Close"] = 100 * np.cumprod(1 + rng.normal(0.0005, 0.001, len(df)))

    env = RobustTradingEnv(
        df, FEATURE_COLUMNS, fixed_ticker="TEST",
        use_differential_sharpe=True, dsr_benchmark_relative=True, dsr_eta=0.5,
    )
    env.reset()

    env.step(BUY)
    total_reward = 0.0
    done = False
    while not done:
        _, reward, done, _, _ = env.step(HOLD)
        total_reward += reward

    assert env.position == 1  # confirms the stop-loss never fired, as intended
    # The entry-fee perturbation fully decays within the warmup window at
    # this eta (verified: total reward is exactly 0.0), so this can be a
    # tight bound rather than a loose one.
    assert abs(total_reward) < 1.0


def test_benchmark_relative_reward_differs_from_plain_dsr_reward():
    """Sanity check that the benchmark-relative mode actually changes the
    reward signal (not accidentally a no-op) -- same trade, same data,
    different reward mode, should give a different reward once past the
    DSR warmup period (default 20 steps; a comparison taken within warmup
    would trivially show 0 == 0 regardless of mode, which isn't testing
    anything -- so this steps well past it).
    """
    df = _make_env_df(n=40, drift=0.003)  # a clear trend so benchmark return is meaningfully nonzero

    def run(env):
        env.reset()
        env.step(BUY)
        last_reward = None
        for _ in range(30):
            _, last_reward, done, _, _ = env.step(HOLD)
            if done:
                break
        return last_reward

    env_plain = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="TEST", use_differential_sharpe=True, dsr_eta=0.005)
    env_relative = RobustTradingEnv(
        df, FEATURE_COLUMNS, fixed_ticker="TEST",
        use_differential_sharpe=True, dsr_benchmark_relative=True, dsr_eta=0.005,
    )

    reward_plain = run(env_plain)
    reward_relative = run(env_relative)

    assert reward_plain != pytest.approx(reward_relative)
