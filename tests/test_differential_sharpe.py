"""Tests for the Differential Sharpe Ratio reward (Moody & Saffell 2001):
the formula must match a hand-computed reference, the first step (and any
zero-variance state) must be safely defined as 0 rather than crashing, a
flat/never-trading policy must net exactly 0 cumulative reward (the same
as the old scheme -- this mechanism doesn't inherently punish inaction,
its benefit is elsewhere), and -- the actual falsifiable behavioral claim
-- two return sequences with the SAME mean but different variance must
accumulate different total reward, with the lower-variance one scoring
higher, since that's what "risk-adjusted" is supposed to mean here.
"""
import numpy as np
import pytest

from scaata.config import DSR_ETA
from scaata.rl.reward import DifferentialSharpeTracker, differential_sharpe_reward


def test_first_step_is_safely_zero_not_a_crash():
    reward, new_A, new_B = differential_sharpe_reward(0.01, A_prev=0.0, B_prev=0.0, eta=0.1)
    assert reward == 0.0
    assert new_A == pytest.approx(0.001)
    assert new_B == pytest.approx(0.00001)


def test_matches_hand_computed_reference_sequence():
    # Reference values computed independently by hand (see conversation
    # sanity-check script) before wiring this into the tracker/env.
    returns = [0.01, 0.01, -0.005, 0.01, 0.02, -0.01, 0.015]
    expected_rewards = [0.0, 1.666667, -2.265834, 1.600875, 0.945091, -2.754486, 1.346651]

    A, B = 0.0, 0.0
    for r, expected in zip(returns, expected_rewards):
        reward, A, B = differential_sharpe_reward(r, A, B, eta=0.1)
        assert reward == pytest.approx(expected, abs=1e-5)


def test_flat_policy_nets_exactly_zero_cumulative_reward():
    """A policy that never trades has return 0 every step -- A and B stay
    at 0, variance estimate stays at 0, reward is defined as 0 throughout.
    This mechanism does not inherently penalize inaction any more than the
    old percent_change*10 reward did (also 0 for a flat policy) -- its
    benefit is elsewhere (no separately-tunable penalty coefficients that
    can dominate and bias training toward inaction).
    """
    tracker = DifferentialSharpeTracker(eta=0.1)
    total_reward = sum(tracker.step(0.0) for _ in range(50))
    assert total_reward == 0.0


def _low_and_high_vol_same_mean_sequences(n=200, mean_return=0.001, seed=0):
    rng = np.random.default_rng(seed)
    low_vol_returns = rng.normal(mean_return, 0.002, n)
    high_vol_returns = rng.normal(mean_return, 0.02, n)
    # Force identical realized means so this isn't just "got a luckier draw".
    low_vol_returns = low_vol_returns - low_vol_returns.mean() + mean_return
    high_vol_returns = high_vol_returns - high_vol_returns.mean() + mean_return
    return low_vol_returns, high_vol_returns


def _cumulative_reward(returns, eta):
    tracker = DifferentialSharpeTracker(eta=eta)
    return sum(tracker.step(r) for r in returns)


def test_lower_variance_same_mean_scores_higher_cumulative_reward_at_small_eta():
    """The actual falsifiable "risk-adjusted" claim: two return sequences
    with identical mean but different variance must NOT score the same --
    the lower-variance one should accumulate more reward, since that's
    what distinguishes a Sharpe-like reward from a raw-return reward.

    This only holds for small `eta` -- see
    `test_large_eta_inverts_the_variance_ranking_do_not_raise_config_default`
    directly below for why the config default must stay in this range.
    """
    low_vol_returns, high_vol_returns = _low_and_high_vol_same_mean_sequences()

    low_vol_total = _cumulative_reward(low_vol_returns, eta=DSR_ETA)
    high_vol_total = _cumulative_reward(high_vol_returns, eta=DSR_ETA)

    assert low_vol_total > high_vol_total, (
        "the lower-variance return sequence (same mean) should accumulate more DSR reward at "
        "config.DSR_ETA -- otherwise this reward isn't actually risk-adjusted, just a rescaled "
        "raw-return reward"
    )


def test_large_eta_inverts_the_variance_ranking_do_not_raise_config_default():
    """Found empirically while implementing this: the Differential Sharpe
    Ratio is a first-order approximation valid only for small `eta`. At
    `eta >= 0.01` (over ~200-step sequences), the ranking above INVERTS --
    cumulative reward favors the higher-variance sequence, the opposite of
    a risk-adjusted signal, which would actively train a policy toward
    more volatile behavior. This test pins that failure mode down so it's
    never silently reintroduced by someone raising `DSR_ETA` for a bigger
    reward magnitude without re-checking this.
    """
    low_vol_returns, high_vol_returns = _low_and_high_vol_same_mean_sequences()

    low_vol_total = _cumulative_reward(low_vol_returns, eta=0.05)
    high_vol_total = _cumulative_reward(high_vol_returns, eta=0.05)

    assert low_vol_total < high_vol_total, (
        "expected the known-broken large-eta regime to still be broken -- if this now passes, "
        "either the formula changed or eta=0.05 is no longer representative; re-verify before "
        "assuming eta can be safely raised"
    )


def test_tracker_reset_clears_state():
    tracker = DifferentialSharpeTracker(eta=0.1)
    tracker.step(0.01)
    tracker.step(0.02)
    assert tracker.A != 0.0 or tracker.B != 0.0
    tracker.reset()
    assert tracker.A == 0.0
    assert tracker.B == 0.0


def test_negative_return_gives_negative_reward_once_variance_established():
    # warmup_steps=0 here since this test targets the core signal logic
    # (does a bad return get penalized once there's a variance estimate to
    # judge it against), not the practical burn-in safety wrapper covered
    # by the warmup-specific tests below.
    tracker = DifferentialSharpeTracker(eta=0.1, warmup_steps=0)
    tracker.step(0.01)  # establish some variance context
    tracker.step(0.01)
    reward = tracker.step(-0.02)
    assert reward < 0


def test_warmup_steps_force_zero_reward_regardless_of_underlying_signal():
    """The burn-in safety net: even a return that would otherwise produce
    a large reward must be suppressed to exactly 0.0 while still within
    `warmup_steps`, and the underlying A/B state must still update
    normally (only the returned reward is suppressed).
    """
    tracker = DifferentialSharpeTracker(eta=0.1, warmup_steps=5)
    for _ in range(5):
        reward = tracker.step(0.02)
        assert reward == 0.0
    assert tracker.A != 0.0 or tracker.B != 0.0  # state still updated during warmup

    # the 6th step is past warmup and should produce a real, nonzero signal
    reward = tracker.step(0.02)
    assert reward != 0.0


def test_reward_clip_bounds_extreme_values():
    """Regression test for a real bug caught via real-environment testing:
    a tiny-but-nonzero variance estimate (not caught by
    `DSR_VARIANCE_EPSILON`) combined with even a modest return produced a
    reward in the millions before this clip existed (a single ~1% return
    at step 2 of a real RobustTradingEnv episode). The clip must bound the
    reward to `[-reward_clip, +reward_clip]` regardless of how extreme the
    raw ratio becomes.
    """
    tracker = DifferentialSharpeTracker(eta=0.005, warmup_steps=0, reward_clip=10.0)
    tracker.step(0.0001)  # tiny return -> tiny (but nonzero) variance estimate
    reward = tracker.step(0.05)  # then a comparatively large return

    assert abs(reward) <= 10.0
