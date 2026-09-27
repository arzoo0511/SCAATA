"""Tests for novelty-modulated Hedge dynamics (Phase 10c): high novelty
must (a) increase the effective learning rate continuously (not via an
if/else regime branch) and (b) shift weight toward the sentiment expert in
proportion to measured novelty, even absent any loss difference.
"""
import numpy as np

from scaata.strategies.hedge import (
    HedgeWeights,
    hedge_update_with_novelty,
    novelty_modulated_eta,
    sentiment_log_prior,
)


def test_novelty_modulated_eta_increases_continuously_with_novelty():
    eta_base = 0.5
    eta_calm = novelty_modulated_eta(eta_base, novelty_score=0.0, kappa=1.0)
    eta_mid = novelty_modulated_eta(eta_base, novelty_score=0.5, kappa=1.0)
    eta_high = novelty_modulated_eta(eta_base, novelty_score=1.0, kappa=1.0)

    assert eta_calm == eta_base  # zero novelty -> no modulation
    assert eta_calm < eta_mid < eta_high
    assert eta_high == eta_base * 2  # kappa=1, novelty=1 -> double the base rate


def test_sentiment_log_prior_is_zero_at_zero_novelty():
    prior = sentiment_log_prior(n_experts=4, sentiment_expert_idx=2, novelty_score=0.0, gamma=0.5)
    np.testing.assert_allclose(prior, np.zeros(4))


def test_sentiment_log_prior_only_boosts_the_sentiment_expert():
    prior = sentiment_log_prior(n_experts=4, sentiment_expert_idx=2, novelty_score=1.0, gamma=0.5)
    assert prior[2] == 0.5
    assert (prior[[0, 1, 3]] == 0.0).all()


def test_high_novelty_shifts_weight_toward_sentiment_expert_despite_equal_losses():
    n_experts = 3
    sentiment_idx = 1
    equal_losses = np.array([0.5, 0.5, 0.5])

    hedge_calm = HedgeWeights(n_experts=n_experts, eta=0.5)
    weights_calm = hedge_update_with_novelty(hedge_calm, equal_losses, novelty_score=0.0, sentiment_expert_idx=sentiment_idx)

    hedge_shock = HedgeWeights(n_experts=n_experts, eta=0.5)
    weights_shock = hedge_update_with_novelty(hedge_shock, equal_losses, novelty_score=1.0, sentiment_expert_idx=sentiment_idx)

    # With equal losses and zero novelty, weights should stay uniform.
    np.testing.assert_allclose(weights_calm, np.ones(n_experts) / n_experts)
    # With high novelty, the sentiment expert should pull ahead purely from the prior.
    assert weights_shock[sentiment_idx] > weights_calm[sentiment_idx]
    assert weights_shock[sentiment_idx] == weights_shock.max()


def test_no_sentiment_expert_applies_eta_modulation_only():
    hedge = HedgeWeights(n_experts=3, eta=0.5)
    losses = np.array([0.0, 1.0, 1.0])

    weights = hedge_update_with_novelty(hedge, losses, novelty_score=1.0, sentiment_expert_idx=None)

    # Still a valid, expert-0-favoring update, just without any prior term.
    assert weights[0] > weights[1]
    assert weights.sum() == 1.0 or abs(weights.sum() - 1.0) < 1e-9
