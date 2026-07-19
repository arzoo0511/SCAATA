"""Loads BC-pretrained weights into RecurrentPPO's policy — the concrete
fix for the v1 gap where the BC model was trained, saved to disk, and never
loaded back into the RL agent.

Architecture note (why this is a *partial* transfer, not a full one):
`ImitationModel` is a plain feedforward net operating directly on the raw
7-dim feature vector, with BatchNorm between its linear layers.
`RecurrentPPO`'s `MlpLstmPolicy` instead runs raw features through an LSTM
first, has no BatchNorm, and its `policy_net`'s first layer consumes the
LSTM's 256-dim hidden state, not the raw features. There is no
shape-compatible (or architecture-compatible) equivalent for BC's first
layer or its BatchNorm layers in the PPO policy — only the final
"64-dim representation -> 3 action logits" mapping (BC's last Linear
layer) and the preceding 64->64 Linear have an exact shape match with
PPO's `mlp_extractor.policy_net.2` and `action_net`. Those two layers are
what get transferred; the LSTM and the first policy_net layer stay at
SB3's default initialization, since they operate on a different input
space than anything BC ever saw. This is a partial warm start, not a full
weight transplant — verified below by checking the transfer actually took
effect, not by claiming full-network output equivalence (which the
architecture mismatch above makes impossible to claim honestly).
"""
from __future__ import annotations

import torch

from scaata.imitation.model import ImitationModel

# bc_state_dict_key -> policy_state_dict_key, only where shapes match exactly.
BC_TO_POLICY_KEY_MAP = {
    "net.4.weight": "mlp_extractor.policy_net.2.weight",
    "net.4.bias": "mlp_extractor.policy_net.2.bias",
    "net.7.weight": "action_net.weight",
    "net.7.bias": "action_net.bias",
}


def load_bc_weights_into_policy(bc_model: ImitationModel, policy) -> dict[str, bool]:
    """Copies BC's final two layers into the PPO policy's corresponding
    layers in place. Returns {policy_key: transferred_bool} so callers/tests
    can confirm the transfer actually happened rather than silently no-op'ing.
    """
    bc_state = bc_model.state_dict()
    policy_state = policy.state_dict()

    transferred = {}
    for bc_key, policy_key in BC_TO_POLICY_KEY_MAP.items():
        if bc_key not in bc_state or policy_key not in policy_state:
            transferred[policy_key] = False
            continue
        if bc_state[bc_key].shape != policy_state[policy_key].shape:
            transferred[policy_key] = False
            continue
        policy_state[policy_key] = bc_state[bc_key].clone()
        transferred[policy_key] = True

    policy.load_state_dict(policy_state, strict=False)
    return transferred


def verify_transfer(bc_model: ImitationModel, policy, fresh_policy_state_dict: dict) -> dict:
    """Two checks that the transfer actually took effect — this is a
    regression guard against the exact v1 bug (weights trained, saved, and
    silently never loaded):

    1. `weights_match_bc`: the transferred tensors are byte-identical to
       BC's corresponding trained weights.
    2. `differs_from_fresh_init`: at those same keys, the loaded policy's
       weights differ from an untouched, freshly-initialized policy's
       weights — a random init matching BC's trained weights by chance is
       astronomically unlikely, so this confirms the load wasn't a silent
       no-op leaving default initialization in place.
    """
    policy_state = policy.state_dict()
    bc_state = bc_model.state_dict()

    weights_match_bc = all(
        torch.equal(bc_state[bc_key], policy_state[policy_key])
        for bc_key, policy_key in BC_TO_POLICY_KEY_MAP.items()
    )
    differs_from_fresh_init = all(
        not torch.equal(policy_state[policy_key], fresh_policy_state_dict[policy_key])
        for policy_key in BC_TO_POLICY_KEY_MAP.values()
    )

    return {
        "weights_match_bc": weights_match_bc,
        "differs_from_fresh_init": differs_from_fresh_init,
    }
