"""Behavior-cloning training, ported from the v1 notebook.

Builds (state, action) pairs from every strategy in the pool's signals and
trains the `ImitationModel` classifier via cross-entropy, exactly as in v1.
The important difference from v1 is downstream: `rl/policy_init.py`
actually loads these weights into the RL policy afterward.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scaata.imitation.model import ImitationModel

ACTION_MAP = {0: 0, 1: 1, -1: 2}  # hold, buy, sell -> class indices 0,1,2


def build_imitation_dataset(
    states: np.ndarray,
    strategy_signals: list[np.ndarray],
    strategy_weights: list[float] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """`strategy_weights` (Phase 2's self-critique down-weighting mechanism):
    each strategy's rows are subsampled to `round(weight * n_states)` before
    being added to the training set, so a down-weighted strategy has
    proportionally less influence on the trained BC classifier without
    being hard-removed from the pool (matches
    `scaata.strategies.pool`/critique's floor-not-delete design).
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, signals in enumerate(strategy_signals):
        weight = 1.0 if strategy_weights is None else strategy_weights[i]
        n_rows = len(states)
        n_sample = max(1, int(round(weight * n_rows)))
        idx = rng.choice(n_rows, size=n_sample, replace=False) if n_sample < n_rows else np.arange(n_rows)
        for j in idx:
            X.append(states[j])
            y.append(ACTION_MAP[int(signals[j])])
    return np.array(X), np.array(y)


def train_imitation_model(
    states: np.ndarray,
    strategy_signals: list[np.ndarray],
    strategy_weights: list[float] | None = None,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> ImitationModel:
    torch.manual_seed(seed)
    X, y = build_imitation_dataset(states, strategy_signals, strategy_weights=strategy_weights, seed=seed)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ImitationModel(input_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    return model


def predict_action_probs(model: ImitationModel, states: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(states, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()
