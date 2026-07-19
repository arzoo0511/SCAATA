"""Meta-strategy-selector classifier, ported from the v1 notebook.

Predicts, per state, which pooled strategy would have realized the best
one-step-ahead return. In v1 this was trained and then never used again —
Phase 2 actually feeds its output into `strategy_weights` (shared agent
state) and from there into the RL env's observation (see
`scaata/rl/env.py`'s `meta_weights` feature).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class StrategySelector(nn.Module):
    def __init__(self, input_dim: int, num_strategies: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_strategies),
        )

    def forward(self, x):
        return self.net(x)


def build_meta_dataset(
    states: np.ndarray,
    strategy_signals: list[np.ndarray],
    future_returns: np.ndarray,
    strategy_weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Target at each step t is the strategy whose signal * next-step return
    was highest — i.e. the strategy that would have been most profitable to
    follow at that state, ported from v1 with one addition: `profit` is
    scaled by the strategy's current `strategy_weights` entry (Phase 2's
    self-critique down-weighting), so a strategy the critique loop has
    penalized is less likely to be selected as "best" even if its raw
    profit was highest, without ever being fully excluded (floor, not delete).
    """
    meta_X, meta_y = [], []
    n = len(states) - 1
    for t in range(n):
        best_strategy_id, best_ret = None, -np.inf
        for i, signals in enumerate(strategy_signals):
            weight = 1.0 if strategy_weights is None else strategy_weights[i]
            profit = signals[t] * future_returns[t] * weight
            if profit > best_ret:
                best_ret = profit
                best_strategy_id = i
        if best_strategy_id is not None:
            meta_X.append(states[t])
            meta_y.append(best_strategy_id)
    return np.array(meta_X), np.array(meta_y)


def train_meta_selector(
    states: np.ndarray,
    strategy_signals: list[np.ndarray],
    future_returns: np.ndarray,
    strategy_weights: list[float] | None = None,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[StrategySelector, int]:
    torch.manual_seed(seed)
    num_strategies = max(len(strategy_signals), 1)
    meta_X, meta_y = build_meta_dataset(states, strategy_signals, future_returns, strategy_weights=strategy_weights)

    X_tensor = torch.tensor(meta_X, dtype=torch.float32)
    y_tensor = torch.tensor(meta_y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = StrategySelector(input_dim=states.shape[1], num_strategies=num_strategies)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    return model, num_strategies


def predict_strategy_weights(model: StrategySelector, states: np.ndarray) -> np.ndarray:
    """Returns a (n_states, num_strategies) softmax probability matrix —
    this is what feeds `strategy_weights` in the shared agent state, and
    (as a scalar top-confidence summary) the RL env's observation."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(states, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()


def attach_meta_confidence(df, meta_model: StrategySelector, feature_columns: list[str]):
    """Adds a `meta_confidence` column (the meta-selector's top-strategy
    softmax probability per row) to `df`. This is the concrete fix for gap
    3 — the RL env needs no structural change, since `RobustTradingEnv`
    already generalizes to `len(feature_columns)`; callers just include
    `scaata.config.META_CONFIDENCE_COLUMN` in the feature_columns list.
    """
    from scaata.config import META_CONFIDENCE_COLUMN

    states = df[feature_columns].values
    weights = predict_strategy_weights(meta_model, states)
    out = df.copy()
    out[META_CONFIDENCE_COLUMN] = weights.max(axis=1)
    return out


def attach_meta_implied_action(df, meta_model: StrategySelector, feature_columns: list[str], strategy_signals: list[np.ndarray], column: str = "meta_implied_action"):
    """Adds a column with the *action* (0=hold/1=buy/2=sell, matching
    `scaata.rl.env`'s action space) implied by the meta-selector's
    top-weighted strategy at each row — used for the optional soft
    reward-shaping bonus in `RobustTradingEnv` (rewards PPO for agreeing
    with the meta-selector), kept separate from `meta_confidence` since
    this is an action label, not an observation feature.
    """
    states = df[feature_columns].values
    weights = predict_strategy_weights(meta_model, states)
    top_strategy_idx = weights.argmax(axis=1)

    signal_to_action = {-1: 2, 0: 0, 1: 1}
    implied_actions = np.zeros(len(df), dtype=int)
    for i, strat_idx in enumerate(top_strategy_idx):
        if strat_idx < len(strategy_signals):
            implied_actions[i] = signal_to_action[int(strategy_signals[strat_idx][i])]

    out = df.copy()
    out[column] = implied_actions
    return out
