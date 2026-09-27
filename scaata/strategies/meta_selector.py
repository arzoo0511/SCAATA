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
    def __init__(self, input_dim: int, num_strategies: int, hidden_dims: list[int] | None = None, dropout: float = 0.3):
        """`hidden_dims` defaults to a single 128-unit layer (the original
        v1-ported architecture) -- callers with a larger strategy pool to
        discriminate among (Phase 11 grew this from ~9 to 50+ real
        candidates) and a richer feature set to draw on should pass a
        deeper/wider stack, e.g. `[256, 128, 64]`, via
        `scaata.config.META_SELECTOR_HIDDEN_DIMS`.
        """
        super().__init__()
        hidden_dims = hidden_dims or [128]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_strategies))
        self.net = nn.Sequential(*layers)

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


def _accuracy(model: StrategySelector, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).argmax(dim=1).numpy()
    return float((preds == y).mean())


def train_meta_selector(
    states: np.ndarray,
    strategy_signals: list[np.ndarray],
    future_returns: np.ndarray,
    strategy_weights: list[float] | None = None,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    hidden_dims: list[int] | None = None,
    val_frac: float = 0.0,
    class_balanced_loss: bool = False,
) -> tuple[StrategySelector, int] | tuple[StrategySelector, int, dict]:
    """`val_frac > 0` holds out that fraction as a time-based (not random)
    validation split -- the same in-sample-vs-held-out discipline used
    throughout this codebase (Phase 11's evolution fitness gap, the
    walk-forward harness) -- and returns a third `metrics` dict
    (`train_accuracy`, `val_accuracy`) so a retrain can be judged by
    something more concrete than "trust me it's better," instead of the
    2-tuple return every existing caller expects. Default `val_frac=0.0`
    preserves the exact prior return signature/behavior for those callers.

    `class_balanced_loss=True` weights `CrossEntropyLoss` by inverse class
    frequency. Found necessary live: with plain (unweighted) loss and no
    early stopping, this classifier reliably collapses to predicting a
    single majority-class strategy with confidence 1.0 for every single
    input row, on every one of the 9 real tickers tested -- meaning
    `meta_confidence` (fed into the RL policy's observation space in every
    `meta_only`/`full_pipeline` ablation run so far) has been a hardcoded
    constant with zero variance, contributing no real information. More
    training epochs and a deeper network make this WORSE, not better (both
    give the majority class more room to dominate), since neither addresses
    the actual cause: the "which of N strategies wins tomorrow" label is
    heavily imbalanced and only weakly related to today's features, so
    unweighted cross-entropy training happily converges to "always guess
    the prior" as a lower-loss solution than trying to discriminate.
    """
    torch.manual_seed(seed)
    num_strategies = max(len(strategy_signals), 1)
    meta_X, meta_y = build_meta_dataset(states, strategy_signals, future_returns, strategy_weights=strategy_weights)

    if val_frac > 0:
        split_idx = int(len(meta_X) * (1 - val_frac))
        train_X, train_y = meta_X[:split_idx], meta_y[:split_idx]
        val_X, val_y = meta_X[split_idx:], meta_y[split_idx:]
    else:
        train_X, train_y = meta_X, meta_y
        val_X, val_y = None, None

    X_tensor = torch.tensor(train_X, dtype=torch.float32)
    y_tensor = torch.tensor(train_y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = StrategySelector(input_dim=states.shape[1], num_strategies=num_strategies, hidden_dims=hidden_dims)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    class_weight = None
    if class_balanced_loss:
        counts = np.bincount(train_y, minlength=num_strategies).astype(np.float32)
        counts = np.where(counts > 0, counts, 1.0)  # avoid div-by-zero for a class absent from this ticker's training data
        inv_freq = 1.0 / counts
        class_weight = torch.tensor(inv_freq * (num_strategies / inv_freq.sum()), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
        model.train()

    if val_frac <= 0:
        return model, num_strategies

    metrics = {
        "train_accuracy": _accuracy(model, train_X, train_y),
        "val_accuracy": _accuracy(model, val_X, val_y) if len(val_X) > 0 else float("nan"),
        "n_train": len(train_X),
        "n_val": len(val_X),
    }
    return model, num_strategies, metrics


def build_meta_regression_dataset(
    states: np.ndarray,
    strategy_signals: list[np.ndarray],
    future_returns: np.ndarray,
    strategy_weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The regression counterpart to `build_meta_dataset`: instead of
    collapsing each day's per-strategy profit vector down to a single
    best-strategy class label (discarding every other strategy's relative
    performance that day), keeps the full `(n, num_strategies)` continuous
    profit matrix as the target. Same underlying `signal * future_return *
    weight` computation as `build_meta_dataset` -- this is the strictly
    more informative version of the same data, not a different
    computation.
    """
    n = len(states) - 1
    num_strategies = len(strategy_signals)
    meta_X, meta_Y = [], []
    for t in range(n):
        profits = np.array([
            signals[t] * future_returns[t] * (1.0 if strategy_weights is None else strategy_weights[i])
            for i, signals in enumerate(strategy_signals)
        ])
        meta_X.append(states[t])
        meta_Y.append(profits)
    return np.array(meta_X), np.array(meta_Y).reshape(-1, num_strategies)


def train_meta_selector_regression(
    states: np.ndarray,
    strategy_signals: list[np.ndarray],
    future_returns: np.ndarray,
    strategy_weights: list[float] | None = None,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    hidden_dims: list[int] | None = None,
    val_frac: float = 0.0,
) -> tuple[StrategySelector, int] | tuple[StrategySelector, int, dict]:
    """Trains the same `StrategySelector` architecture (raw linear output,
    no softmax) to directly regress each strategy's expected one-step
    profit (MSE loss) instead of classifying a single winner
    (`train_meta_selector`'s CrossEntropyLoss). Motivation: that
    classification version was found to reliably collapse to predicting a
    single constant majority-class strategy with zero input-dependent
    variance, on every real ticker tested (see `train_meta_selector`'s
    docstring) -- a regression target preserves every strategy's relative
    performance each day instead of discarding it into one argmax label,
    which is a fundamentally richer training signal and doesn't have the
    same "always guess the majority class" attractor... in principle. This
    is a genuine, untested-before-now hypothesis, not an assumed fix --
    `metrics` (when `val_frac > 0`) reports `pred_std_across_rows`
    specifically to check whether it actually escapes the collapse (a
    regression model can just as easily collapse to predicting the
    unconditional mean profit vector for every input, the MSE-loss
    equivalent of the same failure mode) rather than assuming richer
    targets fix it.
    """
    torch.manual_seed(seed)
    num_strategies = max(len(strategy_signals), 1)
    meta_X, meta_Y = build_meta_regression_dataset(states, strategy_signals, future_returns, strategy_weights=strategy_weights)

    if val_frac > 0:
        split_idx = int(len(meta_X) * (1 - val_frac))
        train_X, train_Y = meta_X[:split_idx], meta_Y[:split_idx]
        val_X, val_Y = meta_X[split_idx:], meta_Y[split_idx:]
    else:
        train_X, train_Y = meta_X, meta_Y
        val_X, val_Y = None, None

    X_tensor = torch.tensor(train_X, dtype=torch.float32)
    Y_tensor = torch.tensor(train_Y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = StrategySelector(input_dim=states.shape[1], num_strategies=num_strategies, hidden_dims=hidden_dims)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_Y)
            loss.backward()
            optimizer.step()
        model.train()

    if val_frac <= 0:
        return model, num_strategies

    model.eval()
    with torch.no_grad():
        train_preds = model(torch.tensor(train_X, dtype=torch.float32)).numpy()
        val_preds = model(torch.tensor(val_X, dtype=torch.float32)).numpy() if len(val_X) > 0 else np.zeros((0, num_strategies))

    train_rank_acc = float((train_preds.argmax(axis=1) == train_Y.argmax(axis=1)).mean())
    val_rank_acc = float((val_preds.argmax(axis=1) == val_Y.argmax(axis=1)).mean()) if len(val_X) > 0 else float("nan")

    metrics = {
        "train_mse": float(np.mean((train_preds - train_Y) ** 2)),
        "val_mse": float(np.mean((val_preds - val_Y) ** 2)) if len(val_X) > 0 else float("nan"),
        "train_rank_accuracy": train_rank_acc,
        "val_rank_accuracy": val_rank_acc,
        # Does the model's own top pick actually vary row-to-row, or is
        # this the regression version of the same constant-output
        # collapse? Std of the argmax'd predicted-best-strategy index
        # across rows -- 0 means every row gets the same answer.
        "pred_argmax_std_train": float(train_preds.argmax(axis=1).std()),
        "pred_argmax_std_val": float(val_preds.argmax(axis=1).std()) if len(val_X) > 0 else float("nan"),
        "n_train": len(train_X),
        "n_val": len(val_X),
    }
    return model, num_strategies, metrics


def predict_expected_profits(model: StrategySelector, states: np.ndarray) -> np.ndarray:
    """Raw regression output (expected one-step profit per strategy, no
    softmax -- these are real-valued profit estimates, not probabilities)
    from a model trained by `train_meta_selector_regression`."""
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(states, dtype=torch.float32)).numpy()


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
