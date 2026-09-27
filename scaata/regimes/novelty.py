"""Continuous novelty / distributional-shift score (Phase 10) — replaces
the discrete `{calm_bull, volatile_bull, bear, crisis}` labels in
`scaata.regimes.detector` as the thing that modulates cross-signal
weighting (`scaata.strategies.hedge`). Two independent, causal methods
(the same "two independent sources, report agreement" convention already
used for GDELT+FinBERT and HMM+threshold):

1. Rolling Mahalanobis distance of today's feature vector against a
   trailing reference distribution, converted to a chi-square p-value —
   a ranking signal, not a calibrated probability (daily returns are
   fat-tailed, violating the underlying normality assumption).
2. A rolling recent-vs-historical discriminator (a logistic classifier
   refit periodically, distinguishing "recent window" rows from
   "historical window" rows); its predicted P(recent) *is* the novelty
   score, gated by its own held-out AUC since a handful of features over a
   short recent window can easily produce an overfit, uninformative
   classifier.

`threshold_regime_labels` (the discrete buckets) is left untouched for
backward compatibility with existing tests/robustness harness — novelty is
additive, not a replacement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from scaata.config import (
    NOVELTY_AUC_TRUST_THRESHOLD,
    NOVELTY_HISTORICAL_WINDOW,
    NOVELTY_RECENT_WINDOW,
    NOVELTY_REF_WINDOW,
    NOVELTY_REFIT_EVERY_DAYS,
    NOVELTY_RIDGE_EPS,
)


def _mahalanobis_d2(x: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> float:
    diff = x - mean
    return float(diff @ cov_inv @ diff)


def rolling_mahalanobis_d2(
    df: pd.DataFrame,
    feature_columns: list[str],
    ref_window: int = NOVELTY_REF_WINDOW,
    ridge_eps: float = NOVELTY_RIDGE_EPS,
) -> pd.Series:
    """Mahalanobis squared-distance of each row's feature vector against the
    mean/covariance of the trailing `ref_window` days strictly *before* that
    row (row t itself is never included in its own reference distribution).
    NaN until enough trailing history exists. Computed per-`Ticker`.
    """
    out_frames = []
    for _, group in df.groupby("Ticker"):
        group = group.sort_index()
        values = group[feature_columns].values
        n = len(values)
        d2 = np.full(n, np.nan)

        for t in range(n):
            start = t - ref_window
            if start < 0:
                continue
            ref = values[start:t]
            if len(ref) < len(feature_columns) + 1:
                continue
            mean = ref.mean(axis=0)
            cov = np.cov(ref, rowvar=False) + ridge_eps * np.eye(len(feature_columns))
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                continue
            d2[t] = _mahalanobis_d2(values[t], mean, cov_inv)

        out_frames.append(pd.Series(d2, index=group.index, name="mahalanobis_d2"))
    return pd.concat(out_frames).sort_index()


def mahalanobis_to_novelty(d2: pd.Series, dof: int) -> pd.Series:
    """Converts squared-distance to a novelty score in [0, 1]: `1 -
    chi2.sf(d2, dof)`, so 1 means "maximally surprising" (near-zero p-value
    under the reference distribution) and 0 means "perfectly typical".
    """
    return d2.apply(lambda x: 1.0 - chi2.sf(x, df=dof) if pd.notna(x) else np.nan)


def rolling_mahalanobis_novelty(
    df: pd.DataFrame,
    feature_columns: list[str],
    ref_window: int = NOVELTY_REF_WINDOW,
    ridge_eps: float = NOVELTY_RIDGE_EPS,
) -> pd.Series:
    d2 = rolling_mahalanobis_d2(df, feature_columns, ref_window, ridge_eps)
    return mahalanobis_to_novelty(d2, dof=len(feature_columns))


class RollingDiscriminatorNovelty:
    """A logistic classifier distinguishing "recent window" feature rows
    from "historical window" rows. `fit()` also holds out a validation
    split purely to report `auc_` — the trust gate callers should check
    before believing `score()` — then refits on the full historical+recent
    sample for the model actually used to score new rows.
    """

    def __init__(self, recent_window: int = NOVELTY_RECENT_WINDOW, historical_window: int = NOVELTY_HISTORICAL_WINDOW):
        self.recent_window = recent_window
        self.historical_window = historical_window
        self.model_: LogisticRegression | None = None
        self.auc_: float = float("nan")

    def fit(self, historical_X: np.ndarray, recent_X: np.ndarray) -> "RollingDiscriminatorNovelty":
        X = np.vstack([historical_X, recent_X])
        y = np.concatenate([np.zeros(len(historical_X)), np.ones(len(recent_X))])

        if len(X) < 10 or len(np.unique(y)) < 2:
            self.model_, self.auc_ = None, float("nan")
            return self

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.3, random_state=0, stratify=y
            )
            probe = LogisticRegression(max_iter=1000).fit(X_train, y_train)
            preds = probe.predict_proba(X_val)[:, 1]
            self.auc_ = roc_auc_score(y_val, preds) if len(np.unique(y_val)) > 1 else float("nan")
        except ValueError:
            self.auc_ = float("nan")

        self.model_ = LogisticRegression(max_iter=1000).fit(X, y)
        return self

    def score(self, x: np.ndarray) -> float:
        if self.model_ is None:
            return float("nan")
        return float(self.model_.predict_proba(x.reshape(1, -1))[0, 1])

    @property
    def is_trusted(self) -> bool:
        return pd.notna(self.auc_) and self.auc_ >= NOVELTY_AUC_TRUST_THRESHOLD


def rolling_discriminator_novelty(
    df: pd.DataFrame,
    feature_columns: list[str],
    recent_window: int = NOVELTY_RECENT_WINDOW,
    historical_window: int = NOVELTY_HISTORICAL_WINDOW,
    refit_every_days: int = NOVELTY_REFIT_EVERY_DAYS,
) -> pd.DataFrame:
    """Per-`Ticker` causal application of `RollingDiscriminatorNovelty`:
    every `refit_every_days`, refits on the trailing
    `historical_window`+`recent_window` days strictly before the row being
    scored, then scores rows with the frozen model until the next refit —
    the model is never fit on data at or after the row it scores. Returns
    `discriminator_novelty` (P(recent), NaN when untrusted/insufficient
    history) and `discriminator_auc` (the trust-gate value active at that
    row).
    """
    out_frames = []
    for _, group in df.groupby("Ticker"):
        group = group.sort_index()
        values = group[feature_columns].values
        n = len(values)
        scores = np.full(n, np.nan)
        aucs = np.full(n, np.nan)
        detector: RollingDiscriminatorNovelty | None = None

        for t in range(n):
            if detector is None or t % refit_every_days == 0:
                hist_start = t - historical_window - recent_window
                hist_end = t - recent_window
                recent_start = t - recent_window
                if hist_start >= 0 and recent_start >= 0:
                    historical_X = values[hist_start:hist_end]
                    recent_X = values[recent_start:t]
                    if len(historical_X) >= 10 and len(recent_X) >= 5:
                        detector = RollingDiscriminatorNovelty(recent_window, historical_window).fit(
                            historical_X, recent_X
                        )
                    else:
                        detector = None
                else:
                    detector = None

            if detector is not None and detector.model_ is not None and detector.is_trusted:
                scores[t] = detector.score(values[t])
                aucs[t] = detector.auc_

        out_frames.append(pd.DataFrame({"discriminator_novelty": scores, "discriminator_auc": aucs}, index=group.index))
    return pd.concat(out_frames).sort_index()


def combined_novelty_score(mahalanobis: pd.Series, discriminator: pd.Series, method: str = "mean") -> pd.Series:
    """Averages the two independent novelty estimates (row-wise, skipping
    whichever is NaN) — if only one method has enough history/trust at a
    given row, that one is used alone rather than propagating a NaN.
    """
    if method != "mean":
        raise ValueError(f"unknown combination method: {method}")
    combined = pd.concat([mahalanobis.rename("m"), discriminator.rename("d")], axis=1)
    return combined.mean(axis=1, skipna=True)
