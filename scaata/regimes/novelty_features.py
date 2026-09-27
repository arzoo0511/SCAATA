"""Merges the Phase 10 continuous novelty score into the technical feature
set so it can actually reach the RL policy / meta-selector, the same gap
`scaata.features.sentiment_features.merge_sentiment_into_features` closed
for sentiment: `novelty_score` was fully computed (Mahalanobis distance +
a rolling discriminator, `scaata.regimes.novelty`) but only ever consumed
ad hoc, inline, inside `critique_node` to modulate Hedge's learning rate --
never fed to the RL observation space, so the trading policy itself was
always blind to it despite the mechanism existing since Phase 10.
"""
from __future__ import annotations

import pandas as pd

from scaata.regimes.novelty import (
    combined_novelty_score,
    rolling_discriminator_novelty,
    rolling_mahalanobis_novelty,
)


def compute_novelty_column(df: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    """Per-`Ticker` combined novelty score (mean of the Mahalanobis and
    trust-gated discriminator estimates), computed the same way
    `critique_node`'s ad hoc `_current_novelty_score` does it, just as a
    full aligned column instead of a single latest-row scalar. Both
    underlying methods are already strictly causal (trailing-window only),
    so no additional lookahead care is needed here beyond reusing them.
    """
    available_columns = [c for c in feature_columns if c in df.columns]
    if not available_columns:
        return pd.Series(float("nan"), index=df.index)

    mahalanobis = rolling_mahalanobis_novelty(df, available_columns)
    discriminator = rolling_discriminator_novelty(df, available_columns)["discriminator_novelty"]
    return combined_novelty_score(mahalanobis, discriminator)


def merge_novelty_into_features(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    novelty_col: str = "novelty_score",
    neutral_fill: float = 0.0,
) -> pd.DataFrame:
    """Attaches `novelty_col` to `feature_df` (DatetimeIndex, `Ticker`
    column). Rows with insufficient trailing history for either novelty
    method (early in a ticker's series) get `neutral_fill` (0.0, "not
    surprising") rather than being dropped -- same convention as
    `merge_sentiment_into_features`, so a short warmup period doesn't
    shrink the usable dataset.

    Idempotent: an existing `novelty_col` column is replaced, not
    suffixed, matching `merge_sentiment_into_features`'s handling of the
    same re-merge scenario.
    """
    out = feature_df.copy()
    if novelty_col in out.columns:
        out = out.drop(columns=[novelty_col])

    novelty = compute_novelty_column(out, feature_columns)
    out[novelty_col] = novelty.reindex(out.index).fillna(neutral_fill)
    return out
