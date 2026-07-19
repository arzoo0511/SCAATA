"""Human-readable macro-event annotations for charts only.

These are a hardcoded lookup, not inferred from data, so there is no
look-ahead-bias concern here in the modeling sense — but to keep that
guarantee real, this module must never be imported from `scaata.features`
or `scaata.rl`. It exists purely for `scaata.evaluation` chart titles/labels
and the Phase 3 "3rd eye" 2020-era vs 2026-era comparison.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MacroEvent:
    name: str
    start: str
    end: str


MACRO_EVENTS: list[MacroEvent] = [
    MacroEvent("COVID Crash", "2020-02-19", "2020-03-23"),
    MacroEvent("ZIRP / Meme-Stock Bull", "2020-03-24", "2021-12-31"),
    MacroEvent("Rate-Hike Bear Market", "2022-01-01", "2022-10-31"),
    MacroEvent("Recovery", "2022-11-01", "2022-12-31"),
    MacroEvent("AI Rally", "2023-01-01", "2024-12-31"),
    MacroEvent("2025-2026", "2025-01-01", "2026-12-31"),
]


def label_for_date(date: pd.Timestamp) -> str:
    for event in MACRO_EVENTS:
        if pd.Timestamp(event.start) <= date <= pd.Timestamp(event.end):
            return event.name
    return "Unlabeled"


def annotate(dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series([label_for_date(d) for d in dates], index=dates, name="macro_event")
