"""Look-ahead-bias regression test for GDELT sentiment alignment: a record
published after US market close must never end up attributed to that same
trading day's feature — this is the specific leakage risk flagged for the
GDELT integration in the rebuild plan.
"""
import pandas as pd

from scaata.data.gdelt import align_sentiment_to_trading_days, attribute_trading_date


def test_pre_close_record_stays_same_day():
    # 10:00am ET on 2024-01-10 (a Wednesday, no DST edge case)
    ts = pd.Timestamp("2024-01-10 15:00:00")  # UTC-5 in January
    assert attribute_trading_date(ts) == pd.Timestamp("2024-01-10")


def test_post_close_record_rolls_to_next_day():
    # 5:00pm ET on 2024-01-10 -- after the 4:00pm close
    ts = pd.Timestamp("2024-01-10 22:00:00")
    assert attribute_trading_date(ts) == pd.Timestamp("2024-01-11")


def test_exactly_at_close_rolls_forward():
    # 4:00pm ET exactly -- treated as post-close (>=), not same-day
    ts = pd.Timestamp("2024-01-10 21:00:00")
    assert attribute_trading_date(ts) == pd.Timestamp("2024-01-11")


def test_alignment_reaggregates_rolled_records_together():
    df = pd.DataFrame({
        "date": [
            pd.Timestamp("2024-01-10 15:00:00"),  # stays on the 10th
            pd.Timestamp("2024-01-10 22:00:00"),  # rolls to the 11th
            pd.Timestamp("2024-01-11 14:00:00"),  # already on the 11th
        ],
        "ticker": ["AAPL"] * 3,
        "tone": [2.0, -4.0, 0.0],
        "mention_volume": [1, 1, 1],
    })
    aligned = align_sentiment_to_trading_days(df)

    day10 = aligned[aligned["date"] == pd.Timestamp("2024-01-10")]
    day11 = aligned[aligned["date"] == pd.Timestamp("2024-01-11")]

    assert len(day10) == 1 and day10["tone"].iloc[0] == 2.0 and day10["mention_volume"].iloc[0] == 1
    # the rolled 22:00 record and the native 2024-01-11 record must be merged
    assert len(day11) == 1
    assert day11["mention_volume"].iloc[0] == 2
    assert day11["tone"].iloc[0] == -2.0  # mean of -4.0 and 0.0
