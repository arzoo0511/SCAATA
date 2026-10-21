import numpy as np
import pandas as pd
import pytest

from scaata.agent import earnings as E


def test_surprise_score_is_bounded_and_signed():
    assert E.surprise_score({"estimate": 10.0, "reported": 11.0}) == pytest.approx(np.tanh(1.0))
    assert E.surprise_score({"estimate": 10.0, "reported": 9.0}) < 0
    assert -1 <= E.surprise_score({"estimate": 0.0001, "reported": 5.0}) <= 1


def test_a_surprise_is_usable_only_from_the_session_after_its_ist_date():
    sessions = pd.bdate_range("2024-01-15", periods=5)                      # Mon..Fri
    # 20:00 ET on Monday = 06:30 IST Tuesday -> usable from Wednesday's close
    assert E.first_usable_session("2024-01-16T01:00:00+00:00", sessions) == pd.Timestamp("2024-01-17")
    # 05:00 ET Tuesday = 15:30 IST Tuesday -> still Wednesday
    assert E.first_usable_session("2024-01-16T10:00:00+00:00", sessions) == pd.Timestamp("2024-01-17")
    assert E.first_usable_session("2024-01-19T10:00:00+00:00", sessions) is None


def test_results_views_fade_out_over_the_hold_period(monkeypatch):
    closes = pd.DataFrame({"A.NS": np.ones(10)}, index=pd.bdate_range("2024-01-01", periods=10))
    events = pd.DataFrame([{"symbol": "A.NS", "usable": closes.index[2], "score": 0.8}])
    v = E.results_views(closes, events, hold=4)["A.NS"]
    assert v.iloc[1] == 0.0 and v.iloc[2] == pytest.approx(0.8)
    assert v.iloc[3] == pytest.approx(0.6) and v.iloc[6] == 0.0
