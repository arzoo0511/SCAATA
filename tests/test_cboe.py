"""Tests for the CBOE put/call ratio loader (Phase 9c): parsing must find
the real header row despite CBOE's variable-length descriptive preamble,
fetch failures must degrade to an empty frame (never raise, matching
`scaata.data.gdelt`'s fail-soft discipline), and the lag applied before
merging onto features must actually shift dates forward (an off-by-one here
would leak an unpublished ratio into the day it's meant to describe).
"""
import pandas as pd
import pytest
import requests

from scaata.data import cboe


SAMPLE_CSV = (
    "CBOE Total Put/Call Ratio\n"
    "Some descriptive preamble line\n"
    "Another preamble line\n"
    "DATE,CALL,PUT,TOTAL,P/C Ratio\n"
    "01/02/2024,100,80,180,0.80\n"
    "01/03/2024,100,120,220,1.20\n"
)


def test_parse_putcall_csv_finds_header_past_preamble():
    parsed = cboe._parse_putcall_csv(SAMPLE_CSV)
    assert list(parsed["total_putcall_ratio"]) == [0.80, 1.20]
    assert parsed["date"].iloc[0] == pd.Timestamp("2024-01-02")


def test_parse_putcall_csv_returns_empty_on_unrecognized_layout():
    parsed = cboe._parse_putcall_csv("garbage,not,a,real,header\n1,2,3,4,5\n")
    assert list(parsed.columns) == cboe.EMPTY_COLUMNS
    assert parsed.empty


def test_fetch_returns_empty_frame_on_http_error(monkeypatch):
    class FakeResponse:
        status_code = 500
        text = "server error"

    monkeypatch.setattr(cboe.requests, "get", lambda *a, **k: FakeResponse())
    result = cboe.fetch_total_putcall_ratio("2024-01-01", "2024-01-31", use_cache=False)
    assert list(result.columns) == cboe.EMPTY_COLUMNS
    assert result.empty


def test_fetch_returns_empty_frame_on_network_error(monkeypatch):
    def raise_error(*a, **k):
        raise requests.RequestException("connection refused")

    monkeypatch.setattr(cboe.requests, "get", raise_error)
    result = cboe.fetch_total_putcall_ratio("2024-01-01", "2024-01-31", use_cache=False)
    assert result.empty


def test_attach_putcall_feature_lags_dates_forward():
    putcall_df = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
        "total_putcall_ratio": [0.80, 1.20],
    })
    feature_df = pd.DataFrame(
        {"Ticker": "TEST", "returns": [0.0, 0.0, 0.0]},
        index=pd.DatetimeIndex(
            [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")], name="Date"
        ),
    )

    merged = cboe.attach_putcall_feature(feature_df, putcall_df, lag_days=1)

    # the 01-02 ratio (0.80) only becomes visible on 01-03, one day later
    assert merged.loc[pd.Timestamp("2024-01-03"), "total_putcall_ratio"] == 0.80
    assert merged.loc[pd.Timestamp("2024-01-04"), "total_putcall_ratio"] == 1.20


def test_attach_putcall_feature_neutral_fills_when_empty():
    feature_df = pd.DataFrame(
        {"Ticker": "TEST", "returns": [0.0]}, index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")], name="Date")
    )
    merged = cboe.attach_putcall_feature(feature_df, pd.DataFrame(columns=cboe.EMPTY_COLUMNS), neutral_fill=1.0)
    assert merged["total_putcall_ratio"].iloc[0] == 1.0
