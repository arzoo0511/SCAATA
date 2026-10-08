from datetime import date

import pandas as pd
import pytest

from scaata.agent import news_history as nh

RSS = b"""<rss><channel>
<item><title>ITC raises cigarette prices - Times of India</title><source>Times of India</source>
<pubDate>Wed, 06 Mar 2019 08:00:00 GMT</pubDate></item>
<item><title>Old story - X</title><source>X</source><pubDate>Wed, 06 Mar 2013 08:00:00 GMT</pubDate></item>
</channel></rss>"""


class FakeHttp:
    def __init__(self):
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        class R:
            content = RSS
            def raise_for_status(self): pass
        return R()


def test_fetch_week_bounds_the_query_and_drops_stray_dates():
    http = FakeHttp()
    items = nh.fetch_week("ITC.NS", date(2019, 3, 4), http)
    assert "after:2019-03-04" in http.urls[0] and "before:2019-03-11" in http.urls[0]
    assert [h["title"] for h in items] == ["ITC raises cigarette prices"]


def test_weeks_are_mondays_and_complete():
    ws = nh.weeks(date(2019, 3, 6), date(2019, 3, 25))
    assert ws == [date(2019, 3, 4), date(2019, 3, 11), date(2019, 3, 18)]


def _h(ts, sentiment):
    return {"title": f"ITC news {ts} {sentiment}", "published_utc": ts, "relevant": True,
            "sentiment": sentiment, "material": 0.9, "scorer": "test"}


def test_news_views_use_only_headlines_known_by_that_evening():
    sessions = pd.DatetimeIndex(["2019-03-05", "2019-03-06", "2019-03-12"])
    scored = [_h("2019-03-06T08:00:00+00:00", 1.0)]          # morning of the 6th
    views = nh.news_views("ITC.NS", sessions, scored, months={"2019-03"})
    assert views.iloc[0] == 0.0                              # not yet published
    assert views.iloc[1] > 0                                 # known by the 6th's evening run
    assert views.iloc[2] == 0.0                              # older than the lookback


def test_news_views_are_missing_for_unscored_months():
    views = nh.news_views("ITC.NS", pd.DatetimeIndex(["2019-04-01"]), [], months={"2019-03"})
    assert views.isna().all()


def test_with_news_fills_the_news_view_and_treats_missing_as_silent():
    from scaata.agent.backtest import with_news

    t1, t2 = pd.Timestamp("2019-03-05"), pd.Timestamp("2019-03-06")
    views = {t: {"A.NS": {"trend": 0.1, "volatility": 0.0, "momentum": 0.2, "news": 0.0}} for t in (t1, t2)}
    news = {"A.NS": pd.Series({t1: 0.5, t2: float("nan")})}
    out = with_news(views, news)
    assert out[t1]["A.NS"]["news"] == 0.5 and out[t2]["A.NS"]["news"] == 0.0
    assert out[t1]["A.NS"]["trend"] == 0.1 and views[t1]["A.NS"]["news"] == 0.0


def test_score_all_scores_months_newest_regime_first_and_stops_at_the_budget(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(nh, "ARCHIVE_DIR", tmp_path)
    monkeypatch.setattr(nh, "LEDGER_PATH", tmp_path / "ledger.json")
    monkeypatch.setattr(nh, "TOKENS_PER_MINUTE_PAUSE", 0)
    monkeypatch.setattr(nh, "weeks", lambda *a, **k: [date(2024, 6, 3), date(2024, 7, 1)])
    for monday, ts in ((date(2024, 6, 3), "2024-06-04T08:00:00+00:00"), (date(2024, 7, 1), "2024-07-02T08:00:00+00:00")):
        (tmp_path / "ITC").mkdir(exist_ok=True)
        (tmp_path / "ITC" / f"{monday}.json").write_text(json.dumps([{"title": "ITC x", "source": "s",
                                                                      "published_utc": ts}]), encoding="utf-8")
    calls = []
    monkeypatch.setattr(nh.news, "llm_scores", lambda h, c, client: calls.append(h[0]["published_utc"]) or
                        [{"relevant": True, "sentiment": 0.5, "material": 0.8, "scorer": "fake"}] * len(h))
    assert nh.score_all(["ITC.NS"], client=object(), budget=1) is False
    assert calls == ["2024-07-02T08:00:00+00:00"]                 # after the model's cutoff, first
    assert nh.score_all(["ITC.NS"], client=object(), budget=5) is True
    assert nh.scored_months("ITC.NS") == {"2024-06", "2024-07"}
