"""Phase F: the promotion race."""
import numpy as np

from scaata.research import race


def _row(**kw):
    base = {"e_up": 25.0, "e_down": 0.5, "sessions": 300, "drawdown_vs_baseline": 0.0, "promotable": True, "why": "x"}
    return {**base, **kw}


def test_promotion_needs_every_condition():
    assert race.promotion(_row(), True)["verdict"] == "promote"
    for bad in (_row(e_up=5.0), _row(sessions=10), _row(drawdown_vs_baseline=-0.05), _row(promotable=False)):
        assert race.promotion(bad, True)["verdict"] == "keep racing"
    assert race.promotion(_row(), False)["verdict"] == "keep racing"       # poison test failing blocks it
    assert race.promotion(_row(), None)["verdict"] == "keep racing"        # so does not knowing


def test_demotion_is_the_mirror_image():
    assert race.promotion(_row(e_up=0.1, e_down=30.0, promotable=False), True)["verdict"] == "demote"


def test_frozen_settings_are_written_down_not_read_from_live_code():
    assert race.FROZEN == {"act_threshold": 1 / 3, "min_hold_days": 10, "learning_rate": 0.05, "fixed_share": round(1 / 252, 5)}
    assert race.ENTRANTS[0]["params"] == race.FROZEN


def test_sessions_to_threshold():
    rng = np.random.default_rng(0)
    d = rng.normal(0.0005, 0.005, 5000)
    n = race._sessions_to_threshold(d)
    assert n and 100 < n < 2000
    assert race._sessions_to_threshold(-d) is None


def test_exploratory_entries_cannot_be_promoted():
    for e in race.ENTRANTS:
        if e["role"] == "challenger" and e["id"] != "rebound":
            assert e["promotable"] is False
