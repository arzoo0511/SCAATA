"""What kind of event a headline is about.

The news scorer says whether a story is good or bad for a company. This
says what the story *is*: results, a rate move, crude prices, a regulator,
a deal. With that, later research can ask "what did this stock do after
this kind of news?" instead of treating every headline alike.

It is a fixed list of words, on purpose: it costs nothing, it is the same
every time, it can be applied to every headline already on file, and it
cannot change how any headline is scored or how the agent trades. The tag is
a label on the record and nothing reads it when deciding.

The first matching rule wins, so order matters: a results story that
mentions crude is a results story. Words that appear in a company's own
name ("LNG" in Petronet LNG) are not used alone, or every headline about
that company would get the label.
"""
from __future__ import annotations

import re

EVENTS = {
    "results": ["q1", "q2", "q3", "q4", "quarter", "quarterly", "results", "net profit", "profit", "earnings",
                "revenue", "nii", "pat", "margin", "margins", "guidance"],
    "crude": ["crude", "brent", "oil price", "oil prices", "opec", "lng price", "lng prices", "lng import",
              "lng imports", "spot lng", "gas price", "gas prices", "petrol",
              "diesel", "fuel price", "fuel prices", "lpg", "refinery", "refining", "atf"],
    "rates": ["repo", "rate hike", "rate cut", "rate hikes", "rate cuts", "interest rate", "interest rates",
              "monetary policy", "mclr", "lending rate", "lending rates", "deposit rate", "deposit rates",
              "fd rate", "fd rates", "emi", "emis", "yield", "yields"],
    "legal": ["fraud", "lawsuit", "court", "arrest", "arrested", "scam", "cheating", "fir", "tribunal", "nclt"],
    "management": ["ceo", "md", "chairman", "appoints", "appointed", "appointment", "reappoints", "reappointment",
                   "re-appointment", "resigns", "resignation", "director", "successor", "steps down"],
    "deal": ["acquire", "acquires", "acquisition", "merger", "stake", "deal", "partnership", "partners",
             "joint venture", "jv", "mou", "contract", "order", "pact", "ties up", "tie-up", "agreement", "bags",
             "demerge", "demerger", "demerges", "block deal"],
    "rating": ["upgrade", "upgrades", "downgrade", "downgrades", "target price", "rating", "brokerage",
               "brokerages", "outperform", "underperform", "buy call", "overweight", "underweight"],
    "payout": ["dividend", "buyback", "bonus", "stock split", "qip", "ncd", "ncds", "bonds", "fundraise",
               "fund raise", "raises rs", "raise rs"],
    "regulation": ["rbi", "sebi", "irdai", "penalty", "fine", "fined", "ban", "regulator", "regulatory", "gst",
                   "tax notice", "tax demand", "probe", "cbi", "notice", "compliance", "government", "ministry",
                   "policy", "budget", "tariff", "cess", "duty"],
    "business": ["launch", "launches", "expansion", "expands", "capex", "plant", "project", "opens", "invest",
                 "investment", "capacity", "pipeline", "terminal", "branch", "branches"],
}
LABELS = {"results": "Results", "crude": "Crude and fuel", "rates": "Interest rates", "legal": "Legal trouble",
          "management": "Management", "deal": "Deals and orders", "rating": "Analyst ratings",
          "payout": "Payouts and fundraising", "regulation": "Regulation and policy", "business": "Business moves",
          "other": "Other"}
_PATTERNS = {event: re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(w) for w in words) + r")(?![a-z0-9])")
             for event, words in EVENTS.items()}


def event_type(title: str) -> str:
    """The first event whose words appear in the headline, as whole words; "other" if none do."""
    text = (title or "").lower()
    for event, pattern in _PATTERNS.items():
        if pattern.search(text):
            return event
    return "other"
