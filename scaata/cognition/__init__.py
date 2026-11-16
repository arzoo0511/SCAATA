"""Cognitive HANSEI: what the system currently believes, on what evidence,
and how its own predictions have turned out.

This is not a chatbot and it generates no text of its own. It assembles a
single, inspectable picture from records the other parts already keep:

    episodic    the journal: what it saw, believed and did each day
    procedural  trust in each advisor, and the brain's fixed rules
    semantic    measured sensitivities of each stock to the market, crude, rupee
    research    the Lab's experiments and verdicts
    working     today's views, queued orders and material news

Every belief carries a kind (fact, relationship, finding, hypothesis,
assumption), a status that says how far the evidence goes, and the numbers
behind it. Nothing here writes to the live agent's state or changes how it
trades. It reads JSON files only, so it runs on the hosted dashboard.
"""
