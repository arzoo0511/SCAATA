"""MACROVERSE: what happens to HANSEI when the world moves.

Two kinds of run, never mixed up:

    replay    a real historical episode (2008, March 2020, ...), on real
              prices, traded by HANSEI's real brain
    scenario  a synthetic what-if (crude +60%, market -20%, ...), built from
              measured sensitivities plus resampled noise, traded by the same
              real brain on each simulated path

Every output says which kind it is. Macroverse reads the research price
snapshot and writes only under `macroverse/`; it never touches the live
paper book, journal or memory, and cannot change how the agent trades.
"""
