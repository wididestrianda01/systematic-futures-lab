"""Universe map: 16 CME roots → pysystemtrade instrument filenames.

Amended 2026-09-09 (free-only): source is pysystemtrade's raw-leg CSV snapshot
(pinned commit; frozen 2024-03-28). The four dropped roots are recorded with
reasons so the boundary is visible in code, not just prose.
"""

INSTRUMENTS = {
    "ES": "SP500",
    "NQ": "NASDAQ",
    "YM": "DOW",
    "ZN": "US10",
    "ZB": "US30",
    "ZF": "US5",
    "GC": "GOLD",
    "SI": "SILVER",
    "HG": "COPPER",
    "CL": "CRUDE_W",
    "NG": "GAS_US",
    "ZC": "CORN",
    "ZW": "WHEAT",
    "6E": "EUR",
    "6J": "JPY",
    "6B": "GBP",
}

UNIVERSE = INSTRUMENTS  # keys are the canonical roots

DROPPED = {
    "KC": "ICE soft — no free feed (universe trimmed 2026-09-09)",
    "SB": "ICE soft — no free feed (universe trimmed 2026-09-09)",
    "CC": "ICE soft — no free feed (universe trimmed 2026-09-09)",
    "BZ": "Brent legs start 2020-08 — cannot fill the 2010–2019 develop window",
}


def instrument_for(root: str) -> str:
    """Instrument filename stem for a universe root; hard-fails on unknowns and dropped roots."""
    if root not in INSTRUMENTS:
        raise KeyError(f"{root}: not in the 16-root universe (see DROPPED)")
    return INSTRUMENTS[root]
