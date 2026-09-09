"""Venue map: the 17-symbol CME-only universe.

Amended 2026-09-09: Databento hit a credit-card wall; source is CME Group free EOD.
The ICE softs (KC/SB/CC) are dropped — no free ICE EOD feed; recorded here so the
boundary is visible in code, not just prose.
"""

DATASET = "CME-FREE-EOD"

UNIVERSE = {
    "ES": "S&P 500 E-mini",
    "NQ": "Nasdaq-100 E-mini",
    "YM": "Dow E-mini",
    "ZN": "10Y T-Note",
    "ZB": "30Y T-Bond",
    "ZF": "5Y T-Note",
    "GC": "Gold (COMEX)",
    "SI": "Silver (COMEX)",
    "HG": "Copper (COMEX)",
    "CL": "WTI Crude (NYMEX)",
    "BZ": "Brent Crude (NYMEX)",
    "NG": "Natural Gas (NYMEX)",
    "ZC": "Corn (CBOT)",
    "ZW": "Wheat (CBOT)",
    "6E": "Euro FX",
    "6J": "Japanese Yen",
    "6B": "British Pound",
}

DROPPED = {
    "KC": "ICE soft — no free ICE EOD feed (universe trimmed 2026-09-09)",
    "SB": "ICE soft — no free ICE EOD feed (universe trimmed 2026-09-09)",
    "CC": "ICE soft — no free ICE EOD feed (universe trimmed 2026-09-09)",
}


def dataset_for(symbol: str) -> str:
    """Dataset for a universe root; KeyError on unknown symbols (hard-fail policy)."""
    if symbol not in UNIVERSE:
        raise KeyError(f"{symbol}: not in the 17-symbol CME universe (see DROPPED for pruned softs)")
    return DATASET
