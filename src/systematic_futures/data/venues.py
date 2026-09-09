"""Venue map: the 20-symbol universe → dataset. Softs are ICE, everything else CME."""

# Dataset codes: GLBX.MDP3 is canonical; IFUS.ICE confirmed live at eval time (ticket 07).
DATASETS = {"CME": "GLBX.MDP3", "ICE-US": "IFUS.ICE"}

UNIVERSE = {
    "ES": ("CME", "S&P 500 E-mini"),
    "NQ": ("CME", "Nasdaq-100 E-mini"),
    "YM": ("CME", "Dow E-mini"),
    "ZN": ("CME", "10Y T-Note"),
    "ZB": ("CME", "30Y T-Bond"),
    "ZF": ("CME", "5Y T-Note"),
    "GC": ("CME", "Gold (COMEX)"),
    "SI": ("CME", "Silver (COMEX)"),
    "HG": ("CME", "Copper (COMEX)"),
    "CL": ("CME", "WTI Crude (NYMEX)"),
    "BZ": ("CME", "Brent Crude (NYMEX)"),
    "NG": ("CME", "Natural Gas (NYMEX)"),
    "ZC": ("CME", "Corn (CBOT)"),
    "ZW": ("CME", "Wheat (CBOT)"),
    "6E": ("CME", "Euro FX"),
    "6J": ("CME", "Japanese Yen"),
    "6B": ("CME", "British Pound"),
    "KC": ("ICE-US", "Coffee C"),
    "SB": ("ICE-US", "Sugar No. 11"),
    "CC": ("ICE-US", "Cocoa"),
}


def dataset_for(symbol: str) -> str:
    """Dataset code for a universe root; KeyError on unknown symbols (hard-fail policy)."""
    return DATASETS[UNIVERSE[symbol][0]]


def roots_for(dataset: str) -> tuple[str, ...]:
    return tuple(s for s, (venue, _) in UNIVERSE.items() if DATASETS[venue] == dataset)
