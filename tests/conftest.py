"""Shared test helpers — panel construction pinned in one place."""

from systematic_futures.data.continuous import back_adjust
from systematic_futures.data.roll_calendar import build_roll_calendar, extract_contract_prices
from systematic_futures.data.synthetic import make_synthetic_multiple_prices
from systematic_futures.engine import to_wide


def continuous_wide(symbols=("ES", "GC"), seed=42, start="2020-01-01", end="2021-06-30"):
    """Wide continuous-close panel built from the seeded synthetic fixture."""
    mp = make_synthetic_multiple_prices(symbols, start=start, end=end, seed=seed)
    return to_wide(back_adjust(extract_contract_prices(mp), build_roll_calendar(mp)))
