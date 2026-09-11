"""Frozen-panel loading, shaping and measurement — the one place a panel is read, put on
the close grid, and measured.

Both phase scripts (the families read tables, the decision read ML harness) read the same two wide
panels: the back-adjusted continuous closes the seam runs on, and the raw-leg
basis series the carry family and the ML features bind. Hoisted here so the
loader — and its gate — exist once.

The wide panel is a **union calendar**: 16 roots with staggered history starts
and quote holes (CME Sunday rows, holidays), so ~44% of its cells are empty. A
plain `pct_change` on that grid turns every hole into two lost returns — the
hole and the observation after it — and any rolling window crossing a hole
becomes NaN, which silently starves trailing-window signals and the vol overlay.
Every return and every trailing window in this project is therefore measured
**per symbol, over that symbol's own observations**: `consecutive_returns`,
`trailing_return`, `trailing_std`, `forward_return`. NaN means "this symbol has
no such observation", never "missing data to be filled".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from systematic_futures.data.manifest import verify_manifest

DERIVED = Path("data/derived")
MANIFEST = Path("manifests/derived.json")

# The split contract: develop 2010–2019, validate 2020–2021, OOT 2022 → 2024Q1.
DEVELOP_START = pd.Timestamp("2010-01-01")
VALIDATE_END = pd.Timestamp("2021-12-31")
OOT_START = pd.Timestamp("2022-01-01")


def to_wide(frame: pd.DataFrame, value: str = "close") -> pd.DataFrame:
    """Long [date, symbol, ..., value] frame -> wide date x symbol panel, rows sorted."""
    return frame.pivot(index="date", columns="symbol", values=value).sort_index()


def wide_panel(path: Path, value: str) -> pd.DataFrame:
    """Concatenate one derived sub-store's Parquet files into a date x symbol panel."""
    frames = [pd.read_parquet(p) for p in sorted(path.glob("*.parquet"))]
    return to_wide(pd.concat(frames, ignore_index=True), value)


def monthly_to_daily(monthly: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """Reindex a monthly-grid panel (index = calendar periods) onto the close panel's days.

    The same-calendar-month score is the lab's one monthly-grid object; every
    consumer maps it onto the days it applies to here, so which month a day
    belongs to — and that a month without a score stays NaN instead of being
    carried forward — is stated once.
    """
    daily = monthly.reindex(closes.index.to_period("M"))
    daily.index = closes.index
    return daily


def load_frozen(
    derived: Path = DERIVED, manifest: Path = MANIFEST
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(wide continuous closes, wide basis) — manifest verified before any read."""
    verify_manifest(derived, manifest)
    return wide_panel(derived / "continuous", "close"), wide_panel(derived / "basis", "basis")


def develop_validate(panel: pd.DataFrame) -> pd.DataFrame:
    """Slice a panel to the develop+validate window — both ends, asserted.

    The derived panel starts in the 1970s, so an end-bounded slice alone would
    quietly run every metric on pre-develop history. Asserts that the panel
    reaches past both bounds, so a truncated loader cannot pass as untouched.
    """
    if panel.index.min() >= DEVELOP_START:
        raise ValueError(f"panel starts {panel.index.min().date()}, at/after develop start")
    if panel.index.max() <= VALIDATE_END:
        raise ValueError(f"panel ends {panel.index.max().date()}, the OOT window must stay intact")
    return panel.loc[DEVELOP_START:VALIDATE_END]


def oot_window(panel: pd.DataFrame) -> pd.DataFrame:
    """Slice a panel to the single out-of-sample window — the frozen snapshot's end is its bound.

    The mirror of `develop_validate`: that one refuses a panel whose OOT tail is gone, this one
    refuses a panel with no tail to read. Stated here once, so no phase script types the
    boundary itself and two scripts cannot disagree about where 2022 starts.
    """
    if panel.index.max() <= VALIDATE_END:
        raise ValueError(f"panel ends {panel.index.max().date()}, there is no OOT window to read")
    return panel.loc[OOT_START:]


def _per_symbol(panel: pd.DataFrame, measure) -> pd.DataFrame:
    """Apply `measure` to each symbol's own dense series (holes dropped), back to wide.

    The loop is deliberate: a `stack`/`groupby`/`rolling` chain on the union grid
    either keeps the NaN holes (pandas' new `stack` does not drop them) or breaks
    windows that cross one, and pandas' grouped-rolling index shape is not stable
    across versions. Sixteen columns are cheap; being right is not optional here.
    """
    return pd.DataFrame(
        {symbol: measure(panel[symbol].dropna()) for symbol in panel.columns}
    ).reindex(index=panel.index, columns=panel.columns)


def consecutive_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol daily returns, each measured over that symbol's own prior observation."""
    return _per_symbol(closes, lambda s: s.pct_change())


def trailing_return(closes: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Per-symbol return over the symbol's own previous `periods` observations."""
    return _per_symbol(closes, lambda s: s.pct_change(periods))


def trailing_std(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-symbol rolling standard deviation (ddof=1) over the symbol's own observations."""
    return _per_symbol(returns, lambda s: s.rolling(window, min_periods=window).std(ddof=1))


def trailing_mean(closes: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-symbol rolling mean of the close over the symbol's own observations."""
    return _per_symbol(closes, lambda s: s.rolling(window, min_periods=window).mean())


def forward_return(closes: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Per-symbol return over the symbol's own next `periods` observations.

    The only forward-looking measurement in the project — used to build training
    labels, never a feature.
    """
    return _per_symbol(closes, lambda s: s.shift(-periods) / s - 1.0)


def held_positions(exposure: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """The position each symbol actually holds each session.

    An exposure is a decision formed on the sessions a symbol quotes; a session it
    does not quote is missing information, not a decision to go flat. The held
    position is therefore the symbol's last exposure formed on one of its own
    quoted sessions, carried across holes (and flat before its first quote). That
    is what makes the return spanning a hole accrue to the held position, and what
    keeps a data hole from showing up as phantom turnover. A *quoted* session with
    a flat exposure is a decision, and is held as one.
    """
    quoted = closes.notna()
    return exposure.where(quoted).ffill().where(quoted.cummax(), 0.0).fillna(0.0)
