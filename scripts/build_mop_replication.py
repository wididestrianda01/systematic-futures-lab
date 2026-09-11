"""Run the MOP (2012) replication: published-statistics check, then the factor.

Two checks, both written to `results/replication/`:

1. `table1_vol.csv` — our annualised mean/volatility per instrument against MOP
   Table 1's published values, for each construction (ratio back-adjusted and the
   paper's own front-contract splice). This is the external check on the data
   layer: if the splice were wrong, per-instrument volatility would not line up.
2. `tsmom.csv` — the MOP Eq. (5) factor, per asset class and diversified, over
   the paper's window and over the free data's own 2010-2024 window, gross and at
   the lab's 2 bps, next to the published class means and the published
   post-2009 trend figure.

Run: uv run python scripts/build_mop_replication.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from systematic_futures.data.continuous import back_adjust
from systematic_futures.data.manifest import verify_manifest
from systematic_futures.data.panels import to_wide
from systematic_futures.data.roll_calendar import build_roll_calendar, extract_contract_prices
from systematic_futures.engine.metrics import sharpe
from systematic_futures.replication import (
    AVAILABLE,
    PUBLISHED_MONTHLY,
    TABLE1,
    factor_returns,
    live_instruments,
    naive_splice,
)
from systematic_futures.replication.universe import HOP_2010_2016, MOP_WINDOW

REPL = Path("data/replication")
OUT = Path("results/replication")
MANIFEST = Path("manifests/replication.json")
WINDOWS = [MOP_WINDOW, HOP_2010_2016[:2], ("2010-01-01", "2024-03-31")]
CONSTRUCTIONS = ("adjusted", "naive")


def load_panels() -> dict[str, pd.DataFrame]:
    """Both constructions as wide close panels, one column per MOP Table 1 label."""
    verify_manifest(REPL, MANIFEST)
    out: dict[str, list[pd.Series]] = {c: [] for c in CONSTRUCTIONS}
    for stem, instrument in sorted(AVAILABLE.items()):
        mp = pd.read_csv(REPL / "multiple_prices" / f"{stem}.csv")
        mp["DATETIME"] = pd.to_datetime(mp["DATETIME"])
        mp["symbol"] = stem
        raw = extract_contract_prices(mp)
        adjusted = to_wide(back_adjust(raw, build_roll_calendar(mp))).iloc[:, 0]
        out["adjusted"].append(adjusted.rename(instrument.name))
        out["naive"].append(naive_splice(mp).rename(instrument.name))
    return {c: pd.concat(series, axis=1, sort=False).sort_index() for c, series in out.items()}


def annualised_vol(series: pd.Series) -> tuple[float, float]:
    """(mean, volatility) in annualised percent, from monthly compounded returns."""
    rets = series.pct_change().dropna()
    monthly = (1.0 + rets).groupby(rets.index.to_period("M")).prod() - 1.0
    return float(monthly.mean() * 12 * 100), float(monthly.std(ddof=1) * np.sqrt(12) * 100)


def table1_check(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for instrument in TABLE1:
        if instrument.stem is None:
            continue
        row = {
            "instrument": instrument.name,
            "asset": instrument.asset,
            "stem": instrument.stem,
            "published_start": instrument.start,
            "published_mean": instrument.mean,
            "published_vol": instrument.vol,
        }
        for construction, panel in panels.items():
            series = panel[instrument.name].dropna()
            if len(series) < 250:
                row[f"first_{construction}"] = ""
                row[f"mean_{construction}"] = np.nan
                row[f"vol_{construction}"] = np.nan
                row[f"vol_ratio_{construction}"] = np.nan
                continue
            mean, vol = annualised_vol(series)
            row[f"first_{construction}"] = str(series.index[0].date())
            row[f"mean_{construction}"] = round(mean, 2)
            row[f"vol_{construction}"] = round(vol, 2)
            row[f"vol_ratio_{construction}"] = round(vol / instrument.vol, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def factor_rows(closes: pd.DataFrame, construction: str) -> list[dict]:
    """MOP Eq. (5) over every window x asset class, gross and at 2 bps."""
    rows = []
    for start, end in WINDOWS:
        window = slice(start, end)
        for asset in ("ALL", "COM", "EQ", "FI", "FX"):
            columns = [i.name for i in TABLE1 if i.asset == asset] if asset != "ALL" else None
            if columns is not None:
                columns = [c for c in columns if c in closes.columns]
                if not columns:
                    continue
            book = factor_returns(closes, columns, bps=0.0).loc[window].dropna()
            if len(book) < 250:
                continue
            monthly = (1.0 + book).groupby(book.index.to_period("M")).prod() - 1.0
            net = factor_returns(closes, columns, bps=2.0).loc[window].dropna()
            live = live_instruments(closes, columns).loc[window]
            rows.append(
                {
                    "window": f"{start}..{end}",
                    "construction": construction,
                    "asset": asset,
                    "n_instruments": int(live.max()),
                    "mean_monthly_pct": round(float(monthly.mean() * 100), 3),
                    "vol_annual_pct": round(float(monthly.std(ddof=1) * np.sqrt(12) * 100), 2),
                    "sharpe_gross": round(sharpe(book), 2),
                    "sharpe_2bps": round(sharpe(net), 2),
                    "published_monthly_pct": PUBLISHED_MONTHLY.get(asset, np.nan)
                    if (start, end) == MOP_WINDOW
                    else np.nan,
                }
            )
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panels = load_panels()

    published = table1_check(panels)
    published.to_csv(OUT / "table1_vol.csv", index=False)

    factor = pd.DataFrame([row for c, panel in panels.items() for row in factor_rows(panel, c)])
    factor.to_csv(OUT / "tsmom.csv", index=False)

    covered = published.dropna(subset=["vol_ratio_adjusted"])
    print(f"Table 1 coverage: {len(covered)}/{len(published)} instruments with a free series")
    print(
        "vol ratio ours/published (adjusted): median "
        f"{covered.vol_ratio_adjusted.median():.2f}, range "
        f"{covered.vol_ratio_adjusted.min():.2f}-{covered.vol_ratio_adjusted.max():.2f}"
    )
    print(
        "vol ratio ours/published (naive splice): median "
        f"{covered.vol_ratio_naive.median():.2f}, range "
        f"{covered.vol_ratio_naive.min():.2f}-{covered.vol_ratio_naive.max():.2f}"
    )
    for asset in ("ALL", "COM", "EQ", "FI", "FX"):
        sel = factor[(factor.asset == asset) & (factor.window == "1985-01-01..2009-12-31")]
        if sel.empty:
            continue
        for _, r in sel.iterrows():
            pub = (
                f" published {r.published_monthly_pct:.2f}%/mo"
                if pd.notna(r.published_monthly_pct)
                else ""
            )
            print(
                f"  {asset:3s} {r.construction:8s} n={r.n_instruments:2d} "
                f"mean {r.mean_monthly_pct:+.3f}%/mo vol {r.vol_annual_pct:5.2f}% "
                f"SR {r.sharpe_gross:+.2f} gross / {r.sharpe_2bps:+.2f} at 2bps{pub}"
            )
    for _, r in factor[
        (factor.window == "2010-01-01..2016-12-31") & (factor.asset == "ALL")
    ].iterrows():
        print(
            f"  2010-2016 ALL {r.construction:8s} n={r.n_instruments}: SR {r.sharpe_gross:+.2f} gross / {r.sharpe_2bps:+.2f} at 2bps (HOP publishes 0.41 net)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
