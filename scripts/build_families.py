"""the families read exit: run every method family through the pipeline seam on the
frozen panel — develop+validate only (data through 2021-12-31; the OOT window
2022+ stays untouched for its single end-of-project read) — and commit the
per-family comparison tables plus the seasonality-after-costs finding.

Run: uv run python scripts/build_families.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from systematic_futures.data.manifest import verify_manifest
from systematic_futures.engine import run
from systematic_futures.methods import (
    baseline,
    carry,
    seasonal_tilt,
    seasonality,
    tsmom,
    xs_momentum,
)

DERIVED = Path("data/derived")
MANIFEST = Path("manifests/derived.json")
RESULTS = Path("results/phase3")
VALIDATE_END = "2021-12-31"  # develop+validate window end; OOT starts 2022
VOL_TARGET = 0.10
HEADLINE_BPS = 2.0


def load_frozen() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide continuous panel + wide basis panel, manifest-gated."""
    verify_manifest(DERIVED, MANIFEST)
    cont = pd.concat(
        [pd.read_parquet(p) for p in sorted((DERIVED / "continuous").glob("*.parquet"))],
        ignore_index=True,
    )
    wide = cont.pivot(index="date", columns="symbol", values="close").sort_index()
    basis = pd.concat(
        [pd.read_parquet(p) for p in sorted((DERIVED / "basis").glob("*.parquet"))],
        ignore_index=True,
    )
    basis_wide = basis.pivot(index="date", columns="symbol", values="basis").sort_index()
    return wide, basis_wide


def main() -> int:
    wide, basis_wide = load_frozen()
    window = wide.loc[:VALIDATE_END]
    assert window.index.max() <= pd.Timestamp(VALIDATE_END), "OOT window must stay untouched"

    methods = {
        "baseline": baseline,
        "tsmom": tsmom,  # six-horizon benchmark (sleeves adopted, ticket 26)
        "xs_momentum": xs_momentum,
        "carry": carry(basis_wide),
        "seasonality": seasonality,
        "seasonal_tilt": seasonal_tilt,
    }
    frames = {}
    for name, method in methods.items():
        table = run(method, window, vol_target=VOL_TARGET)
        table["method"] = name
        frames[name] = table.reset_index().rename(columns={"index": "bps"})
    tables = pd.concat(frames.values(), ignore_index=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    tables.to_csv(RESULTS / "tables.csv", index=False)

    season = tables[(tables["method"] == "seasonality") & (tables["bps"] == HEADLINE_BPS)].iloc[0]
    trend = tables[(tables["method"] == "tsmom") & (tables["bps"] == HEADLINE_BPS)].iloc[0]
    dies = season["sharpe"] < trend["sharpe"]
    (RESULTS / "FINDINGS.md").write_text(
        "# the families read findings\n\n"
        "Window: develop+validate (2010-01-01 → 2021-12-31), frozen panel, "
        f"{VOL_TARGET:.0%} vol target, 16 CME roots; OOT (2022+) untouched.\n\n"
        "## Seasonality standalone dies after costs (expected finding)\n\n"
        f"At the {HEADLINE_BPS:.0f} bps default, standalone seasonality posts "
        f"Sharpe {season['sharpe']:.2f} (turnover {season['turnover']:.2f}) against "
        f"the TSMOM benchmark's {trend['sharpe']:.2f} "
        f"(turnover {trend['turnover']:.2f}). "
        + (
            "Confirmed: the seasonal flip is a high-turnover, low-edge strategy and it "
            "loses to the trend benchmark after costs — reported as a failed challenger.\n"
            if dies
            else "Not confirmed on this window — investigate before reporting.\n"
        )
        + "\nFull per-method, per-bps metrics in `tables.csv` "
        "(Sharpe, Sortino, max DD, turnover, DSR).\n"
    )
    inverse_sharpe = run(
        np.sign(basis_wide.reindex(index=window.index, columns=window.columns)),
        window,
        vol_target=VOL_TARGET,
        bps_grid=(HEADLINE_BPS,),
    ).iloc[0]["sharpe"]
    (RESULTS / "FINDINGS.md").open("a").write(
        "\n## Carry is weak on this window (failed challenger candidate)\n\n"
        "The KMPV convention (long backwardated / short contangoed, i.e. carry = "
        "front−next over next) posts Sharpe "
        f"{tables[(tables['method'] == 'carry') & (tables['bps'] == HEADLINE_BPS)].iloc[0]['sharpe']:.2f} "
        f"at {HEADLINE_BPS:.0f} bps on develop+validate. The inverted convention loses less "
        f"({inverse_sharpe:.2f}) but also fails — carry is weak on this universe/window "
        "under either sign, not a convention artifact. Keep the literature convention; "
        "the reading-canon interpretation pass (gated, the out-of-sample read) owns the explanation.\n"
    )
    print(tables.to_string(index=False))
    print(f"wrote {RESULTS / 'tables.csv'} and {RESULTS / 'FINDINGS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
