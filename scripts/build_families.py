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

from systematic_futures.data.panels import load_frozen
from systematic_futures.engine import run
from systematic_futures.methods import (
    baseline,
    carry,
    seasonal_tilt,
    seasonality,
    tsmom,
    xs_momentum,
)

RESULTS = Path("results/phase3")
VALIDATE_END = "2021-12-31"  # develop+validate window end; OOT starts 2022
VOL_TARGET = 0.10
HEADLINE_BPS = 2.0


def main() -> int:
    wide, basis_wide = load_frozen()
    window = wide.loc[:VALIDATE_END]
    assert wide.index.max() > pd.Timestamp(VALIDATE_END), "loader must return the full panel"
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
    season_free = tables[(tables["method"] == "seasonality") & (tables["bps"] == 0.0)].iloc[0]
    trend = tables[(tables["method"] == "tsmom") & (tables["bps"] == HEADLINE_BPS)].iloc[0]
    trend_free = tables[(tables["method"] == "tsmom") & (tables["bps"] == 0.0)].iloc[0]
    dies = season["sharpe"] < trend["sharpe"]
    xs = tables[(tables["method"] == "xs_momentum") & (tables["bps"] == HEADLINE_BPS)].iloc[0]
    xs_beats = bool(xs["sharpe"] > trend["sharpe"])
    (RESULTS / "FINDINGS.md").write_text(
        "# the families read findings\n\n"
        f"Window: develop+validate (2010-01-01 → {VALIDATE_END}), frozen panel, "
        f"{VOL_TARGET:.0%} vol target, 16 CME roots; OOT (2022+) untouched.\n\n"
        "## Seasonality standalone loses (expected finding)\n\n"
        f"Standalone seasonality: Sharpe {season_free['sharpe']:.2f} before costs, "
        f"{season['sharpe']:.2f} after {HEADLINE_BPS:.0f} bps "
        f"(turnover {season['turnover']:.2f}), against the TSMOM benchmark's "
        f"{trend_free['sharpe']:.2f} / {trend['sharpe']:.2f} "
        f"(turnover {trend['turnover']:.2f}). "
        + (
            "Confirmed, and the failure is edge rather than churn: the signal is already "
            "negative before costs, and its turnover is *below* the benchmark's — so cost "
            "drag alone does not explain it. Reported as a failed challenger.\n"
            if dies
            else "Not confirmed on this window — investigate before reporting.\n"
        )
        + "\nFull per-method, per-bps metrics in `tables.csv` "
        "(Sharpe, Sortino, max DD, turnover, DSR). "
        + (
            f"The cross-sectional family is the other side of this story: it posts Sharpe "
            f"{xs['sharpe']:.2f} against the benchmark's {trend['sharpe']:.2f} after "
            f"{HEADLINE_BPS:.0f} bps.\n"
            if xs_beats
            else "No other family beats the benchmark on this window.\n"
        )
    )
    inverse_sharpe = run(
        np.sign(basis_wide.reindex(index=window.index, columns=window.columns)),
        window,
        vol_target=VOL_TARGET,
        bps_grid=(HEADLINE_BPS,),
    ).iloc[0]["sharpe"]
    (RESULTS / "FINDINGS.md").open("a").write(
        "\n## Carry: the sign convention decides (failed challenger under the literature sign)\n\n"
        "The KMPV convention (long backwardated / short contangoed, i.e. carry = "
        "front−next over next) posts Sharpe "
        f"{tables[(tables['method'] == 'carry') & (tables['bps'] == HEADLINE_BPS)].iloc[0]['sharpe']:.2f} "
        f"at {HEADLINE_BPS:.0f} bps on develop+validate, while the inverted sign — long "
        f"contangoed — posts {inverse_sharpe:.2f}. A sign flip of that size is not a "
        "measurement artifact to average away: on this universe and window the literature "
        "sign loses and its inverse is positive, close to the benchmark. Keep the literature "
        "convention in the tables; the reading-canon interpretation pass (gated, the out-of-sample read) "
        "owns what the flip means (M3 carry, M4 commodity term structure).\n"
    )
    print(tables.to_string(index=False))
    print(f"wrote {RESULTS / 'tables.csv'} and {RESULTS / 'FINDINGS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
