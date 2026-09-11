"""Report figures: the committed derived series, drawn for the condensed report.

Reads `results/out_of_sample/curves.csv` (per-method daily net returns at the headline cost, both windows) —
derived statistics only, so no market data is needed to build the report.

Run: uv run python scripts/build_report_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # a headless backend, chosen before pyplot is imported

import matplotlib.pyplot as plt
import pandas as pd

from systematic_futures.harness import HEADLINE_BPS, VOL_TARGET

CURVES = Path("results/out_of_sample/curves.csv")
FIGURES = Path("docs/report/figures")
WINDOWS = (
    ("develop_validate", "Develop+validate 2010-2021"),
    ("oot", "Out-of-sample 2022-2024Q1"),
)


def main() -> int:
    curves = pd.read_csv(CURVES, parse_dates=["date"])
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.2))
    for ax, (window, title) in zip(axes, WINDOWS):
        for name, group in curves[curves["window"] == window].groupby("method"):
            equity = (1.0 + group.set_index("date")["ret"]).cumprod()
            ax.plot(equity.index, equity.to_numpy(), linewidth=1.0, label=name)
        ax.set_title(
            f"{title} — net equity, {HEADLINE_BPS:.0f} bps/side, "
            f"{VOL_TARGET:.0%} per-symbol vol target"
        )
        ax.set_ylabel("equity (start = 1)")
        ax.legend(ncols=4, fontsize=7)
        ax.grid(alpha=0.25)
    fig.tight_layout()

    out = FIGURES / "equity.pdf"
    # No embedded creation date: the committed figure must rebuild byte-for-byte.
    fig.savefig(out, metadata={"CreationDate": None})
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
