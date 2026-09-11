"""TSMOM parameter sweep on the frozen panel — the trend family's sensitivity read.

One sleeve per horizon through the identical seam, 0/2/5/10 bps, develop+validate
window; the ensemble row is the benchmark the phase tables use. This sweeps the
family's own parameter (which horizons, at which cost), never the comparison set.

The sweep is cross-checked against vectorbt in `tests/test_vectorbt.py` — the
checker stays a checker, so nothing here imports vectorbt.

Run: uv run python scripts/sweep_tsmom.py
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

from systematic_futures.data.panels import develop_validate, load_frozen
from systematic_futures.engine import run
from systematic_futures.harness import BENCHMARK, VOL_TARGET
from systematic_futures.methods.tsmom import TSMOM_HORIZONS, horizon_signal, tsmom

RESULTS = Path("results/phase2")


def main() -> int:
    wide, _ = load_frozen()
    window = develop_validate(wide)

    methods = {f"sleeve_{h}d": partial(horizon_signal, lookback=h) for h in TSMOM_HORIZONS}
    methods[BENCHMARK] = tsmom

    frames = []
    for name, method in methods.items():
        table = run(method, window, vol_target=VOL_TARGET)
        table["method"] = name
        frames.append(table.reset_index().rename(columns={"index": "bps"}))
    sweep = pd.concat(frames, ignore_index=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(RESULTS / "tsmom_sweep.csv", index=False)
    print(sweep.to_string(index=False))
    print(f"\nwrote {RESULTS / 'tsmom_sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
