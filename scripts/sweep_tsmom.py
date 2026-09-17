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

from systematic_futures.catalogue import BENCHMARK
from systematic_futures.comparison import table_for
from systematic_futures.data.panels import develop_validate, load_frozen
from systematic_futures.methods.tsmom import TSMOM_HORIZONS, horizon_signal, tsmom
from systematic_futures.protocol import PROTOCOL, Method

RESULTS = Path("results/phase2")


def main() -> int:
    wide, _ = load_frozen()
    window = develop_validate(wide)

    methods = {f"sleeve_{h}d": Method(partial(horizon_signal, lookback=h)) for h in TSMOM_HORIZONS}
    methods[BENCHMARK] = Method(tsmom)

    sweep = table_for(methods, window, protocol=PROTOCOL)

    RESULTS.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(RESULTS / "tsmom_sweep.csv", index=False)
    print(sweep.to_string(index=False))
    print(f"\nwrote {RESULTS / 'tsmom_sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
