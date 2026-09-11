"""Build derived data from the frozen snapshot: observed-leg roll calendar,
back-adjusted continuous closes, front/deferred basis — then freeze the manifest.

Gated at both ends: `manifests/raw.json` is verified before anything is read, and
the pre-existing derived store is verified against `manifests/derived.json` before
it is overwritten — so a drifted input, or an out-of-band edit to the derived
store, fails the run instead of being absorbed. The fresh manifest is written
after the build, and `panels.load_frozen` verifies against it on every read.

Run: uv run python scripts/build_derived.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from systematic_futures.data.basis import compute_basis
from systematic_futures.data.continuous import back_adjust
from systematic_futures.data.manifest import verify_manifest, write_manifest
from systematic_futures.data.roll_calendar import (
    build_roll_calendar,
    contract_month,
    extract_contract_prices,
)
from systematic_futures.data.venues import INSTRUMENTS

RAW = Path("data/raw")
DERIVED = Path("data/derived")
MANIFESTS = Path("manifests")
RAW_MANIFEST = MANIFESTS / "raw.json"
DERIVED_MANIFEST = MANIFESTS / "derived.json"


def main() -> int:
    verify_manifest(RAW, RAW_MANIFEST)  # builds on a gated snapshot, never whatever is on disk
    if DERIVED_MANIFEST.exists():
        verify_manifest(DERIVED, DERIVED_MANIFEST)  # refuse to overwrite a drifted derived store
    DERIVED.mkdir(parents=True, exist_ok=True)
    (DERIVED / "continuous").mkdir(exist_ok=True)
    (DERIVED / "basis").mkdir(exist_ok=True)
    calendars = []
    report = []
    for root, instrument in INSTRUMENTS.items():
        mp = pd.read_csv(RAW / "multiple_prices" / f"{instrument}.csv")
        mp["DATETIME"] = pd.to_datetime(mp["DATETIME"])
        mp["symbol"] = root
        cal = build_roll_calendar(mp)
        raw = extract_contract_prices(mp)
        cont = back_adjust(raw, cal)
        basis = compute_basis(raw, cal)
        calendars.append(cal)
        cont.to_parquet(DERIVED / "continuous" / f"{root}.parquet")
        basis.to_parquet(DERIVED / "basis" / f"{root}.parquet")

        # roll sanity: successor strictly later than front; no suspicious calendar gaps
        bad_successors = int(
            (cal["next"].map(contract_month) - cal["front"].map(contract_month) < 1).sum()
        )
        gaps = cal["date"].diff().dropna().dt.days
        report.append(
            {
                "root": root,
                "first": str(cal["date"].min().date()),
                "last": str(cal["date"].max().date()),
                "days": len(cal),
                "rolls": int(cal["front"].ne(cal["front"].shift()).sum()) - 1,
                "bad_successors": bad_successors,
                "max_gap_days": int(gaps.max()),
                "basis_mean": round(basis["basis"].mean(), 5),
            }
        )
    pd.concat(calendars, ignore_index=True)[["date", "symbol", "front", "next"]].to_parquet(
        DERIVED / "roll_calendar.parquet"
    )
    pd.DataFrame(report).to_csv(DERIVED / "coverage_report.csv", index=False)
    print(pd.DataFrame(report).to_string(index=False))

    write_manifest(DERIVED, DERIVED_MANIFEST)
    print("derived store frozen; manifest gate: OK (manifests/derived.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
