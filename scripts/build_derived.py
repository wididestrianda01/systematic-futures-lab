"""Build derived data from the frozen snapshot: observed-leg roll calendar,
back-adjusted continuous closes, front/deferred basis — then freeze the manifest.

Run: uv run python scripts/build_derived.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from systematic_futures.data.basis import compute_basis
from systematic_futures.data.continuous import back_adjust
from systematic_futures.data.manifest import verify_manifest, write_manifest
from systematic_futures.data.roll_calendar import build_roll_calendar, extract_contract_prices
from systematic_futures.data.venues import INSTRUMENTS

RAW = Path("data/raw")
DERIVED = Path("data/derived")
MANIFESTS = Path("manifests")


def month_index(contract_id: float) -> float:
    """YYYYMM00 → absolute month index (year-boundary-safe successor checks)."""
    return (contract_id // 10000) * 12 + (contract_id // 100) % 100


def main() -> int:
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
            (cal["next"].map(month_index) - cal["front"].map(month_index) < 1).sum()
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

    manifest = write_manifest(DERIVED)
    MANIFESTS.mkdir(exist_ok=True)
    (MANIFESTS / "derived.json").write_text(manifest.read_text())
    verify_manifest(DERIVED, MANIFESTS / "derived.json")
    print("derived store frozen; manifest gate: OK (manifests/derived.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
