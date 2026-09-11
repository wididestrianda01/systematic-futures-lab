"""Download the pinned pysystemtrade snapshot → data/raw/, freeze the manifest.

Idempotent: existing files are kept (the snapshot is frozen upstream — a change
would be a manifest-gate error, not an update). Re-run verifies the gate.

Run: uv run python scripts/fetch_raw.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from systematic_futures.data.manifest import verify_manifest, write_manifest
from systematic_futures.data.venues import INSTRUMENTS

SNAPSHOT_SHA = "b4a25e6e1e33a54a3ecfb45c0f6db5e2b60b84f8"
BASE = f"https://raw.githubusercontent.com/robcarver17/pysystemtrade/{SNAPSHOT_SHA}/data/futures"
DATA_DIR = Path("data/raw")
MANIFESTS = Path("manifests")
MP_COLUMNS = [
    "DATETIME",
    "CARRY",
    "CARRY_CONTRACT",
    "PRICE",
    "PRICE_CONTRACT",
    "FORWARD",
    "FORWARD_CONTRACT",
]


def fetch(rel: str) -> bytes:
    with urllib.request.urlopen(f"{BASE}/{rel}", timeout=60) as resp:
        return resp.read()


def main() -> int:
    for instrument in INSTRUMENTS.values():
        out = DATA_DIR / "multiple_prices" / f"{instrument}.csv"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(fetch(f"multiple_prices_csv/{instrument}.csv"))
        df = pd.read_csv(out)
        missing = [c for c in MP_COLUMNS if c not in df.columns]
        if missing or df.empty:
            raise RuntimeError(f"{instrument}: malformed multiple_prices (missing {missing})")
        print(f"ok {instrument}: {len(df)} rows, {df['PRICE'].notna().sum()} priced days")
    manifest = write_manifest(DATA_DIR)
    MANIFESTS.mkdir(exist_ok=True)
    (MANIFESTS / "raw.json").write_text(manifest.read_text())
    verify_manifest(DATA_DIR, MANIFESTS / "raw.json")
    print("raw store frozen; manifest gate: OK (manifests/raw.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
