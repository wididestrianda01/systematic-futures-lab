"""Download the MOP-replication subset of the pinned snapshot → data/replication/, freeze it.

Same snapshot, same gate as `fetch_raw.py`, different universe: the instruments
MOP (2012) Table 1 names, as far as the free store carries them. Kept apart from
`data/raw/` so the lab's certified manifest and this replication's manifest never
gate each other.

Files in the store that no mapped instrument claims are removed — the store is
exactly what this script fetches, so a stray download cannot silently become
part of a published number.

Run: uv run python scripts/fetch_replication.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from systematic_futures.data.manifest import freeze
from systematic_futures.replication import AVAILABLE

SNAPSHOT_SHA = "b4a25e6e1e33a54a3ecfb45c0f6db5e2b60b84f8"
BASE = f"https://raw.githubusercontent.com/robcarver17/pysystemtrade/{SNAPSHOT_SHA}/data/futures"
REPL = Path("data/replication")
MANIFEST = Path("manifests/replication.json")
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
    target = REPL / "multiple_prices"
    target.mkdir(parents=True, exist_ok=True)
    for stem in sorted(AVAILABLE):
        out = target / f"{stem}.csv"
        if not out.exists():
            out.write_bytes(fetch(f"multiple_prices_csv/{stem}.csv"))
        df = pd.read_csv(out)
        missing = [c for c in MP_COLUMNS if c not in df.columns]
        if missing or df.empty:
            raise RuntimeError(f"{stem}: malformed multiple_prices (missing {missing})")
    for stray in sorted(target.glob("*.csv")):
        if stray.stem not in AVAILABLE:
            stray.unlink()
            print(f"removed unmapped {stray.name}")
    freeze(REPL, MANIFEST)
    print(f"{len(AVAILABLE)} instruments frozen; manifest gate: OK ({MANIFEST})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
