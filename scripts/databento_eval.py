"""Estimate the full-fetch billable size on Databento free credits.

Run: uv run python scripts/databento_eval.py   (needs DATABENTO_API_KEY in env)
Compares both datasets (CME + ICE softs) across the three schemas against ~$125 credits.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime

import pandas as pd

from systematic_futures.data.venues import DATASETS, roots_for

SCHEMAS = ("ohlcv-1d", "statistics", "definition")
START = "2009-01-01"


def main() -> int:
    import databento as db

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        print("DATABENTO_API_KEY not set", file=sys.stderr)
        return 1
    client = db.Historical(key)
    available = client.metadata.list_datasets()
    rows = []
    for dataset in DATASETS.values():
        roots = roots_for(dataset)
        if dataset not in available:
            print(f"{dataset}: NOT AVAILABLE — candidates: "
                  f"{[d for d in available if 'ICE' in d.upper() or 'GLBX' in d.upper()]}")
            continue
        for schema in SCHEMAS:
            size = client.metadata.get_billable_size(
                dataset=dataset,
                symbols=list(roots),
                schema=schema,
                start=START,
                end=datetime.now(tz=UTC).date().isoformat(),
            )["billable_size"]
            rows.append(
                {"dataset": dataset, "schema": schema, "symbols": len(roots),
                 "gib": round(size / 2**30, 3)}
            )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if len(df):
        print(f"total: {df['gib'].sum():.3f} GiB — price at console rate vs ~$125 credits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
