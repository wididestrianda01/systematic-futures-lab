"""Fetch contract-level raw data → data/raw/{dataset_dir}/{schema}/{root}.parquet,
then freeze the manifest. Idempotent: existing files are skipped unless --force;
re-runs are gated by the manifest (drift = hard failure).

Run: uv run python scripts/fetch_raw.py   (needs DATABENTO_API_KEY; first run
verifies stat_type mapping for open interest and definition field names).
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from systematic_futures.data.manifest import verify_manifest, write_manifest
from systematic_futures.data.venues import roots_for

SCHEMAS = ("definition", "ohlcv-1d", "statistics")
START = "2009-01-01"
DATA_DIR = Path("data/raw")
DATASET_DIRS = {"GLBX.MDP3": "glbx", "IFUS.ICE": "iceus"}
OI_STAT_TYPE = 8  # DBN stat_type for open interest — verify against first-run print


def fetch(client, dataset: str, root: str, schema: str) -> pd.DataFrame:
    resp = client.timeseries.get_dataset(
        dataset=dataset,
        symbols=[root],  # parent symbol → all contracts under the root
        schema=schema,
        start=START,
        end=datetime.now(tz=UTC).date().isoformat(),
    )
    df = resp.to_df().reset_index()
    if df.empty:
        raise RuntimeError(f"{dataset}/{schema}/{root}: empty response — symbol surprise")
    ts_col = next(c for c in ("ts_event", "ts_recv", "date", "time") if c in df.columns)
    df["date"] = pd.to_datetime(df[ts_col]).dt.normalize()
    return df


def normalize(df: pd.DataFrame, schema: str) -> pd.DataFrame:
    if schema == "ohlcv-1d":
        return df[["date", "raw_symbol", "open", "high", "low", "close", "volume"]]
    if schema == "statistics":
        # DBN statistics rows carry a stat_type discriminator; the first run prints
        # observed values so the open-interest mapping is verified, never assumed.
        if "stat_type" in df.columns:
            print("stat_type values seen:", sorted(df["stat_type"].unique()))
            df = df[df["stat_type"] == OI_STAT_TYPE]
        return df[["date", "raw_symbol", "open_interest"]]
    if schema == "definition":
        daily = df.sort_values("date").drop_duplicates("raw_symbol", keep="last")
        ts = next(c for c in ("expiration", "last_trade_time") if c in daily.columns)
        return daily[["raw_symbol", ts]].rename(columns={ts: "expiration"})
    raise ValueError(schema)



def main() -> int:
    import databento as db

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        print("DATABENTO_API_KEY not set", file=sys.stderr)
        return 1
    force = "--force" in sys.argv
    client = db.Historical(key)
    for dataset, ddir in DATASET_DIRS.items():
        roots = list(roots_for(dataset))
        for schema in SCHEMAS:
            for root in roots:
                out = DATA_DIR / ddir / schema / f"{root}.parquet"
                if out.exists() and not force:
                    continue
                df = normalize(fetch(client, dataset, root, schema), schema)
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out)
                print(f"wrote {out} ({len(df)} rows)")
        got = {p.stem for p in (DATA_DIR / ddir).rglob("*.parquet")}
        missing = set(roots) - got
        if missing:
            raise RuntimeError(f"{dataset}: missing roots after fetch: {sorted(missing)}")
    write_manifest(DATA_DIR)
    verify_manifest(DATA_DIR)
    print("raw store frozen; manifest gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
