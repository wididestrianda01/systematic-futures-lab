"""DuckDB query layer over the on-disk Parquet store (raw + derived)."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

VIEWS = {
    "raw_ohlcv": "raw/*/ohlcv-1d/*.parquet",
    "raw_definitions": "raw/*/definition/*.parquet",
    "raw_statistics": "raw/*/statistics/*.parquet",
    "roll_calendar": "derived/roll_calendar.parquet",
    "continuous": "derived/continuous/*.parquet",
    "basis": "derived/basis/*.parquet",
}

COVERAGE_QUERY = """
select regexp_extract(filename, '([^/]+)\\.parquet$', 1) as root,
       year(date) as year, count(*) as rows
from raw_ohlcv
group by 1, 2
order by 1, 2
"""

ROLL_COUNTS_QUERY = """
with transitions as (
    select symbol, front,
           lag(front) over (partition by symbol order by date) as prev_front
    from roll_calendar
)
select symbol, count(*) filter (where prev_front is not null
                                and front is distinct from prev_front) as rolls
from transitions
group by 1
order by 1
"""

OI_SUMMARY_QUERY = """
select regexp_extract(filename, '([^/]+)\\.parquet$', 1) as root,
       count(*) as days, avg(open_interest) as mean_oi, max(date) as last_date
from raw_statistics
group by 1
order by 1
"""


def register_views(con: duckdb.DuckDBPyConnection, data_dir: Path | str) -> None:
    """Views over the frozen store; filename=true exposes the source path."""
    for name, rel in VIEWS.items():
        path = str(Path(data_dir) / rel).replace("'", "''")
        con.execute(
            f"create or replace view {name} as "
            f"select * from read_parquet('{path}', filename=true)"
        )


def query(data_dir: Path | str, sql: str) -> pd.DataFrame:
    """One-shot analytical query against the frozen store."""
    con = duckdb.connect()
    try:
        register_views(con, data_dir)
        return con.execute(sql).df()
    finally:
        con.close()
