"""DuckDB query layer over the on-disk store (raw snapshot CSVs + derived Parquet)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

VIEWS = {
    "multiple_prices": "raw/multiple_prices/*.csv",
    "roll_calendar": "derived/roll_calendar.parquet",
    "continuous": "derived/continuous/*.parquet",
    "basis": "derived/basis/*.parquet",
}

COVERAGE_QUERY = """
select regexp_extract(filename, '([^/]+)\\.csv$', 1) as instrument,
       min(cast(DATETIME as date)) as first_date,
       max(cast(DATETIME as date)) as last_date,
       count(*) as rows
from multiple_prices
group by 1
order by 1
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

CONTINUOUS_COVERAGE_QUERY = """
select symbol, year(date) as year, count(*) as rows
from continuous
group by 1, 2
order by 1, 2
"""


def register_views(con: duckdb.DuckDBPyConnection, data_dir: Path | str) -> None:
    """Views over the frozen store; filename=true exposes the source path."""
    for name, rel in VIEWS.items():
        path = str(Path(data_dir) / rel).replace("'", "''")
        read = "read_csv" if rel.endswith(".csv") else "read_parquet"
        con.execute(
            f"create or replace view {name} as select * from {read}('{path}', filename=true)"
        )


def query(data_dir: Path | str, sql: str) -> pd.DataFrame:
    """One-shot analytical query against the frozen store."""
    con = duckdb.connect()
    try:
        register_views(con, data_dir)
        return con.execute(sql).df()
    finally:
        con.close()
