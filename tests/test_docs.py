"""The hand-typed numbers in the docs are the committed tables' values, at the precision they print.

The condensed report's two tables, the memo's three and the self-test's quoted rows are transcribed by
hand from `results/phase{3,4,5}/tables.csv`. This test is what keeps them from drifting when the tables
are regenerated — the drift the 2026-09-17 sanity check found (15 memo cells, 3 report cells) is what
it fails on. Comparison is exact at the precision the document prints: `0.746` and a table value of
0.7457 agree, `0.753` does not.

Scope: tables, and the `(...)` rows inside the sections that quote them. Prose that restates a table
row in a sentence is still hand-transcribed — the two occurrences of the 5 bps survivors list are —
and stays out of this check deliberately.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/report/report.tex"
MEMO = ROOT / "docs/findings/memo.md"

HEADLINE = 2.0
LEVELS = (0.0, 2.0, 5.0, 10.0)


@pytest.fixture(scope="module")
def tables() -> dict[int, pd.DataFrame]:
    return {phase: pd.read_csv(ROOT / f"results/phase{phase}/tables.csv") for phase in (3, 4, 5)}


def headline(tables: dict[int, pd.DataFrame], phase: int) -> pd.DataFrame:
    """One row per method at the headline cost level."""
    return tables[phase][tables[phase]["bps"] == HEADLINE].set_index("method")


def value(table: pd.DataFrame, method: str, bps: float, column: str) -> float:
    return float(table[(table["method"] == method) & (table["bps"] == bps)][column].iloc[0])


# How each document names a family. Longest names first: the label is matched whole before tokens.
FAMILIES = {
    "baseline": "baseline",
    "Baseline": "baseline",
    "tsmom": "tsmom",
    "TSMOM": "tsmom",
    "xs_momentum": "xs_momentum",
    "XS momentum": "xs_momentum",
    "carry": "carry",
    "Carry": "carry",
    "seasonality": "seasonality",
    "Seasonality": "seasonality",
    "seasonal_tilt": "seasonal_tilt",
    "Seasonal tilt": "seasonal_tilt",
    "6a": "ml_defaults",
    "ml_defaults": "ml_defaults",
    "6b": "ml_tuned",
    "ml_tuned": "ml_tuned",
}


def family(label: str) -> str:
    """The method a document's row label names, however that document spells it."""
    cleaned = re.sub(r"\\texttt\{[^}]*\}", "", label)
    cleaned = re.sub(r"\(benchmark\)", "", cleaned).replace("`", "").strip()
    if cleaned in FAMILIES:
        return FAMILIES[cleaned]
    for token in cleaned.split():
        if token in FAMILIES:
            return FAMILIES[token]
    raise KeyError(f"unmapped row label: {label!r}")


def cell_value(cell: str) -> float | None:
    """A table cell as a number, or None when the cell is not one (a label, a note, blank)."""
    cleaned = re.sub(r"[\\${}`]", "", cell).strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def check(where: str, printed: float, expected: float) -> None:
    """Compare at the precision printed: the number in the doc IS the rounded table value."""
    decimals = len(repr(printed).split(".")[-1])
    assert round(expected, decimals) == printed, (
        f"{where}: the doc prints {printed}, the committed table says {round(expected, decimals)}"
    )


def section(text: str, start: str, end: str) -> list[str]:
    """The lines between two markers, the start marker included."""
    body = text[text.index(start) :]
    return body[: body.index(end, len(start))].splitlines()


def test_report_headline_table_matches_the_committed_tables(tables):
    """Table 1: Sharpe and DSR, develop+validate and out-of-sample, at the headline cost."""
    lines = section(REPORT.read_text(), "Family & Sharpe & DSR", "\\end{table}")
    rows = [line for line in lines if "&" in line and line.strip().endswith("\\\\")]
    rows = [row for row in rows if not row.startswith("Family")]
    assert len(rows) == 8
    columns = (("sharpe", 4), ("dsr", 4), ("sharpe", 5), ("dsr", 5))
    for row in rows:
        cells = [cell.strip() for cell in row.rstrip("\\").split("&")]
        method = family(cells[0])
        printed = [cell_value(cell) for cell in cells[1:]]
        assert len(printed) == len(columns)
        for (column, phase), got in zip(columns, printed):
            where = "develop+validate" if phase == 4 else "out-of-sample"
            check(
                f"report.tex Table 1 {method} {column} ({where})",
                got,
                value(tables[phase], method, HEADLINE, column),
            )


def test_report_cost_ladder_matches_the_committed_table(tables):
    """Table 2: Sharpe by cost level, develop+validate."""
    lines = section(REPORT.read_text(), "Family & $0\\bps$", "\\end{table}")
    rows = [line for line in lines if "&" in line and line.strip().endswith("\\\\")]
    rows = [row for row in rows if not row.startswith("Family")]
    assert len(rows) == 8
    for row in rows:
        cells = [cell.strip() for cell in row.rstrip("\\").split("&")]
        method = family(cells[0])
        printed = [cell_value(cell) for cell in cells[1:]]
        assert len(printed) == len(LEVELS)
        for bps, got in zip(LEVELS, printed):
            check(
                f"report.tex Table 2 {method} @{bps:g}bps",
                got,
                value(tables[4], method, bps, "sharpe"),
            )


@pytest.mark.parametrize(
    ("marker", "ending", "phase"), [("### 4.1", "### 4.2", 4), ("### 4.2", "### 4.3", 5)]
)
def test_memo_headline_tables_match_the_committed_tables(tables, marker, ending, phase):
    """Memo §4.1/§4.2: Sharpe, DSR and turnover, develop+validate and out-of-sample."""
    lines = section(MEMO.read_text(), marker, ending)
    rows = [line for line in lines if line.startswith("| `")]
    assert len(rows) == 8
    source = tables[phase]
    for row in rows:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        method = family(cells[0])
        for column, cell in zip(("sharpe", "dsr", "turnover"), cells[1:4]):
            check(
                f"memo {marker} {method} {column}",
                cell_value(cell),
                value(source, method, HEADLINE, column),
            )


def test_memo_cost_ladder_matches_the_committed_tables(tables):
    """Memo §4.3: Sharpe by cost level, each cell a develop+validate / out-of-sample pair."""
    lines = section(MEMO.read_text(), "### 4.3 Cost sensitivity", "## 5. Multiple-testing")
    rows = [line for line in lines if line.startswith("| `")]
    assert len(rows) == 8
    for row in rows:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        method = family(cells[0])
        assert len(cells) - 1 == len(LEVELS)
        for bps, cell in zip(LEVELS, cells[1:]):
            printed = [float(number) for number in re.findall(r"-?\d+\.\d+", cell)]
            assert len(printed) == 2, f"memo §4.3 {method} @{bps:g}bps: {cell!r}"
            for phase, got in zip((4, 5), printed):
                check(
                    f"memo §4.3 {method} @{bps:g}bps {'dev' if phase == 4 else 'oot'}",
                    got,
                    value(tables[phase], method, bps, "sharpe"),
                )
