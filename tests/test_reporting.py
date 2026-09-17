"""The reporting path at its seam: the committed outcome documents, and the quoted attribution."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from systematic_futures.catalogue import BENCHMARK, LABELS, ML_VARIANTS
from systematic_futures.comparison import headline_rows
from systematic_futures.decision import verdict
from systematic_futures.protocol import PROTOCOL
from systematic_futures.reporting import (
    benchmark_line,
    family_names,
    outcome_split,
    outcome_table,
    sharpes_by_bucket,
)

ROOT = Path(__file__).resolve().parents[1]
#: The sub-periods the report and the memo quote: the trend drought, its recovery, and COVID.
SUB_PERIOD = lambda frame: pd.cut(
    frame["date"].dt.year, [2009, 2014, 2019, 2021], labels=["2010-2014", "2015-2019", "2020-2021"]
)


def applied(phase: int) -> tuple[dict, dict[str, dict]]:
    """The verdict and its robustness read, taken from a committed phase's tables."""
    tables = pd.read_csv(ROOT / f"results/phase{phase}/tables.csv")
    like = pd.read_csv(ROOT / f"results/phase{phase}/tables_like_for_like.csv")
    primary = verdict(
        headline_rows(tables, PROTOCOL.headline_bps), ML_VARIANTS, benchmark=BENCHMARK
    )
    sensitivity = {
        variant: verdict(
            headline_rows(like[like["window"] == variant], PROTOCOL.headline_bps),
            ML_VARIANTS,
            benchmark=BENCHMARK,
        )
        for variant in ML_VARIANTS
    }
    return primary, sensitivity


def test_the_committed_decision_records_are_what_the_renderer_produces():
    """DECISION.md and OOT.md carry the rendered read of the committed tables, not a second copy."""
    primary, sensitivity = applied(4)
    body = outcome_table(primary, sensitivity, ML_VARIANTS, LABELS)
    decision = (ROOT / "results/decision/DECISION.md").read_text()
    assert body in decision
    assert benchmark_line(primary, BENCHMARK) in decision

    primary, sensitivity = applied(5)
    meta = headline_rows(pd.read_csv(ROOT / "results/out_of_sample/tables.csv"), PROTOCOL.headline_bps)
    coverage = {v: [f"{meta.loc[v, 'signal_coverage']:.0%}"] for v in ML_VARIANTS}
    body = outcome_table(
        primary,
        sensitivity,
        ML_VARIANTS,
        LABELS,
        extra_headers=("OOT coverage",),
        extra_cells=coverage,
    )
    assert body in (ROOT / "results/out_of_sample/OOT.md").read_text()


def test_the_outcome_partition_and_the_names_are_what_the_sentences_are_written_from():
    primary = {"beats": {"ml_defaults": True, "ml_tuned": False}}
    winners, losers = outcome_split(primary, ML_VARIANTS)
    assert winners == ["ml_defaults"] and losers == ["ml_tuned"]
    assert family_names(winners, LABELS) == "6a `ml_defaults`"
    assert family_names(ML_VARIANTS, LABELS) == "6a `ml_defaults`, 6b `ml_tuned`"


def test_the_attribution_is_the_numbers_the_report_and_the_memo_quote():
    """The sub-period figures the docs quote come from the committed curves, by this bucketing."""
    curves = pd.read_csv(ROOT / "results/out_of_sample/curves.csv", parse_dates=["date"])
    develop = sharpes_by_bucket(curves[curves["window"] == "develop_validate"], SUB_PERIOD)

    assert develop.loc["ml_defaults", "2010-2014"].round(2) == -0.80  # the report's figure
    assert (
        develop.loc["ml_tuned", "2010-2014"] != develop.loc["ml_tuned", "2010-2014"]
    )  # no coverage
    assert develop.loc["ml_defaults", "2015-2019"].round(2) == 1.50
    assert develop.loc["ml_defaults", "2020-2021"].round(2) == 1.17
    assert develop.loc["tsmom", "2015-2019"] < 0 < develop.loc["tsmom", "2020-2021"]  # the drought

    by_year = sharpes_by_bucket(
        curves[curves["window"] == "oot"], lambda frame: frame["date"].dt.year
    )
    assert list(by_year.columns) == [2022, 2023, 2024]  # 2024 is Q1 only
    assert by_year.loc["tsmom", 2022] > 0 > by_year.loc["baseline", 2022]


def test_a_bucket_without_two_observations_reads_as_nan():
    """One session is not a Sharpe: a thin bucket is NaN, never a fabricated zero."""
    curves = pd.DataFrame(
        {
            "window": "oot",
            "method": "baseline",
            "date": pd.to_datetime(["2022-01-03", "2022-01-04", "2023-01-03"]),
            "ret": [0.01, -0.02, 0.03],
        }
    )
    table = sharpes_by_bucket(curves, lambda frame: frame["date"].dt.year)
    assert table.loc["baseline", 2022] < 0
    assert math.isnan(table.loc["baseline", 2023])  # a single session is not a Sharpe
