"""The comparison table's shape, its reads, and the seam that keeps them free of the ML stack."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
from conftest import continuous_wide

from systematic_futures.comparison import (
    covered_dates,
    headline_rows,
    protocol_meta,
    resolved,
    table_for,
)
from systematic_futures.engine import run
from systematic_futures.methods import baseline, tsmom
from systematic_futures.protocol import PROTOCOL, Method, Protocol

BENCHMARK = "tsmom"
SET = {"baseline": Method(baseline), BENCHMARK: Method(tsmom)}
METRICS = ["sharpe", "sortino", "max_dd", "turnover", "dsr"]


def test_table_for_is_one_row_per_method_per_cost_level():
    closes = continuous_wide()
    table = table_for(resolved(SET, closes), closes)
    assert sorted(table["method"].unique()) == ["baseline", "tsmom"]
    assert list(table["bps"].unique()) == [0.0, 2.0, 5.0, 10.0]
    assert len(table) == 2 * 4
    assert table.notna().all().all()


def test_trials_deflate_only_the_method_that_carries_them():
    closes = continuous_wide()
    plain = headline_rows(table_for(resolved(SET, closes), closes), PROTOCOL.headline_bps)
    searched = {**SET, BENCHMARK: Method(tsmom, trials=20)}
    deflated = headline_rows(table_for(resolved(searched, closes), closes), PROTOCOL.headline_bps)
    assert deflated.loc[BENCHMARK, "dsr"] < plain.loc[BENCHMARK, "dsr"]  # more trials, harsher bar
    assert deflated.loc["baseline", "dsr"] == plain.loc["baseline", "dsr"]


def test_headline_rows_carry_the_seam_metrics_of_each_method():
    closes = continuous_wide()
    rows = headline_rows(table_for(resolved(SET, closes), closes), PROTOCOL.headline_bps)
    assert list(rows.index) == ["baseline", "tsmom"]
    expected = run(tsmom, closes, vol_target=PROTOCOL.vol_target).loc[PROTOCOL.headline_bps]
    np.testing.assert_array_equal(rows.loc[BENCHMARK, METRICS].to_numpy(), expected.to_numpy())


def test_the_table_is_sized_by_the_protocol_it_is_handed():
    """The overlay is the protocol's: a different protocol is a different table, not a different call."""
    closes = continuous_wide()
    half = Protocol(vol_target=PROTOCOL.vol_target / 2)
    rows = headline_rows(table_for(resolved(SET, closes), closes, protocol=half), 0.0)
    expected = run(tsmom, closes, vol_target=half.vol_target).loc[0.0]
    np.testing.assert_array_equal(rows.loc[BENCHMARK, METRICS].to_numpy(), expected.to_numpy())


def test_protocol_meta_reports_the_trials_and_the_covered_share_of_the_window():
    closes = continuous_wide()
    methods = resolved(SET, closes)
    dates = closes.index[60:]
    meta = protocol_meta(methods, dates)
    covered = covered_dates(methods[BENCHMARK].signal, dates)
    assert meta.loc[BENCHMARK, "trials"] == 1
    assert meta.loc[BENCHMARK, "signal_coverage"] == len(covered) / len(dates)
    assert 0.0 < meta.loc[BENCHMARK, "signal_coverage"] <= 1.0


def test_table_for_masks_every_method_to_the_dates_it_is_given():
    closes = continuous_wide()
    dates = closes.index[-100:]
    methods = resolved(SET, closes)
    masked = headline_rows(table_for(methods, closes, dates=dates), 0.0)
    expected = run(tsmom, closes, vol_target=PROTOCOL.vol_target, dates=dates).loc[0.0]
    np.testing.assert_array_equal(masked.loc[BENCHMARK, METRICS].to_numpy(), expected.to_numpy())
    full = headline_rows(table_for(methods, closes), 0.0)
    assert masked.loc["baseline", "turnover"] != full.loc["baseline", "turnover"]


def test_importing_the_table_shape_costs_no_learner():
    """The seam the split exists for: the shape and the rule cross no family and no ML stack."""
    code = (
        "import sys, systematic_futures.comparison, systematic_futures.decision;"
        "heavy = [m for m in ('lightgbm', 'optuna') if m in sys.modules];"
        "print(heavy); raise SystemExit(1 if heavy else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, f"the table shape loaded {done.stdout.strip()}"
