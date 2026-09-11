"""The comparison harness at its seam: the comparison set, the assembly, the one read."""

from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import continuous_wide

from systematic_futures.engine import run
from systematic_futures.harness import (
    BENCHMARK,
    HEADLINE_BPS,
    VOL_TARGET,
    classic_set,
    headline_rows,
    table_for,
)
from systematic_futures.methods import baseline, tsmom


def test_table_for_is_one_row_per_method_per_cost_level():
    closes = continuous_wide()
    table = table_for({"baseline": baseline, BENCHMARK: tsmom}, closes)
    assert sorted(table["method"].unique()) == ["baseline", "tsmom"]
    assert list(table["bps"].unique()) == [0.0, 2.0, 5.0, 10.0]
    assert len(table) == 2 * 4
    assert table.notna().all().all()


def test_trials_deflate_only_the_method_they_name():
    closes = continuous_wide()
    methods = {"baseline": baseline, BENCHMARK: tsmom}
    plain = headline_rows(table_for(methods, closes))
    deflated = headline_rows(table_for(methods, closes, trials={BENCHMARK: 20}))
    assert deflated.loc[BENCHMARK, "dsr"] < plain.loc[BENCHMARK, "dsr"]  # more trials, harsher bar
    assert deflated.loc["baseline", "dsr"] == plain.loc["baseline", "dsr"]


def test_headline_rows_carry_the_seam_metrics_of_each_method():
    closes = continuous_wide()
    rows = headline_rows(table_for({"baseline": baseline, BENCHMARK: tsmom}, closes))
    assert list(rows.index) == ["baseline", "tsmom"]
    expected = run(tsmom, closes, vol_target=VOL_TARGET).loc[HEADLINE_BPS]
    np.testing.assert_array_equal(
        rows.loc[BENCHMARK, ["sharpe", "sortino", "max_dd", "turnover", "dsr"]].to_numpy(),
        expected.to_numpy(),
    )


def test_classic_set_binds_the_basis_panel_it_is_given():
    closes = continuous_wide(("ES", "GC"))
    basis = closes.pct_change(21)  # stand-in basis panel: trailing, never forward-looking
    up = classic_set(basis)["carry"](closes)
    down = classic_set(-basis)["carry"](closes)
    assert up.notna().any().any()  # the bound family does read the panel it was handed
    pd.testing.assert_frame_equal(down, -up)
