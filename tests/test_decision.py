"""The pre-declared decision rule at its seam: winner, failed challenger, disagreeing reads."""

from __future__ import annotations

import pandas as pd
import pytest

from systematic_futures.decision import require_same_verdict, verdict

BENCH = "tsmom"


def rows(spec: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """The headline rows a read hands the rule: method -> (sharpe, dsr)."""
    return pd.DataFrame(
        {"sharpe": {k: v[0] for k, v in spec.items()}, "dsr": {k: v[1] for k, v in spec.items()}}
    )


def test_a_variant_beats_the_benchmark_only_on_strictly_greater_dsr():
    read = verdict(
        rows(
            {
                BENCH: (0.10, 0.50),
                "ml_a": (0.80, 0.60),
                "ml_b": (0.20, 0.40),
                "ml_c": (0.30, 0.50),
            }
        ),
        ("ml_a", "ml_b", "ml_c"),
        benchmark=BENCH,
    )
    assert read["beats"] == {"ml_a": True, "ml_b": False, "ml_c": False}  # a tie is not a win
    assert read["benchmark_sharpe"] == 0.10 and read["benchmark_dsr"] == 0.50
    assert read["variant_dsr"] == {"ml_a": 0.60, "ml_b": 0.40, "ml_c": 0.50}
    assert read["variant_sharpe"]["ml_a"] == 0.80


def test_reads_may_differ_in_numbers_but_not_in_verdict():
    primary = verdict(
        rows({BENCH: (0.0, 0.50), "ml_a": (0.9, 0.90), "ml_b": (0.1, 0.10)}),
        ("ml_a", "ml_b"),
        benchmark=BENCH,
    )
    sharper_benchmark = verdict(
        rows({BENCH: (0.0, 0.40), "ml_a": (0.7, 0.80), "ml_b": (0.2, 0.20)}),
        ("ml_a", "ml_b"),
        benchmark=BENCH,
    )
    require_same_verdict(primary, {"like-for-like": sharper_benchmark})

    flip = verdict(
        rows({BENCH: (0.0, 0.50), "ml_a": (0.9, 0.10), "ml_b": (0.1, 0.10)}),
        ("ml_a", "ml_b"),
        benchmark=BENCH,
    )
    with pytest.raises(ValueError, match="must not depend on the read"):
        require_same_verdict(primary, {"like-for-like": flip})
