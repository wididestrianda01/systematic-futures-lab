"""Cross-sectional ranking — the shared map from per-symbol scores to a
centered tilt in [-1, 1] (strongest +1, weakest -1, median 0; ties average).

One implementation, three users: the XS momentum family, the ML feature ranks,
and the ML signal map. They must speak the same cross-sectional language or the
comparison stops being apples-to-apples. Symbols without a score on a day are
excluded from that day's ranking (NaN in, NaN out); a day with a single score
is NaN everywhere (nothing to rank against).
"""

from __future__ import annotations

import pandas as pd


def centered_rank(scores: pd.DataFrame) -> pd.DataFrame:
    """Centered per-row rank of `scores`, computed from that row alone."""
    n = scores.notna().sum(axis=1)
    rank = scores.rank(axis=1)  # average ranks on ties; NaN symbols skipped
    return rank.sub(n.add(1.0).div(2.0), axis=0).div(n.sub(1.0).div(2.0), axis=0)
