"""ML family — variants 6a/6b, seam-compatible like every classic family.

The seam contract is identical to `methods/`: a callable taking the wide
continuous close panel and returning a signal table on the same grid, with the
shared overlay and cost model applied by the engine. What differs is only what
lives underneath: a point-in-time feature panel, a hand-rolled purged and
embargoed walk-forward splitter, and a pooled LightGBM refit per fold.
"""

from systematic_futures.ml.cv import Fold, leakage_violations, purged_walk_forward
from systematic_futures.ml.features import FEATURES, feature_panel, forward_label
from systematic_futures.ml.model import lgbm_defaults, lgbm_tuned

__all__ = [
    "FEATURES",
    "Fold",
    "feature_panel",
    "forward_label",
    "leakage_violations",
    "lgbm_defaults",
    "lgbm_tuned",
    "purged_walk_forward",
]
