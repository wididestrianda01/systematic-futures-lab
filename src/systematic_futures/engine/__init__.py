"""Shared vectorized engine — one accounting path for every method family."""

from systematic_futures.engine.core import account, curve, run

__all__ = ["account", "curve", "run"]
