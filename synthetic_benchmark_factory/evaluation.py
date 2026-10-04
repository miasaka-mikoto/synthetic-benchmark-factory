"""Backward-compatible evaluator module.

The implementation lives in :mod:`synthetic_benchmark_factory.metrics`; this
module keeps the intuitive ``evaluation`` import available to integrations.
"""

from .metrics import *  # noqa: F401,F403

