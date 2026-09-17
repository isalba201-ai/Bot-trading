"""Phase 3: point-in-time feature computation over validated candles.

See FEATURES.md for the exact formula and point-in-time contract of every
feature, and ARCHITECTURE.md for how this subpackage fits between data
validation (Phase 2) and the backtesting engine (Phase 4).
"""

from otc_research.features.engine import FEATURE_NAMES, FEATURE_SET_VERSION, compute_features

__all__ = ["FEATURE_NAMES", "FEATURE_SET_VERSION", "compute_features"]
