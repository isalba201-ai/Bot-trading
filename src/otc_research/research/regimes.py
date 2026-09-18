"""Market regime classification, derived purely from already-computed
point-in-time features (``adx_14``, ``atr_expansion_ratio``) — no new data
source, no look-ahead: the regime label at timestamp ``t`` depends only on
feature values already guaranteed causal by features/indicators.py.

First version, deliberately simple (a fixed 2x3 grid), same spirit as
``structure_bias``'s docstring: kept only if ablation in the discovery
layer shows it actually helps, refined otherwise. The thresholds below
are the standard Wilder convention for ADX (>=25 trending, per the
original textbook definition) and a plain symmetric split for the
volatility-expansion ratio around 1.0 — not fit to this project's data,
so they can't have been data-mined into looking better than they are.
"""

from __future__ import annotations

import pandas as pd

ADX_TREND_THRESHOLD = 25.0
VOL_EXPANSION_THRESHOLD = 1.2
VOL_CONTRACTION_THRESHOLD = 0.8

TREND_STRENGTH_LABELS = ("strong_trend", "weak_trend")
VOLATILITY_LABELS = ("expanding_vol", "normal_vol", "contracting_vol")

#: Every possible combined regime label, fixed in advance so downstream
#: code (discovery, contingency tables) can enumerate them without having
#: to first scan the data.
REGIME_LABELS: tuple[str, ...] = tuple(
    f"{trend}_{vol}" for trend in TREND_STRENGTH_LABELS for vol in VOLATILITY_LABELS
)


def _trend_strength_label(adx: float) -> str:
    return "strong_trend" if adx >= ADX_TREND_THRESHOLD else "weak_trend"


def _volatility_label(atr_expansion_ratio: float) -> str:
    if atr_expansion_ratio >= VOL_EXPANSION_THRESHOLD:
        return "expanding_vol"
    if atr_expansion_ratio <= VOL_CONTRACTION_THRESHOLD:
        return "contracting_vol"
    return "normal_vol"


def classify_regime(adx_14: pd.Series, atr_expansion_ratio: pd.Series) -> pd.Series:
    """Combined regime label per row: ``"{strong|weak}_trend_{expanding|
    normal|contracting}_vol"``. NaN wherever either input is NaN (not
    enough history yet) — never a fabricated default regime.
    """
    values = []
    for adx_value, vol_value in zip(adx_14, atr_expansion_ratio):
        if pd.isna(adx_value) or pd.isna(vol_value):
            values.append(None)
        else:
            values.append(f"{_trend_strength_label(adx_value)}_{_volatility_label(vol_value)}")
    return pd.Series(values, index=adx_14.index, dtype=object)
