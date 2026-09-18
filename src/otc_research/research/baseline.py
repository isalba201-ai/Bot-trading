"""Stages 1-4 (plus the bootstrap utility for stage 6) of the statistical
discovery methodology — approved plan point 4: unconditional baseline,
conditional-on-one-feature (binned), contingency tables for categorical
features, and regime-conditional versions of the same. All BEFORE any
interaction search (discovery.py) or ML (models.py).

Every result here is Wilson-CI-grounded — never a bare win rate — reusing
``backtest.metrics.wilson_confidence_interval`` exactly as the rest of
this codebase does. Callers are responsible for only ever passing TRAIN-
split rows in here during exploration (see discovery.py); this module has
no opinion about which split it's given, same separation of concerns as
``backtest.metrics``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from otc_research.backtest.metrics import wilson_confidence_interval


@dataclass(frozen=True)
class WinRateStat:
    n: int
    wins: int
    win_rate: float | None
    ci_low: float | None
    ci_high: float | None


def win_rate_stat(target: pd.Series) -> WinRateStat:
    """``target`` is a 0/1/NaN series (NaN = unresolved/tie, already
    excluded — see dataset.py). Never fabricates a win rate/CI for n=0.
    """
    resolved = target.dropna()
    n = len(resolved)
    if n == 0:
        return WinRateStat(n=0, wins=0, win_rate=None, ci_low=None, ci_high=None)
    wins = int(resolved.sum())
    win_rate = wins / n
    ci = wilson_confidence_interval(wins, n)
    ci_low, ci_high = ci if ci is not None else (None, None)
    return WinRateStat(n=n, wins=wins, win_rate=win_rate, ci_low=ci_low, ci_high=ci_high)


def unconditional_baseline(df: pd.DataFrame, target_col: str) -> WinRateStat:
    """Stage 1: P(target_col wins) with no conditioning at all — the
    sanity check everything else is compared against.
    """
    return win_rate_stat(df[target_col])


@dataclass(frozen=True)
class BinStat:
    feature: str
    bin_low: float
    bin_high: float
    stat: WinRateStat


def conditional_by_bins(
    df: pd.DataFrame, feature_col: str, target_col: str, n_bins: int = 10
) -> list[BinStat]:
    """Stage 2: quantile-bins ``feature_col`` (deciles by default) and
    reports win-rate stats for ``target_col`` within each bin. Bin edges
    are computed from THIS dataframe's own feature distribution — pass
    only TRAIN rows during exploration (discovery.py enforces this at the
    call site, this function has no built-in split awareness). Returns an
    empty list if there aren't enough distinct values to form bins.
    """
    valid = df[[feature_col, target_col]].dropna(subset=[feature_col])
    if valid.empty:
        return []
    try:
        binned = pd.qcut(valid[feature_col], q=n_bins, duplicates="drop")
    except ValueError:
        return []

    results = []
    for interval, group in valid.groupby(binned, observed=True):
        stat = win_rate_stat(group[target_col])
        results.append(
            BinStat(
                feature=feature_col,
                bin_low=float(interval.left),
                bin_high=float(interval.right),
                stat=stat,
            )
        )
    return sorted(results, key=lambda b: b.bin_low)


@dataclass(frozen=True)
class CategoryStat:
    feature: str
    category: str
    stat: WinRateStat


def conditional_by_category(
    df: pd.DataFrame, category_col: str, target_col: str
) -> list[CategoryStat]:
    """Stage 3: contingency-table-style win rate per category of a
    categorical column (e.g. ``regime``, ``trading_session_code``,
    ``day_of_week``).
    """
    valid = df[[category_col, target_col]].dropna(subset=[category_col])
    results = []
    for category, group in valid.groupby(category_col, observed=True):
        results.append(
            CategoryStat(feature=category_col, category=str(category), stat=win_rate_stat(group[target_col]))
        )
    return results


def regime_conditional_bins(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    *,
    regime_col: str = "regime",
    n_bins: int = 10,
) -> dict[str, list[BinStat]]:
    """Stage 4: repeats ``conditional_by_bins`` independently within each
    regime — whether a feature's relationship with the target holds up,
    weakens, or reverses depending on market regime.
    """
    result: dict[str, list[BinStat]] = {}
    valid = df.dropna(subset=[regime_col])
    for regime, group in valid.groupby(regime_col, observed=True):
        result[str(regime)] = conditional_by_bins(group, feature_col, target_col, n_bins=n_bins)
    return result


def two_sided_binomial_p_value(wins: int, n: int, null_p: float = 0.5) -> float | None:
    """Two-sided p-value for H0: true win rate == ``null_p``, via the
    normal approximation to the binomial (score-test form: standard error
    computed under the null, not the plug-in sample proportion — the more
    accurate choice for a proportion test, same reasoning that motivates
    Wilson over the naive interval). Deliberately dependency-free (no
    scipy) — this project only adds a dependency when a stage genuinely
    needs it (see research/models.py's scikit-learn for stage 8).

    Used by discovery.py to rank/threshold condition trials before
    Benjamini-Hochberg FDR correction. Returns None for n == 0 — never a
    fabricated p-value.
    """
    if n <= 0:
        return None
    if not 0.0 < null_p < 1.0:
        raise ValueError("null_p must be between 0 and 1")
    p_hat = wins / n
    standard_error = math.sqrt(null_p * (1.0 - null_p) / n)
    if standard_error == 0.0:
        return None
    z = (p_hat - null_p) / standard_error
    cdf = 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))
    p_value = 2.0 * (1.0 - cdf)
    return min(1.0, max(0.0, p_value))


def bootstrap_mean_ci(
    values: Sequence[float], *, n_resamples: int = 1000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float] | None:
    """Stage 6: percentile bootstrap CI for the mean of ``values`` (e.g.
    per-trade payout-adjusted expectancy) — used where a closed-form CI
    isn't available, the way Wilson's is for a win rate. Seeded for
    reproducibility, same convention as every other RNG in this codebase.
    Returns None for an empty input rather than fabricating a CI.
    """
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
    if len(arr) == 0:
        return None
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    resampled_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        resampled_means[i] = sample.mean()

    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(resampled_means, alpha))
    high = float(np.quantile(resampled_means, 1.0 - alpha))
    return low, high
