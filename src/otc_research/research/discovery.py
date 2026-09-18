"""Stage 5 (+ the logging/FDR-correction discipline from plan points 7/13):
systematic 2-3 way condition search over TRAIN rows only.

Every combination tried is logged to ``ConditionTrial`` BEFORE any
correction or selection happens — this is what makes the search auditable
rather than an invisible fishing expedition, the fine-grained extension of
``Hypothesis``'s registration discipline. Benjamini-Hochberg FDR
correction (not Bonferroni) is applied across every trial from one run,
because a grid of many correlated bin combinations makes Bonferroni
needlessly conservative — it would likely reject a real effect along with
the noise.

This module has no opinion about which split it's given — same
separation of concerns as baseline.py. The caller (the step 7 research
run) is responsible for only ever passing TRAIN rows in here.
"""

from __future__ import annotations

import itertools
import json
import uuid
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from sqlalchemy.orm import Session

from otc_research.db.models import ConditionTrial
from otc_research.research.baseline import (
    WinRateStat,
    two_sided_binomial_p_value,
    win_rate_stat,
)

#: Quantile bins per feature in an interaction search — coarser than
#: baseline.py's per-feature default of 10, since the combination count
#: grows as n_bins**combo_size across every feature pair/triple.
DEFAULT_N_BINS = 4


@dataclass(frozen=True)
class Condition:
    """One combination of (feature, bin_low, bin_high) parts, ANDed
    together — e.g. "rsi_14 in (30, 45] AND adx_14 in (25, 40]".
    """

    parts: tuple[tuple[str, float, float], ...]

    def matches(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for feature, low, high in self.parts:
            mask &= (df[feature] > low) & (df[feature] <= high)
        return mask

    def matches_row(self, features: dict) -> bool:
        """Scalar version of ``matches``, for a single point-in-time
        feature dict — used by ``research.condition_strategy.
        ConditionStrategy.decide`` to run a discovered condition through
        the Phase 4 backtest engine unchanged.
        """
        return all(low < features[feature] <= high for feature, low, high in self.parts)

    def label(self) -> str:
        return " AND ".join(f"{f}∈({lo:.4g},{hi:.4g}]" for f, lo, hi in self.parts)

    def to_json(self) -> str:
        return json.dumps([{"feature": f, "low": lo, "high": hi} for f, lo, hi in self.parts])

    @staticmethod
    def from_json(condition_json: str) -> "Condition":
        """Inverse of ``to_json`` — reconstructs a ``Condition`` from a
        stored ``ConditionTrial.condition_json`` value, used by anything
        that re-evaluates a past trial (e.g. the Step 8 execution-
        decomposition tool) without re-running discovery.
        """
        parts = tuple(
            (item["feature"], item["low"], item["high"]) for item in json.loads(condition_json)
        )
        return Condition(parts=parts)


def _quantile_bin_edges(series: pd.Series, n_bins: int) -> list[tuple[float, float]]:
    try:
        cut = pd.qcut(series.dropna(), q=n_bins, duplicates="drop")
    except ValueError:
        return []
    edges = sorted({(interval.left, interval.right) for interval in cut.cat.categories})
    return edges


def generate_conditions(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    combo_size: int,
    n_bins: int = DEFAULT_N_BINS,
) -> list[Condition]:
    """All ``combo_size``-way combinations of distinct features from
    ``feature_cols``, each quantile-binned into ``n_bins``, producing one
    ``Condition`` per bin combination — the exhaustive grid for one
    interaction order. Callers are expected to have already restricted
    ``feature_cols`` to a deliberately chosen subset (not every v4
    feature at once) to keep the combinatorics tractable — this function
    does not second-guess that choice.
    """
    bins_by_feature = {f: _quantile_bin_edges(df[f], n_bins) for f in feature_cols}
    conditions: list[Condition] = []
    for combo in itertools.combinations(feature_cols, combo_size):
        bin_choices = [bins_by_feature[f] for f in combo]
        if any(len(b) == 0 for b in bin_choices):
            continue
        for bin_combo in itertools.product(*bin_choices):
            parts = tuple((f, lo, hi) for f, (lo, hi) in zip(combo, bin_combo))
            conditions.append(Condition(parts=parts))
    return conditions


@dataclass(frozen=True)
class DiscoveryResult:
    condition: Condition
    stat: WinRateStat
    p_value: float | None
    fdr_significant: bool


def _benjamini_hochberg(p_values: Sequence[float | None], *, q: float) -> list[bool]:
    """Standard BH step-up procedure. Trials with no p-value (excluded for
    being underpowered — see ``run_discovery``'s ``min_sample_size``) are
    always ``False``: not tested, not "not significant".
    """
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    flags = [False] * len(p_values)
    m = len(indexed)
    if m == 0:
        return flags

    indexed.sort(key=lambda pair: pair[1])
    largest_k = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        if p <= (rank / m) * q:
            largest_k = rank

    for rank, (original_index, _) in enumerate(indexed, start=1):
        if rank <= largest_k:
            flags[original_index] = True
    return flags


def run_discovery(
    session: Session,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    asset: str,
    timeframe: str,
    feature_set_version: str,
    combo_sizes: tuple[int, ...] = (2, 3),
    n_bins: int = DEFAULT_N_BINS,
    min_sample_size: int = 30,
    fdr_q: float = 0.05,
    regime: str | None = None,
    run_id: str | None = None,
    conditions: Sequence[Condition] | None = None,
) -> list[DiscoveryResult]:
    """Runs the interaction search over ``df`` (caller's responsibility:
    TRAIN rows only) for one target column, logs every combination tried
    to ``ConditionTrial``, applies Benjamini-Hochberg FDR correction
    across all of them, and returns results sorted by p-value (untested/
    underpowered trials last). ``regime`` is metadata only, recorded on
    each row — pass an already regime-filtered ``df`` if you want a
    regime-scoped search; this function does not filter by regime itself.

    ``conditions``, if given, is used VERBATIM instead of generating the
    quantile-binned grid from ``feature_cols``/``combo_sizes``/``n_bins``
    (which are then ignored) — added for the Step 9 addendum's H9
    (session-bias) restricted discovery, which needs one bin per literal
    hour value (0..23), not a quantile split of ``hour_utc`` that could
    land bin edges off integer-hour boundaries. Every other discipline
    (trial logging, BH-FDR correction, p-value sort) is identical either
    way — this is a substitution of WHICH conditions are tried, not a
    lighter-weight path.
    """
    run_id = run_id or str(uuid.uuid4())
    scoped = df if regime is None else df[df["regime"] == regime]

    if conditions is not None:
        all_conditions: list[Condition] = list(conditions)
    else:
        all_conditions = []
        for size in combo_sizes:
            all_conditions.extend(
                generate_conditions(scoped, feature_cols, combo_size=size, n_bins=n_bins)
            )

    trial_rows: list[tuple[Condition, WinRateStat, float | None]] = []
    trial_orm_rows: list[ConditionTrial] = []
    for condition in all_conditions:
        mask = condition.matches(scoped)
        stat = win_rate_stat(scoped.loc[mask, target_col])
        p_value = (
            two_sided_binomial_p_value(stat.wins, stat.n) if stat.n >= min_sample_size else None
        )
        trial_rows.append((condition, stat, p_value))

        orm_row = ConditionTrial(
            run_id=run_id,
            asset=asset,
            timeframe=timeframe,
            feature_set_version=feature_set_version,
            target_col=target_col,
            regime=regime,
            condition_json=condition.to_json(),
            n_features_combined=len(condition.parts),
            sample_size=stat.n,
            wins=stat.wins,
            win_rate=stat.win_rate,
            win_rate_ci_low=stat.ci_low,
            win_rate_ci_high=stat.ci_high,
            p_value=p_value,
        )
        session.add(orm_row)
        trial_orm_rows.append(orm_row)
    session.commit()

    significant_flags = _benjamini_hochberg([p for _, _, p in trial_rows], q=fdr_q)
    for orm_row, significant in zip(trial_orm_rows, significant_flags):
        orm_row.fdr_significant = significant
    session.commit()

    results = [
        DiscoveryResult(condition=c, stat=s, p_value=p, fdr_significant=sig)
        for (c, s, p), sig in zip(trial_rows, significant_flags)
    ]
    results.sort(key=lambda r: (r.p_value is None, r.p_value if r.p_value is not None else 1.0))
    return results
