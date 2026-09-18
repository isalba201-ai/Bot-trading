"""Step 9 addendum: builds the ``(base_strategy, strategy_factory,
perturbation_param_grid)`` triple ``research.candidacy.evaluate_candidacy``
needs for each hand-designed H1-H20 strategy, so they can go through the
exact same four-gate funnel already applied to discovered conditions —
including gate 2's parameter-sensitivity sweep.

Each H1-H20 class already exposes its own threshold(s) as named
constructor kwargs (see ``strategies/h*.py``); this module's job is only
to say, per strategy code, WHICH of those kwargs are the "edge" a
robustness sweep should perturb, and HOW (continuous widen/narrow, mirrors
``research.condition_strategy.perturb_condition``'s own "widen by pct of
the bin's width" semantics; or, for an integer-count threshold like H1's
``min_streak``, the discrete analogue of the same idea: try the adjacent
integers). Strategies with no threshold parameter at all (H5, H11, H14,
H15 — a pure sign/crossover trigger) get a single-point grid: gate 2 then
measures robustness only via the walk-forward-independent Wilson-CI check
at that one point, which is a real if weaker check (documented, not
silently skipped) — there is no threshold to perturb because the
hypothesis itself has none.

H9 (session bias) is deliberately NOT built here — its 24-hour x
2-direction search space needs its own FDR-corrected screening, reusing
``research.discovery`` restricted to the single ``hour_utc`` feature (see
``scripts/run_binary_options_backtest.py``), not a one-off perturbation
grid.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from otc_research.backtest.strategy import Strategy
from otc_research.strategies import (
    H1StreakContinuation,
    H2ExtremeRangeReversion,
    H3MomentumContinuation,
    H4BollingerMeanReversion,
    H5DonchianBreakout,
    H6WickRejection,
    H7RsiExtremeConfirmed,
    H8VolatilityExpansion,
    H10CombinedTrendStructureMomentum,
    H11MacdCrossContinuation,
    H12CciExtremeReversion,
    H13RciExtremeReversion,
    H14EngulfingReversal,
    H15InsideBarBreakout,
    H16MacdRsiConfirmed,
    H17BollingerRciConfirmed,
    H18CciEngulfingConfirmed,
    H19TrendPullback,
    H20BreakoutVolatilityConfirmed,
    H21CciRsiMacdBearishReversal,
)

#: Default edge-perturbation grid, identical to
#: ``candidacy.CandidacyThresholds.edge_perturbation_grid`` -- kept as an
#: independent constant here (not imported) so this module has no
#: dependency on candidacy.py, only the other way around.
DEFAULT_PERTURBATION_GRID: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)


def _mult(value: float, pct: float, *, min_value: float | None = None) -> float:
    """``value`` scaled by ``(1 + pct)`` -- a magnitude threshold read
    further from zero (widened, pct > 0) or closer to zero (narrowed,
    pct < 0), clamped away from a degenerate/invalid bound if given.
    """
    new_value = value * (1.0 + pct)
    if min_value is not None:
        new_value = max(new_value, min_value)
    return new_value


def _widen_band(
    low: float, high: float, pct: float, *, floor: float | None = None, ceil: float | None = None
) -> tuple[float, float]:
    """Mirrors ``research.condition_strategy.perturb_condition``: widens
    (pct > 0) or narrows (pct < 0) a [low, high] band symmetrically by
    pct of its own width.
    """
    width = high - low
    new_low = low - width * pct
    new_high = high + width * pct
    if floor is not None:
        new_low = max(new_low, floor)
    if ceil is not None:
        new_high = min(new_high, ceil)
    return new_low, new_high


def _single_point_grid() -> list[dict[str, object]]:
    """For a threshold-free strategy (H5/H11/H14/H15): one grid point, no
    perturbation possible -- see module docstring.
    """
    return [{}]


def _build(code: str, cls: type, base_kwargs: dict, param_grid: list[dict]):
    def strategy_factory(**kwargs) -> Strategy:
        return cls(**{**base_kwargs, **kwargs})

    base_strategy = strategy_factory(**param_grid[len(param_grid) // 2])
    return base_strategy, strategy_factory, param_grid


def build_candidacy_inputs(
    code: str, expiry_seconds: int, *, perturbation_grid: Sequence[float] = DEFAULT_PERTURBATION_GRID
) -> tuple[Strategy, Callable[..., Strategy], list[Mapping[str, object]]]:
    """Returns ``(base_strategy, strategy_factory, perturbation_param_grid)``
    for hypothesis ``code`` (H1-H8, H10-H20 -- not H9, see module
    docstring), at the strategy's own registered default parameter values
    (STRATEGIES.md), evaluated at ``expiry_seconds`` (overriding each
    class's own ``expiry_seconds=300`` constructor default, per the
    expiry-universe correction: no duration is assumed a priori, every
    strategy is swept across the same h-derived expiry set as discovered
    conditions).
    """
    pcts = list(perturbation_grid)
    ez = {"expiry_seconds": expiry_seconds}

    if code == "H1":
        base = H1StreakContinuation()
        grid = [{"min_streak": max(1, round(base.min_streak * (1 + pct)))} for pct in pcts]
        return _build(code, H1StreakContinuation, ez, grid)

    if code == "H2":
        base = H2ExtremeRangeReversion()
        grid = [
            {"range_ratio_threshold": _mult(base.range_ratio_threshold, pct, min_value=1.01)}
            for pct in pcts
        ]
        return _build(code, H2ExtremeRangeReversion, ez, grid)

    if code == "H3":
        base = H3MomentumContinuation()
        grid = [
            {
                "roc_threshold": _mult(base.roc_threshold, pct, min_value=1e-6),
                "slope_threshold": _mult(base.slope_threshold, pct, min_value=1e-8),
            }
            for pct in pcts
        ]
        return _build(code, H3MomentumContinuation, ez, grid)

    if code == "H4":
        base = H4BollingerMeanReversion()
        grid = []
        for pct in pcts:
            low, high = _widen_band(base.lower_pct_b, base.upper_pct_b, pct)
            grid.append({"lower_pct_b": low, "upper_pct_b": high})
        return _build(code, H4BollingerMeanReversion, ez, grid)

    if code == "H5":
        return _build(code, H5DonchianBreakout, ez, _single_point_grid())

    if code == "H6":
        base = H6WickRejection()
        grid = [
            {"wick_ratio_threshold": min(0.95, max(0.05, _mult(base.wick_ratio_threshold, pct)))}
            for pct in pcts
        ]
        return _build(code, H6WickRejection, ez, grid)

    if code == "H7":
        base = H7RsiExtremeConfirmed()
        grid = []
        for pct in pcts:
            low, high = _widen_band(base.oversold, base.overbought, pct, floor=0.0, ceil=100.0)
            grid.append({"oversold": low, "overbought": high})
        return _build(code, H7RsiExtremeConfirmed, ez, grid)

    if code == "H8":
        base = H8VolatilityExpansion()
        grid = [
            {
                "expansion_threshold": _mult(base.expansion_threshold, pct, min_value=1.01),
                "slope_threshold": _mult(base.slope_threshold, pct, min_value=1e-8),
            }
            for pct in pcts
        ]
        return _build(code, H8VolatilityExpansion, ez, grid)

    if code == "H10":
        base = H10CombinedTrendStructureMomentum()
        grid = [
            {
                "slope_threshold": _mult(base.slope_threshold, pct, min_value=1e-8),
                "roc_threshold": _mult(base.roc_threshold, pct, min_value=1e-6),
            }
            for pct in pcts
        ]
        return _build(code, H10CombinedTrendStructureMomentum, ez, grid)

    if code == "H11":
        return _build(code, H11MacdCrossContinuation, ez, _single_point_grid())

    if code == "H12":
        base = H12CciExtremeReversion()
        grid = [{"threshold": _mult(base.threshold, pct, min_value=1.0)} for pct in pcts]
        return _build(code, H12CciExtremeReversion, ez, grid)

    if code == "H13":
        base = H13RciExtremeReversion()
        grid = [
            {"threshold": min(100.0, max(1.0, _mult(base.threshold, pct)))} for pct in pcts
        ]
        return _build(code, H13RciExtremeReversion, ez, grid)

    if code == "H14":
        return _build(code, H14EngulfingReversal, ez, _single_point_grid())

    if code == "H15":
        return _build(code, H15InsideBarBreakout, ez, _single_point_grid())

    if code == "H16":
        base = H16MacdRsiConfirmed()
        grid = []
        for pct in pcts:
            mid, ceil_ = _widen_band(base.rsi_midpoint, base.rsi_ceiling, pct, floor=0.0, ceil=100.0)
            grid.append({"rsi_midpoint": mid, "rsi_ceiling": ceil_})
        return _build(code, H16MacdRsiConfirmed, ez, grid)

    if code == "H17":
        base = H17BollingerRciConfirmed()
        grid = []
        for pct in pcts:
            low, high = _widen_band(base.lower_pct_b, base.upper_pct_b, pct)
            grid.append(
                {
                    "lower_pct_b": low,
                    "upper_pct_b": high,
                    "rci_threshold": min(100.0, max(1.0, _mult(base.rci_threshold, pct))),
                }
            )
        return _build(code, H17BollingerRciConfirmed, ez, grid)

    if code == "H18":
        base = H18CciEngulfingConfirmed()
        grid = [{"cci_threshold": _mult(base.cci_threshold, pct, min_value=1.0)} for pct in pcts]
        return _build(code, H18CciEngulfingConfirmed, ez, grid)

    if code == "H19":
        base = H19TrendPullback()
        grid = []
        for pct in pcts:
            low, high = _widen_band(base.pullback_low, base.pullback_high, pct, floor=0.0, ceil=50.0)
            grid.append(
                {
                    "slope_threshold": _mult(base.slope_threshold, pct, min_value=1e-8),
                    "pullback_low": low,
                    "pullback_high": high,
                }
            )
        return _build(code, H19TrendPullback, ez, grid)

    if code == "H20":
        base = H20BreakoutVolatilityConfirmed()
        grid = [
            {"expansion_threshold": _mult(base.expansion_threshold, pct, min_value=1.01)}
            for pct in pcts
        ]
        return _build(code, H20BreakoutVolatilityConfirmed, ez, grid)

    if code == "H21":
        base = H21CciRsiMacdBearishReversal()
        grid = []
        for pct in pcts:
            grid.append({
                "cci_threshold": _mult(base.cci_threshold, pct, min_value=1.0),
                "rsi_threshold": min(99.0, max(51.0, _mult(base.rsi_threshold, pct))),
            })
        return _build(code, H21CciRsiMacdBearishReversal, ez, grid)

    raise ValueError(f"no candidacy-input builder for code {code!r} (H9 is handled separately)")


#: Every code this module can build inputs for -- H9 deliberately excluded,
#: matching ``strategies.BASELINE_STRATEGIES``.
SUPPORTED_CODES: tuple[str, ...] = (
    "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H10", "H11", "H12",
    "H13", "H14", "H15", "H16", "H17", "H18", "H19", "H20", "H21",
)
