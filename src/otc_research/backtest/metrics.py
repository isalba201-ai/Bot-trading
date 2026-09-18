"""Statistics for a batch of trade outcomes.

This is where SIGNAL_ENGINE.md's non-negotiable rule gets implemented in
code: "NIVEL DE CONFIANZA ESTIMADO is never invented... reported together
with sample size [and] confidence interval (e.g. Wilson or
Clopper-Pearson at 95%, not just a point estimate)". Every function here
either returns a properly-grounded number or ``None`` — never a
point-estimate with no sample size/CI behind it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

# z-score for a two-sided 95% confidence interval.
Z_95 = 1.959963984540054


def wilson_confidence_interval(wins: int, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion. Preferred over the
    naive normal approximation because it stays well-behaved (doesn't
    exceed [0, 1], doesn't collapse to zero width) even for small samples
    or proportions near 0 or 1 — exactly the case that matters here, since
    a real backtest's sample size is often small enough that the naive
    interval would be misleadingly narrow.

    Returns None (not a fabricated interval) when n == 0.
    """
    if n <= 0:
        return None
    if wins < 0 or wins > n:
        raise ValueError("wins must be between 0 and n")

    p_hat = wins / n
    denom = 1.0 + z * z / n
    center = p_hat + z * z / (2 * n)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))

    low = (center - margin) / denom
    high = (center + margin) / denom
    return max(0.0, low), min(1.0, high)


def break_even_win_rate(payout: float) -> float:
    """Minimum win rate needed to break even on a FIXED-PAYOUT instrument
    (binary options, not plain spot Forex) — BACKTESTING.md's
    ``break_even_win_rate = 1 / (1 + payout)``. ``payout`` is the
    fractional profit on a win (e.g. ``0.85`` for a broker paying 85% on
    a win); a loss is assumed to forfeit the full stake, the standard
    binary options structure. This is NOT 50% — a typical ~0.80-0.90
    payout means the real bar is roughly 53-56%, not 50%, which is why
    every "does it clear 50%?" read elsewhere in this codebase is a
    necessary but not sufficient condition for a binary-options trader
    specifically.
    """
    if payout <= 0:
        raise ValueError("payout must be positive")
    return 1.0 / (1.0 + payout)


def payout_adjusted_expectancy(win_rate: float, payout: float) -> float:
    """Expected return per trade as a fraction of stake, for a
    fixed-payout instrument — BACKTESTING.md's
    ``expectancy_per_trade = win_rate * payout - loss_rate``. Positive
    means the strategy clears THIS payout; negative means it doesn't.
    Linear in ``win_rate``, so it can be applied directly to Wilson CI
    endpoints too (see ``payout_adjusted_expectancy_ci``) without
    distorting the interval.
    """
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError("win_rate must be between 0 and 1")
    if payout <= 0:
        raise ValueError("payout must be positive")
    loss_rate = 1.0 - win_rate
    return win_rate * payout - loss_rate


def payout_adjusted_expectancy_ci(
    win_rate_ci_low: float, win_rate_ci_high: float, payout: float
) -> tuple[float, float]:
    """Maps a win-rate confidence interval through the same linear payout
    formula. Because the transform is linear and increasing in
    ``win_rate``, the endpoints map directly — the low end of the win-rate
    CI maps to the low end of the expectancy CI, and likewise the high
    end, with no distortion.
    """
    return (
        payout_adjusted_expectancy(win_rate_ci_low, payout),
        payout_adjusted_expectancy(win_rate_ci_high, payout),
    )


@dataclass(frozen=True)
class TradeStats:
    sample_size: int  # resolved trades only (VOID excluded — see below)
    wins: int
    losses: int
    voided: int
    win_rate: float | None
    win_rate_ci_low: float | None
    win_rate_ci_high: float | None
    expectancy_pct: float | None  # mean pnl_pct across resolved trades


def summarize_trades(results: Sequence[str], pnl_pcts: Sequence[float]) -> TradeStats:
    """``results`` is a sequence of "WIN"/"LOSS"/"VOID" (see
    simulator.Trade.result), ``pnl_pcts`` the matching per-trade price
    return. VOID trades (execution dropped the signal, or the data needed
    to resolve the expiry wasn't available) are excluded from
    ``sample_size``/win rate/expectancy entirely — a VOID trade is not a
    loss, it's a trade that never happened, and counting it either way
    would misstate the strategy's real performance.
    """
    if len(results) != len(pnl_pcts):
        raise ValueError("results and pnl_pcts must be the same length")

    wins = sum(1 for r in results if r == "WIN")
    losses = sum(1 for r in results if r == "LOSS")
    voided = sum(1 for r in results if r == "VOID")
    sample_size = wins + losses

    win_rate = wins / sample_size if sample_size > 0 else None
    ci = wilson_confidence_interval(wins, sample_size)
    ci_low, ci_high = ci if ci is not None else (None, None)

    resolved_pnls = [pnl for r, pnl in zip(results, pnl_pcts) if r in ("WIN", "LOSS")]
    expectancy_pct = (sum(resolved_pnls) / len(resolved_pnls)) if resolved_pnls else None

    return TradeStats(
        sample_size=sample_size,
        wins=wins,
        losses=losses,
        voided=voided,
        win_rate=win_rate,
        win_rate_ci_low=ci_low,
        win_rate_ci_high=ci_high,
        expectancy_pct=expectancy_pct,
    )
