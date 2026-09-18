"""Step 9 addendum, point 10: equity curve / drawdown / R-multiple /
streak reporting, built ENTIRELY on the ``backtest.simulator.Trade``
objects every ``run_backtest`` call already returns
(``BacktestRunResult.trades``) — no simulator or engine changes needed.

Per-trade R-multiple is the binary-options payoff in units of stake:
``+payout`` on WIN, ``-1.0`` on LOSS — never ``Trade.pnl_pct`` (that field
is the underlying's raw percentage move, a spot-Forex-style number with
no meaning to a fixed-payout binary option; using it here would silently
reintroduce exactly the "spot trading P&L" framing the binary-options
reframe correction retracted). VOID trades (signal dropped / insufficient
data at the end of history) never happened as a real option and are
excluded from every statistic below, not treated as a loss or a win.

Per the user's own instruction, this is deliberately NOT computed for
every strategy — callers should only build a report for strategies/
conditions that already cleared a "sufficiently serious" bar (this
project's choice: at least candidacy gate 2, the robustness sweep) to
avoid burning effort charting noise.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Sequence

from otc_research.backtest.simulator import Trade


@dataclass(frozen=True)
class EquityPoint:
    timestamp: dt.datetime
    trade_r: float
    cumulative_r: float


@dataclass(frozen=True)
class PerformanceReport:
    n_trades: int
    n_win: int
    n_loss: int
    n_void: int
    win_rate: float | None
    profit_factor: float | None
    cumulative_return_r: float
    avg_r_per_trade: float | None
    equity_curve: tuple[EquityPoint, ...]
    max_drawdown_r: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    first_trade_time: dt.datetime | None
    last_trade_time: dt.datetime | None
    trades_per_day: float | None


def _trade_r(trade: Trade, payout: float) -> float:
    return payout if trade.result == "WIN" else -1.0


def build_performance_report(trades: Sequence[Trade], payout: float) -> PerformanceReport:
    """``trades`` may be in any order (mirrors whatever order
    ``simulate()`` produced, which is already chronological by
    ``signal_time`` since it walks the candle series forward) — sorted
    here by ``entry_time`` defensively so the equity curve is never
    accidentally out of order.
    """
    resolved = sorted(
        (t for t in trades if t.result in ("WIN", "LOSS")),
        key=lambda t: t.entry_time,
    )
    n_void = sum(1 for t in trades if t.result == "VOID")
    n_win = sum(1 for t in resolved if t.result == "WIN")
    n_loss = len(resolved) - n_win

    if not resolved:
        return PerformanceReport(
            n_trades=0,
            n_win=0,
            n_loss=0,
            n_void=n_void,
            win_rate=None,
            profit_factor=None,
            cumulative_return_r=0.0,
            avg_r_per_trade=None,
            equity_curve=(),
            max_drawdown_r=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            first_trade_time=None,
            last_trade_time=None,
            trades_per_day=None,
        )

    equity_curve: list[EquityPoint] = []
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    gains = 0.0
    losses = 0.0
    cur_win_streak = 0
    cur_loss_streak = 0
    max_win_streak = 0
    max_loss_streak = 0

    for trade in resolved:
        r = _trade_r(trade, payout)
        cumulative += r
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
        equity_curve.append(
            EquityPoint(timestamp=trade.entry_time, trade_r=r, cumulative_r=cumulative)
        )
        if r > 0:
            gains += r
            cur_win_streak += 1
            cur_loss_streak = 0
        else:
            losses += -r
            cur_loss_streak += 1
            cur_win_streak = 0
        max_win_streak = max(max_win_streak, cur_win_streak)
        max_loss_streak = max(max_loss_streak, cur_loss_streak)

    first_time = resolved[0].entry_time
    last_time = resolved[-1].entry_time
    span_days = (last_time - first_time).total_seconds() / 86400.0
    trades_per_day = len(resolved) / span_days if span_days > 0 else None

    return PerformanceReport(
        n_trades=len(resolved),
        n_win=n_win,
        n_loss=n_loss,
        n_void=n_void,
        win_rate=n_win / len(resolved),
        profit_factor=(gains / losses) if losses > 0 else (float("inf") if gains > 0 else None),
        cumulative_return_r=cumulative,
        avg_r_per_trade=cumulative / len(resolved),
        equity_curve=tuple(equity_curve),
        max_drawdown_r=max_drawdown,
        max_consecutive_wins=max_win_streak,
        max_consecutive_losses=max_loss_streak,
        first_trade_time=first_time,
        last_trade_time=last_time,
        trades_per_day=trades_per_day,
    )
