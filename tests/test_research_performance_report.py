import datetime as dt

from otc_research.backtest.simulator import Trade
from otc_research.research.performance_report import build_performance_report

T0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def _trade(i: int, result: str) -> Trade:
    return Trade(
        signal_time=T0 + dt.timedelta(minutes=i),
        direction="CALL",
        expiry_seconds=60,
        result=result,
        entry_time=T0 + dt.timedelta(minutes=i + 1),
        entry_price=1.1000,
        exit_time=T0 + dt.timedelta(minutes=i + 2),
        exit_price=1.1005 if result == "WIN" else 1.0995,
        pnl_pct=0.05 if result == "WIN" else -0.05,
    )


def test_empty_trade_list_returns_empty_report():
    report = build_performance_report([], payout=0.85)
    assert report.n_trades == 0
    assert report.win_rate is None
    assert report.profit_factor is None
    assert report.cumulative_return_r == 0.0
    assert report.max_drawdown_r == 0.0
    assert report.equity_curve == ()


def test_void_trades_excluded_from_every_statistic():
    trades = [
        _trade(0, "WIN"),
        Trade(
            signal_time=T0 + dt.timedelta(minutes=5),
            direction="PUT",
            expiry_seconds=60,
            result="VOID",
            void_reason="signal_dropped",
        ),
        _trade(10, "LOSS"),
    ]
    report = build_performance_report(trades, payout=0.85)
    assert report.n_trades == 2  # VOID not counted
    assert report.n_void == 1
    assert report.n_win == 1
    assert report.n_loss == 1


def test_known_equity_curve_and_drawdown_hand_computed():
    # Sequence: WIN, WIN, LOSS, LOSS, LOSS, WIN -- payout=0.80.
    # R per trade: +0.8, +0.8, -1, -1, -1, +0.8
    # Cumulative: 0.8, 1.6, 0.6, -0.4, -1.4, -0.6
    # Peak-so-far: 0.8, 1.6, 1.6, 1.6, 1.6, 1.6
    # Drawdown (peak-cum): 0, 0, 1.0, 2.0, 3.0, 2.2 -> max_drawdown = 3.0
    payout = 0.80
    results = ["WIN", "WIN", "LOSS", "LOSS", "LOSS", "WIN"]
    trades = [_trade(i * 5, r) for i, r in enumerate(results)]
    report = build_performance_report(trades, payout=payout)

    assert report.n_trades == 6
    assert report.n_win == 3
    assert report.n_loss == 3
    assert report.win_rate == 0.5

    expected_cum = [0.8, 1.6, 0.6, -0.4, -1.4, -0.6]
    actual_cum = [round(p.cumulative_r, 6) for p in report.equity_curve]
    assert actual_cum == [round(x, 6) for x in expected_cum]

    assert round(report.cumulative_return_r, 6) == round(-0.6, 6)
    assert round(report.max_drawdown_r, 6) == round(3.0, 6)
    assert round(report.avg_r_per_trade, 6) == round(-0.6 / 6, 6)

    # profit factor: gains = 0.8*3 = 2.4, losses = 1*3 = 3.0 -> 0.8
    assert round(report.profit_factor, 6) == round(2.4 / 3.0, 6)


def test_streaks_computed_correctly():
    # WIN WIN WIN LOSS WIN LOSS LOSS LOSS LOSS WIN
    results = ["WIN", "WIN", "WIN", "LOSS", "WIN", "LOSS", "LOSS", "LOSS", "LOSS", "WIN"]
    trades = [_trade(i * 5, r) for i, r in enumerate(results)]
    report = build_performance_report(trades, payout=0.85)
    assert report.max_consecutive_wins == 3
    assert report.max_consecutive_losses == 4


def test_all_wins_gives_none_profit_factor_denominator_handled():
    trades = [_trade(i * 5, "WIN") for i in range(4)]
    report = build_performance_report(trades, payout=0.85)
    assert report.profit_factor == float("inf")
    assert report.max_drawdown_r == 0.0


def test_trades_per_day_computed_from_entry_time_span():
    # 4 trades spanning exactly 2 days (entry_time), 1 win 3 loss doesn't
    # matter for this check -- only timing.
    trades = [
        Trade(
            signal_time=T0,
            direction="CALL",
            expiry_seconds=60,
            result="WIN",
            entry_time=T0,
            entry_price=1.1,
            exit_time=T0,
            exit_price=1.1005,
            pnl_pct=0.05,
        ),
        Trade(
            signal_time=T0,
            direction="CALL",
            expiry_seconds=60,
            result="LOSS",
            entry_time=T0 + dt.timedelta(days=2),
            entry_price=1.1,
            exit_time=T0,
            exit_price=1.0995,
            pnl_pct=-0.05,
        ),
    ]
    report = build_performance_report(trades, payout=0.85)
    assert report.trades_per_day == 1.0  # 2 trades / 2 days
