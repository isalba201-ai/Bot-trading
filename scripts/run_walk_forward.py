#!/usr/bin/env python
"""Phase 7: walk-forward analysis for one strategy.

Slides a (train_span, test_span) window pair across the strategy's whole
ingested history and evaluates it on each fold's test window
independently, then reports the mean, dispersion (population stdev), and
worst fold's win rate across all sufficiently-sampled folds
(BACKTESTING.md: "a strategy that only works in one fold is not
robust").

The train window is currently unused for fitting anything — every
strategy in otc_research.strategies has fixed constructor parameters —
it only shapes each fold to match the standard walk-forward pattern; see
backtest/walkforward.py's module docstring.

Example — H4 on real EUR/USD 1h data, 30-day train / 14-day test folds:
    python scripts/run_walk_forward.py \\
        --strategy otc_research.strategies.h4_bollinger:H4BollingerMeanReversion \\
        --strategy-kwargs '{"expiry_seconds": 3600}' \\
        --pair EUR_USD --timeframe 1h \\
        --train-days 30 --test-days 14
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json

from otc_research.backtest.walkforward import generate_folds, run_walk_forward, summarize_walk_forward
from otc_research.config import load_config
from otc_research.db.models import Candle
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.utils.logging import get_logger

logger = get_logger(__name__)


def _load_strategy_class(dotted_path: str):
    module_path, _, class_name = dotted_path.partition(":")
    if not module_path or not class_name:
        raise ValueError(f"--strategy must be 'module.path:ClassName', got {dotted_path!r}")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", required=True, help="'module.path:ClassName'")
    parser.add_argument("--strategy-kwargs", default="{}", help="JSON object of constructor kwargs")
    parser.add_argument("--pair", required=True, help='e.g. "EUR_USD"')
    parser.add_argument("--timeframe", required=True, help='"1m", "5m", "15m", "1h", ...')
    parser.add_argument("--train-days", type=float, required=True)
    parser.add_argument("--test-days", type=float, required=True)
    parser.add_argument("--step-days", type=float, default=None, help="Defaults to --test-days (non-overlapping folds)")
    parser.add_argument("--scenario", default="optimistic", choices=["optimistic", "realistic", "pessimistic"])
    parser.add_argument("--min-sample-size", type=int, default=20)
    parser.add_argument("--feature-set-version", default=FEATURE_SET_VERSION)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    strategy_cls = _load_strategy_class(args.strategy)
    strategy = strategy_cls(**json.loads(args.strategy_kwargs))

    first_ts, last_ts = (
        session.query(Candle.timestamp)
        .filter(Candle.asset == args.pair, Candle.timeframe == args.timeframe)
        .order_by(Candle.timestamp.asc())
        .first(),
        session.query(Candle.timestamp)
        .filter(Candle.asset == args.pair, Candle.timeframe == args.timeframe)
        .order_by(Candle.timestamp.desc())
        .first(),
    )
    if first_ts is None or last_ts is None:
        raise SystemExit(f"No candles ingested for {args.pair}/{args.timeframe}")

    folds = generate_folds(
        first_ts[0],
        last_ts[0],
        train_span=dt.timedelta(days=args.train_days),
        test_span=dt.timedelta(days=args.test_days),
        step=dt.timedelta(days=args.step_days) if args.step_days is not None else None,
    )
    if not folds:
        raise SystemExit(
            f"No folds fit in the ingested history ({first_ts[0]} .. {last_ts[0]}) "
            f"with train={args.train_days}d test={args.test_days}d"
        )

    fold_results = run_walk_forward(
        session, strategy, args.pair, args.timeframe, config.backtest, folds,
        feature_set_version=args.feature_set_version, rng_seed=args.rng_seed,
    )

    for fr in fold_results:
        result = fr.result_for(args.scenario)
        s = result.stats if result else None
        logger.info(
            "fold=%d test_window=%s..%s n=%s win_rate=%s ci=[%s, %s]",
            fr.fold.fold_index,
            fr.fold.test_window[0].date(),
            fr.fold.test_window[1].date(),
            s.sample_size if s else "n/a",
            f"{s.win_rate:.3f}" if s and s.win_rate is not None else "n/a",
            f"{s.win_rate_ci_low:.3f}" if s and s.win_rate_ci_low is not None else "n/a",
            f"{s.win_rate_ci_high:.3f}" if s and s.win_rate_ci_high is not None else "n/a",
        )

    summary = summarize_walk_forward(fold_results, args.scenario, min_sample_size=args.min_sample_size)
    logger.info(
        "SUMMARY (%s scenario): %d/%d folds sufficiently sampled -- mean win_rate=%s "
        "stdev=%s worst_fold=#%s (%s) -- %s/%s folds showed an edge (CI lower bound > 50%%). "
        "NOT a final edge classification -- see BACKTESTING.md (Monte Carlo still needed).",
        args.scenario,
        summary.n_folds_sufficiently_sampled,
        summary.n_folds,
        f"{summary.mean_win_rate:.3f}" if summary.mean_win_rate is not None else "n/a",
        f"{summary.stdev_win_rate:.3f}" if summary.stdev_win_rate is not None else "n/a",
        summary.worst_fold_index,
        f"{summary.worst_fold_win_rate:.3f}" if summary.worst_fold_win_rate is not None else "n/a",
        summary.n_folds_with_edge,
        summary.n_folds_sufficiently_sampled,
    )


if __name__ == "__main__":
    main()
