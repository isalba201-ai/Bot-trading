#!/usr/bin/env python
"""Run a strategy through the Phase 4 backtesting engine.

No hypothesis strategy exists yet (Phase 5 — see STRATEGIES.md); this CLI
is ready for when they do. Point ``--strategy`` at a dotted
``module:ClassName`` for a zero-argument-constructible class implementing
otc_research.backtest.strategy.Strategy, e.g.:

    python scripts/run_backtest.py --strategy otc_research.strategies.h3:H3Momentum \\
        --pair EUR_USD --timeframe 5m --split train

Only ``--split train`` and ``--split validation`` are safe to run
repeatedly during development. ``--split test`` is BACKTESTING.md's
out-of-sample split — touch it exactly once, after every development
decision is already final; this script logs a loud warning every time it
is used, it does not block it.
"""

from __future__ import annotations

import argparse
import importlib

from otc_research.backtest.engine import run_backtest
from otc_research.backtest.strategy import Strategy
from otc_research.config import load_config
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.utils.logging import get_logger

logger = get_logger(__name__)


def _load_strategy(dotted_path: str) -> Strategy:
    module_path, _, class_name = dotted_path.partition(":")
    if not module_path or not class_name:
        raise ValueError(f"--strategy must be 'module.path:ClassName', got {dotted_path!r}")
    module = importlib.import_module(module_path)
    strategy_cls = getattr(module, class_name)
    return strategy_cls()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", required=True, help="'module.path:ClassName'")
    parser.add_argument("--pair", required=True, help='e.g. "EUR_USD"')
    parser.add_argument("--timeframe", required=True, help='"1m", "5m", "15m", ...')
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--feature-set-version", default=FEATURE_SET_VERSION)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    strategy = _load_strategy(args.strategy)

    results = run_backtest(
        session,
        strategy,
        asset=args.pair,
        timeframe=args.timeframe,
        backtest_config=config.backtest,
        feature_set_version=args.feature_set_version,
        split=args.split,
        rng_seed=args.rng_seed,
    )

    for result in results:
        s = result.stats
        logger.info(
            "%s/%s [%s/%s] n=%d win_rate=%s ci=[%s, %s] expectancy_pct=%s (voided=%d) backtest_run_id=%d",
            args.pair,
            args.timeframe,
            args.split,
            result.scenario,
            s.sample_size,
            f"{s.win_rate:.3f}" if s.win_rate is not None else "n/a",
            f"{s.win_rate_ci_low:.3f}" if s.win_rate_ci_low is not None else "n/a",
            f"{s.win_rate_ci_high:.3f}" if s.win_rate_ci_high is not None else "n/a",
            f"{s.expectancy_pct:.4f}" if s.expectancy_pct is not None else "n/a",
            s.voided,
            result.backtest_run_id,
        )


if __name__ == "__main__":
    main()
