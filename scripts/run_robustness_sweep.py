#!/usr/bin/env python
"""Phase 6: sweep a strategy's parameters (and, optionally, time windows)
through the backtesting engine and report whether a statistically
credible edge survives the perturbation, or only shows up at one lucky
point (BACKTESTING.md's OVERFITTED/FRAGILE rule).

Never offers ``--split test`` — sweeps run many backtests at once, and
that's exactly the kind of run that should never be able to touch the
out-of-sample split by accident (see run_baseline_backtests.py's same
restriction).

Example — H4's Bollinger thresholds on real EUR/USD 1h data:
    python scripts/run_robustness_sweep.py \\
        --strategy otc_research.strategies.h4_bollinger:H4BollingerMeanReversion \\
        --pair EUR_USD --timeframe 1h \\
        --param-grid '[{"lower_pct_b":0.0,"upper_pct_b":1.0,"expiry_seconds":3600},
                        {"lower_pct_b":0.05,"upper_pct_b":0.95,"expiry_seconds":3600},
                        {"lower_pct_b":0.1,"upper_pct_b":0.9,"expiry_seconds":3600}]'
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json

from otc_research.backtest.robustness import evaluate_robustness, run_parameter_sweep
from otc_research.config import load_config
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


def _parse_iso(value: str | None) -> dt.datetime | None:
    if value is None:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_windows(raw: str | None) -> list[tuple[dt.datetime | None, dt.datetime | None]] | None:
    if raw is None:
        return None
    pairs = json.loads(raw)
    return [(_parse_iso(start), _parse_iso(end)) for start, end in pairs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", required=True, help="'module.path:ClassName'")
    parser.add_argument("--pair", required=True, help='e.g. "EUR_USD"')
    parser.add_argument("--timeframe", required=True, help='"1m", "5m", "15m", "1h", ...')
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument(
        "--param-grid",
        required=True,
        help="JSON array of constructor kwarg objects, e.g. '[{\"expiry_seconds\": 3600}, ...]'",
    )
    parser.add_argument(
        "--windows",
        default=None,
        help=(
            'JSON array of [start_iso_or_null, end_iso_or_null] pairs, e.g. '
            '\'[["2026-01-01T00:00:00Z", "2026-04-01T00:00:00Z"], [null, null]]\'. '
            "Defaults to a single unrestricted window (the whole ingested history)."
        ),
    )
    parser.add_argument("--scenario", default="optimistic", choices=["optimistic", "realistic", "pessimistic"])
    parser.add_argument("--min-sample-size", type=int, default=30)
    parser.add_argument("--min-edge-fraction", type=float, default=0.7)
    parser.add_argument("--feature-set-version", default=FEATURE_SET_VERSION)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    strategy_cls = _load_strategy_class(args.strategy)
    param_grid = json.loads(args.param_grid)
    windows = _parse_windows(args.windows)

    points = run_parameter_sweep(
        session,
        strategy_cls,
        param_grid,
        asset=args.pair,
        timeframe=args.timeframe,
        backtest_config=config.backtest,
        feature_set_version=args.feature_set_version,
        split=args.split,
        windows=windows,
        rng_seed=args.rng_seed,
    )

    for point in points:
        result = point.result_for(args.scenario)
        s = result.stats if result else None
        logger.info(
            "params=%s window=%s n=%s win_rate=%s ci=[%s, %s]",
            point.params,
            point.window,
            s.sample_size if s else "n/a",
            f"{s.win_rate:.3f}" if s and s.win_rate is not None else "n/a",
            f"{s.win_rate_ci_low:.3f}" if s and s.win_rate_ci_low is not None else "n/a",
            f"{s.win_rate_ci_high:.3f}" if s and s.win_rate_ci_high is not None else "n/a",
        )

    verdict = evaluate_robustness(
        points, args.scenario, min_sample_size=args.min_sample_size,
        min_edge_fraction=args.min_edge_fraction,
    )
    logger.info(
        "VERDICT (%s scenario): %s -- %d/%d sufficiently-sampled points showed an edge "
        "(min_sample_size=%d, min_edge_fraction=%.2f). This is NOT a final edge "
        "classification -- see BACKTESTING.md (walk-forward + Monte Carlo still needed).",
        args.scenario,
        verdict.classification,
        verdict.n_with_edge,
        verdict.n_sufficiently_sampled,
        verdict.min_sample_size,
        verdict.min_edge_fraction,
    )


if __name__ == "__main__":
    main()
