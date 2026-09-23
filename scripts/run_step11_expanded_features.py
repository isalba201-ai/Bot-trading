#!/usr/bin/env python
"""Step 11 (user-requested, 2026-09-23): discovery over the feature
families never yet searched. Every discovery run in this project so far
(Step 7, Step 8d, Step 10) used the same 10-feature
``CORE_DISCOVERY_FEATURES`` subset. This script searches
``EXPANDED_DISCOVERY_FEATURES`` instead — 21 additional v4 features
spanning momentum (multi-lag returns, ROC, return acceleration), MACD
crossover, candle-shape (wick ratios, body ratio, engulfing, inside-bar
breakout), streaks, volatility (rolling std of returns), structure/
distance-to-extremes — a genuinely new region of the search space, not a
deeper interaction of the same 10 features already tried.

Deliberately EXCLUDES every raw, absolute-price-scale feature (ema_12,
ema_26, bb_mid/upper/lower_20, donchian_high/low_20, atr_14,
macd_line/macd_signal_line/macd_histogram): a quantile-binned condition on an absolute
price level isn't a meaningful, reusable rule, and — per
EXPIRY_UNIVERSE_AUDIT.md Section 6 — isn't conceptually portable to a
different feed (e.g. Pocket Option OTC's own price scale) the way a
ratio/percentage/normalized feature is. Every included feature is already
scale-invariant (a ratio, a percentage, a z-like normalized distance, a
categorical/signed indicator, or a bounded oscillator).

``combo_sizes=(2,)`` only (not 3-way): C(21,3) x 64 bins would be ~5.4M
trials across 8 datasets -- disproportionate to the marginal value of a
3rd untested feature joining two already-untested ones. 2-way across 21
new features is itself the genuinely new, previously-unexplored region;
3-way-among-new-features remains a further, separate, not-yet-taken step
if this one finds something worth deepening.

Own run_id prefix, own independent Benjamini-Hochberg correction, never
pooled with any prior run. Same corrected delay-only execution model,
same four-gate candidacy funnel, same 8-dataset universe as Step 10.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter

from otc_research.backtest.splits import compute_temporal_split
from otc_research.backtest.walkforward import generate_folds
from otc_research.config import load_config
from otc_research.db.models import Candle
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.research.candidacy import CandidacyThresholds, evaluate_condition_candidacy
from otc_research.research.dataset import DEFAULT_HORIZONS, build_dataset
from otc_research.research.discovery import run_discovery
from otc_research.utils.logging import get_logger
from otc_research.utils.timeframes import timeframe_to_seconds

logger = get_logger(__name__)

#: 21 v4 features never used in any prior discovery run (Step 7's
#: CORE_DISCOVERY_FEATURES, Step 8d's +SESSION_FEATURES, Step 10's same
#: CORE set) -- excludes raw price-scale features, see module docstring.
EXPANDED_DISCOVERY_FEATURES: tuple[str, ...] = (
    "ema_slope_12_3",
    "roc_10",
    "return_2", "return_3", "return_10", "return_15", "return_30",
    "cumulative_return_10",
    "return_acceleration",
    "macd_cross_signal",
    "same_color_streak",
    "range_ratio_20",
    "rolling_std_return_20",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "body_ratio",
    "engulfing_signal",
    "inside_bar_breakout_signal",
    "dist_to_high_atr_20",
    "dist_to_low_atr_20",
    "structure_bias",
)

ASSET_TIMEFRAMES: tuple[tuple[str, str], ...] = (
    ("EUR_USD", "1m"), ("EUR_USD", "5m"), ("EUR_USD", "15m"), ("EUR_USD", "1h"),
    ("GBP_USD", "5m"), ("GBP_USD", "1h"),
    ("USD_JPY", "5m"), ("USD_JPY", "1h"),
)

DEFAULT_PAYOUT = 0.85
TOP_N_CANDIDATES_PER_DATASET = 3
N_WALK_FORWARD_FOLDS = 5


def _walk_forward_folds(session, asset, timeframe, backtest_config, *, n_folds=N_WALK_FORWARD_FOLDS):
    first_ts = (
        session.query(Candle.timestamp).filter(Candle.asset == asset, Candle.timeframe == timeframe)
        .order_by(Candle.timestamp.asc()).first()
    )
    last_ts = (
        session.query(Candle.timestamp).filter(Candle.asset == asset, Candle.timeframe == timeframe)
        .order_by(Candle.timestamp.desc()).first()
    )
    history_start, history_end = first_ts[0], last_ts[0]
    total_span = history_end - history_start
    train_val_span = total_span * (backtest_config.train_fraction + backtest_config.validation_fraction)
    train_val_end = history_start + train_val_span
    fold_span = train_val_span / n_folds
    return generate_folds(history_start, train_val_end, train_span=fold_span, test_span=fold_span)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id-prefix", default=None)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--output", default="data/step11_expanded_features_results.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()
    thresholds = CandidacyThresholds()

    run_id_prefix = args.run_id_prefix or dt.datetime.now(dt.timezone.utc).strftime("step11-%Y%m%dT%H%M%S")
    logger.info("Step 11 expanded-feature search starting -- run_id_prefix=%s", run_id_prefix)
    logger.info("Feature set (%d features): %s", len(EXPANDED_DISCOVERY_FEATURES), EXPANDED_DISCOVERY_FEATURES)

    rows = []
    with open(args.output, "a") as out:
        for asset, timeframe in ASSET_TIMEFRAMES:
            logger.info("=== %s / %s ===", asset, timeframe)
            df = build_dataset(session, asset, timeframe, feature_set_version=FEATURE_SET_VERSION, horizons=DEFAULT_HORIZONS)
            if df.empty:
                logger.warning("no dataset rows for %s/%s -- skipping", asset, timeframe)
                continue
            split = compute_temporal_split(len(df), config.backtest.train_fraction, config.backtest.validation_fraction)
            train_df = df.iloc[split.train_slice]
            folds = _walk_forward_folds(session, asset, timeframe, config.backtest)

            all_significant = []
            for horizon in DEFAULT_HORIZONS:
                for target_col in (f"call_wins_{horizon}", f"put_wins_{horizon}"):
                    run_id = f"{run_id_prefix}:{asset}:{timeframe}:{target_col}"
                    results = run_discovery(
                        session, train_df, list(EXPANDED_DISCOVERY_FEATURES), target_col,
                        asset=asset, timeframe=timeframe, feature_set_version=FEATURE_SET_VERSION,
                        combo_sizes=(2,), n_bins=4, min_sample_size=50, fdr_q=0.05, run_id=run_id,
                    )
                    significant = [
                        r for r in results
                        if r.fdr_significant and r.stat.n >= 100 and r.stat.win_rate is not None and r.stat.win_rate > 0.5
                    ]
                    logger.info(
                        "  discovery(expanded) %-14s trials=%-6d fdr_significant_winning=%d",
                        target_col, len(results), len(significant),
                    )
                    direction = "CALL" if target_col.startswith("call_wins") else "PUT"
                    for r in significant[:TOP_N_CANDIDATES_PER_DATASET]:
                        all_significant.append({
                            "target_col": target_col, "horizon": horizon, "direction": direction,
                            "condition": r.condition, "p_value": r.p_value,
                        })

            for entry in all_significant:
                expiry_seconds = entry["horizon"] * timeframe_to_seconds(timeframe)
                verdict = evaluate_condition_candidacy(
                    session, entry["condition"], entry["direction"], expiry_seconds, asset, timeframe,
                    config.backtest, feature_set_version=FEATURE_SET_VERSION, payout=DEFAULT_PAYOUT,
                    walk_forward_folds=folds, thresholds=thresholds, rng_seed=args.rng_seed,
                    label=f"{run_id_prefix}:{entry['target_col']}:{entry['condition'].label()}",
                )
                logger.info(
                    "  CANDIDACY(expanded) %s %s h=%d -- accepted=%s rejected_at=%s",
                    entry["condition"].label(), entry["direction"], entry["horizon"],
                    verdict.accepted, verdict.rejected_at_gate,
                )
                row = {
                    "asset": asset, "timeframe": timeframe, "h": entry["horizon"],
                    "direction": entry["direction"], "label": entry["condition"].label(),
                    "accepted": verdict.accepted, "rejected_at_gate": verdict.rejected_at_gate,
                    "train_sample_size": verdict.train_sample_size,
                    "train_win_rate": verdict.train_win_rate,
                    "test_sample_size": verdict.test_sample_size,
                    "test_win_rate": verdict.test_win_rate,
                    "test_win_rate_ci_low": verdict.test_win_rate_ci_low,
                }
                out.write(json.dumps(row) + "\n")
                rows.append(row)

    logger.info("=" * 78)
    logger.info("STEP 11 SUMMARY (n=%d evaluations)", len(rows))
    counts = Counter("ACCEPTED" if r["accepted"] else (r["rejected_at_gate"] or "unknown") for r in rows)
    for label, count in counts.most_common():
        logger.info("  %-28s %d", label, count)
    accepted = [r for r in rows if r["accepted"]]
    if accepted:
        logger.info("ACCEPTED:")
        for r in accepted:
            logger.info("  %s %s/%s %s h=%d test_wr=%s test_ci_low=%s",
                         r["label"], r["asset"], r["timeframe"], r["direction"], r["h"],
                         r["test_win_rate"], r["test_win_rate_ci_low"])
    else:
        logger.info("ACCEPTED: none")


if __name__ == "__main__":
    main()
