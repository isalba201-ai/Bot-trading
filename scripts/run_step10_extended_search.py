#!/usr/bin/env python
"""Step 10 (user-requested, 2026-09-18): the two pieces of the original
statistical-discovery methodology that were built but never actually run
to completion against real data:

1. **3-way interactions** — Step 7's pipeline (``run_research_pipeline.py``)
   deliberately only ran ``combo_sizes=(2,)`` ("3-way is the natural next
   step ... not run automatically here", per its own docstring). This
   script runs ``combo_sizes=(2, 3)`` — own ``run_id`` prefix, own
   independent Benjamini-Hochberg correction, never pooled with Step 7's
   2-way-only run.
2. **The ML model, run and actually gated** — Step 7's pipeline called
   ``research.models.run_model_comparison`` only for the single best
   discovery hit per dataset, and only ever logged its VALIDATION read
   (never pushed a promising model through ``research.candidacy`` for a
   genuine TEST touch). This script runs all 3 model families
   independently for EVERY (asset, timeframe, horizon, direction), and
   escalates any that clear break-even on VALIDATION with adequate
   coverage into the exact same four-gate candidacy funnel via the new
   ``research.model_strategy.ModelStrategy`` wrapper.

Also expands the asset/timeframe universe from Step 7's original 5
datasets to all 8 real ingested ones (adds EUR_USD/GBP_USD/USD_JPY 1h,
which have never been systematically searched before — only used for
hand-designed H1-H20 strategies).

Same 10-feature ``CORE_DISCOVERY_FEATURES`` subset as Step 7 (not
expanded to the full ~40 v4 features — that would be a further, separate
extension, not part of what was asked for here). Same corrected
delay-only execution model as every other candidacy call in this project
(``evaluate_candidacy``'s own default).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter

from otc_research.backtest.metrics import break_even_win_rate
from otc_research.backtest.splits import compute_temporal_split
from otc_research.backtest.walkforward import generate_folds
from otc_research.config import load_config
from otc_research.db.models import Candle
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.research.candidacy import (
    CandidacyThresholds,
    evaluate_candidacy,
    evaluate_condition_candidacy,
)
from otc_research.research.dataset import DEFAULT_HORIZONS, build_dataset
from otc_research.research.discovery import run_discovery
from otc_research.research.model_strategy import ModelStrategy
from otc_research.research.models import MODEL_FAMILIES, fit_and_evaluate
from otc_research.utils.logging import get_logger
from otc_research.utils.timeframes import timeframe_to_seconds

logger = get_logger(__name__)

CORE_DISCOVERY_FEATURES: tuple[str, ...] = (
    "return_1", "return_5", "rsi_14", "adx_14", "atr_expansion_ratio",
    "cci_20", "rci_9", "bb_pct_b_20", "move_size_atr", "pct_position_in_range_20",
)

ASSET_TIMEFRAMES: tuple[tuple[str, str], ...] = (
    ("EUR_USD", "1m"), ("EUR_USD", "5m"), ("EUR_USD", "15m"), ("EUR_USD", "1h"),
    ("GBP_USD", "5m"), ("GBP_USD", "1h"),
    ("USD_JPY", "5m"), ("USD_JPY", "1h"),
)

DEFAULT_PAYOUT = 0.85
TOP_N_CANDIDATES_PER_DATASET = 3
N_WALK_FORWARD_FOLDS = 5
#: Minimum VALIDATION-read coverage before a model is even considered for
#: escalation to the full candidacy funnel -- an ML model that would only
#: ever trade a handful of times isn't worth a TEST touch either.
MODEL_MIN_VALIDATION_N = 30
MODEL_PROBABILITY_THRESHOLD = 0.5
#: Perturbation grid for the model's own robustness gate (2): sensitivity
#: to the decision threshold, the model-specific analogue of widening a
#: discovered condition's bin edges.
MODEL_THRESHOLD_PERTURBATION_GRID: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55, 0.60)


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


def _run_discovery_3way(session, train_df, asset, timeframe, run_id_prefix):
    """Returns the top FDR-significant, winning-direction condition per
    target_col, from a combo_sizes=(2,3) search -- own run_id, own FDR.
    """
    all_significant = []
    for horizon in DEFAULT_HORIZONS:
        for target_col in (f"call_wins_{horizon}", f"put_wins_{horizon}"):
            run_id = f"{run_id_prefix}:{asset}:{timeframe}:{target_col}"
            results = run_discovery(
                session, train_df, list(CORE_DISCOVERY_FEATURES), target_col,
                asset=asset, timeframe=timeframe, feature_set_version=FEATURE_SET_VERSION,
                combo_sizes=(2, 3), n_bins=4, min_sample_size=50, fdr_q=0.05, run_id=run_id,
            )
            significant = [
                r for r in results
                if r.fdr_significant and r.stat.n >= 100 and r.stat.win_rate is not None and r.stat.win_rate > 0.5
            ]
            logger.info(
                "  discovery(2+3way) %-14s trials=%-7d fdr_significant_winning=%d",
                target_col, len(results), len(significant),
            )
            direction = "CALL" if target_col.startswith("call_wins") else "PUT"
            for r in significant[:TOP_N_CANDIDATES_PER_DATASET]:
                all_significant.append({
                    "target_col": target_col, "horizon": horizon, "direction": direction,
                    "condition": r.condition, "p_value": r.p_value,
                })
    return all_significant


def _run_ml_sweep(session, train_df, validation_df, asset, timeframe):
    """Independent ML model comparison for every (horizon, direction) --
    not gated behind discovery finding anything first (per the user's
    explicit request to run the model against all the real data). Returns
    every promising (VALIDATION CI-low clears break-even, adequate
    coverage) fit, ready for candidacy escalation by the caller.
    """
    break_even = break_even_win_rate(DEFAULT_PAYOUT)
    promising = []
    for horizon in DEFAULT_HORIZONS:
        expiry_seconds = horizon * timeframe_to_seconds(timeframe)
        for target_col in (f"call_wins_{horizon}", f"put_wins_{horizon}"):
            direction = "CALL" if target_col.startswith("call_wins") else "PUT"
            try:
                fits = [
                    fit_and_evaluate(
                        train_df, validation_df, list(CORE_DISCOVERY_FEATURES), target_col,
                        family=family, probability_threshold=MODEL_PROBABILITY_THRESHOLD, seed=0,
                    )
                    for family in MODEL_FAMILIES
                ]
            except ValueError as exc:
                logger.info("  ML %-14s skipped: %s", target_col, exc)
                continue

            for fit in fits:
                stat = fit.validation_stat_at_threshold
                logger.info(
                    "  ML %-20s %-14s val_n=%-5d val_wr=%s coverage=%.3f",
                    fit.family, target_col, stat.n,
                    f"{stat.win_rate:.4f}" if stat.win_rate is not None else "n/a",
                    fit.validation_coverage,
                )
                if stat.n >= MODEL_MIN_VALIDATION_N and stat.ci_low is not None and stat.ci_low > break_even:
                    promising.append({
                        "fit": fit, "target_col": target_col, "horizon": horizon,
                        "direction": direction, "expiry_seconds": expiry_seconds,
                    })
    return promising


def _evaluate_model_candidacy(session, entry, asset, timeframe, config, folds, thresholds, rng_seed):
    fit = entry["fit"]
    base_strategy = ModelStrategy(
        fit.model, CORE_DISCOVERY_FEATURES, entry["direction"], entry["expiry_seconds"],
        probability_threshold=MODEL_PROBABILITY_THRESHOLD,
        label=f"model_{fit.family}_{asset}_{timeframe}_{entry['target_col']}",
    )

    def strategy_factory(probability_threshold: float) -> ModelStrategy:
        return ModelStrategy(
            fit.model, CORE_DISCOVERY_FEATURES, entry["direction"], entry["expiry_seconds"],
            probability_threshold=probability_threshold, label=base_strategy.label,
        )

    param_grid = [{"probability_threshold": t} for t in MODEL_THRESHOLD_PERTURBATION_GRID]
    return evaluate_candidacy(
        session, base_strategy, asset, timeframe, config.backtest,
        feature_set_version=FEATURE_SET_VERSION, payout=DEFAULT_PAYOUT,
        walk_forward_folds=folds, strategy_factory=strategy_factory,
        perturbation_param_grid=param_grid, thresholds=thresholds, rng_seed=rng_seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id-prefix", default=None)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--output", default="data/step10_extended_search_results.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()
    thresholds = CandidacyThresholds()

    run_id_prefix = args.run_id_prefix or dt.datetime.now(dt.timezone.utc).strftime("step10-%Y%m%dT%H%M%S")
    logger.info("Step 10 extended search starting -- run_id_prefix=%s", run_id_prefix)

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
            validation_df = df.iloc[split.validation_slice]

            folds = _walk_forward_folds(session, asset, timeframe, config.backtest)

            # --- 3-way discovery -------------------------------------------------
            significant = _run_discovery_3way(session, train_df, asset, timeframe, run_id_prefix)
            for entry in significant:
                expiry_seconds = entry["horizon"] * timeframe_to_seconds(timeframe)
                verdict = evaluate_condition_candidacy(
                    session, entry["condition"], entry["direction"], expiry_seconds, asset, timeframe,
                    config.backtest, feature_set_version=FEATURE_SET_VERSION, payout=DEFAULT_PAYOUT,
                    walk_forward_folds=folds, thresholds=thresholds, rng_seed=args.rng_seed,
                    label=f"{run_id_prefix}:{entry['target_col']}:{entry['condition'].label()}",
                )
                logger.info(
                    "  CANDIDACY(3way) %s %s h=%d -- accepted=%s rejected_at=%s",
                    entry["condition"].label(), entry["direction"], entry["horizon"],
                    verdict.accepted, verdict.rejected_at_gate,
                )
                row = {
                    "track": "discovery_3way", "asset": asset, "timeframe": timeframe,
                    "h": entry["horizon"], "direction": entry["direction"],
                    "label": entry["condition"].label(), "accepted": verdict.accepted,
                    "rejected_at_gate": verdict.rejected_at_gate,
                    "train_sample_size": verdict.train_sample_size,
                    "train_win_rate": verdict.train_win_rate,
                    "test_sample_size": verdict.test_sample_size,
                    "test_win_rate": verdict.test_win_rate,
                    "test_win_rate_ci_low": verdict.test_win_rate_ci_low,
                }
                out.write(json.dumps(row) + "\n")
                rows.append(row)

            # --- ML sweep ----------------------------------------------------------
            promising = _run_ml_sweep(session, train_df, validation_df, asset, timeframe)
            for entry in promising:
                verdict = _evaluate_model_candidacy(session, entry, asset, timeframe, config, folds, thresholds, args.rng_seed)
                logger.info(
                    "  CANDIDACY(ML/%s) %s %s h=%d -- accepted=%s rejected_at=%s",
                    entry["fit"].family, entry["target_col"], entry["direction"], entry["horizon"],
                    verdict.accepted, verdict.rejected_at_gate,
                )
                row = {
                    "track": f"ml_{entry['fit'].family}", "asset": asset, "timeframe": timeframe,
                    "h": entry["horizon"], "direction": entry["direction"],
                    "label": f"model_{entry['fit'].family}_{entry['target_col']}",
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
    logger.info("STEP 10 SUMMARY (n=%d evaluations)", len(rows))
    counts = Counter("ACCEPTED" if r["accepted"] else (r["rejected_at_gate"] or "unknown") for r in rows)
    for label, count in counts.most_common():
        logger.info("  %-28s %d", label, count)
    accepted = [r for r in rows if r["accepted"]]
    if accepted:
        logger.info("ACCEPTED:")
        for r in accepted:
            logger.info("  %s %s/%s %s h=%d test_wr=%s test_ci_low=%s",
                         r["track"], r["asset"], r["timeframe"], r["direction"], r["h"],
                         r["test_win_rate"], r["test_win_rate_ci_low"])
    else:
        logger.info("ACCEPTED: none")


if __name__ == "__main__":
    main()
