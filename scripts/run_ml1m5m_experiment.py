#!/usr/bin/env python
"""Experimental, independent line of research (user-requested,
2026-09-23): a strategy that analyzes EUR/USD 1-minute candles and, when
it finds an opportunity, enters a binary option expiring **5 minutes
after entry**. This is NOT a retimeframing of ML10_FWD (USD/JPY 1h,
3h expiry) -- it is trained fresh, on its own data window, its own
feature set, its own discovery run. Does not touch H9_FWD, ML10_FWD, or
FORWARD_TEST_CANDIDATES.

Two execution scenarios are evaluated for EVERY candidate, both defined
BEFORE any result is seen (pre-registered per the user's explicit
instruction) -- never chosen after peeking at which one looks better:

  Scenario A (delay_only_scenario(0)): signal at candle close -> entry at
  the very next candle's open (1 minute later) -> exit 5 candles after
  entry (6 minutes total from signal to exit). Zero look-ahead (only uses
  information known at the signal candle's close), but assumes
  near-instant order placement.

  Scenario B (delay_only_scenario(1)): the project's standing convention
  (same as H9_FWD/ML10_FWD/every discovery run) -- one extra candle of
  execution latency. Entry 2 minutes after signal, exit 7 minutes after
  signal in total.

Both scenarios share the exact same TRAIN/VALIDATION/TEST data, features,
targets, and discovered/fitted candidates -- only the execution-timing
assumption differs between the two evaluate_candidacy() calls. A
candidate accepted under A but not B is reported as "sensitive to
latency"; accepted under both is "robust to the delay assumption";
rejected under both is discarded under the existing rules. No result is
used to pick which scenario to report -- both are always reported.

Data window: EUR_USD 1m, 2026-07-21 -> 2026-08-04 (a genuinely new block,
fetched specifically for this experiment -- deliberately disjoint from
the 2026-09-15 -> 2026-09-18 EUR_USD/1m block already spent on Step 10/11
discovery, so this experiment never re-touches data any prior run already
used for FDR-controlled significance testing).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter

from otc_research.backtest.execution import delay_only_scenario
from otc_research.backtest.metrics import break_even_win_rate
from otc_research.backtest.splits import compute_temporal_split
from otc_research.backtest.walkforward import compute_walk_forward_folds
from otc_research.config import load_config
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.research.candidacy import (
    CandidacyThresholds,
    evaluate_candidacy,
    evaluate_condition_candidacy,
)
from otc_research.research.dataset import build_dataset
from otc_research.research.discovery import run_discovery
from otc_research.research.model_strategy import ModelStrategy
from otc_research.research.models import MODEL_FAMILIES, fit_and_evaluate
from otc_research.utils.logging import get_logger

logger = get_logger(__name__)

ASSET = "EUR_USD"
TIMEFRAME = "1m"
HORIZON = 5  # candles = 5 minutes
EXPIRY_SECONDS = HORIZON * 60
DEFAULT_PAYOUT = 0.85

#: The pre-registered, disjoint data window for this experiment -- never
#: extended after seeing results without a fresh, independently-labeled
#: re-run.
WINDOW_START = dt.datetime(2026, 7, 21)
WINDOW_END = dt.datetime(2026, 8, 4, 0, 1)

#: 1-minute-appropriate feature set: momentum/volatility/oscillators/
#: candle-shape/structure (all scale-invariant, same principle as Step
#: 11's EXPANDED_DISCOVERY_FEATURES) PLUS time-of-day/session features,
#: which the technical design flagged as more likely to matter at this
#: granularity than at 1h (intraday liquidity effects dominate short FX
#: horizons more than they do longer ones).
ML1M5M_FEATURES: tuple[str, ...] = (
    "return_1", "return_2", "return_3", "return_5", "roc_10",
    "cumulative_return_10", "return_acceleration",
    "atr_expansion_ratio", "rolling_std_return_20", "move_size_atr",
    "rsi_14", "adx_14", "cci_20", "rci_9", "bb_pct_b_20", "macd_cross_signal",
    "same_color_streak", "upper_wick_ratio", "lower_wick_ratio", "body_ratio",
    "engulfing_signal", "inside_bar_breakout_signal", "range_ratio_20",
    "pct_position_in_range_20", "dist_to_high_atr_20", "dist_to_low_atr_20",
    "structure_bias",
    "hour_utc", "day_of_week", "trading_session_code",
)

TOP_N_CANDIDATES = 5
N_WALK_FORWARD_FOLDS = 5
MODEL_MIN_VALIDATION_N = 30
MODEL_PROBABILITY_THRESHOLD = 0.5
MODEL_THRESHOLD_PERTURBATION_GRID: tuple[float, ...] = (0.45, 0.50, 0.55)

#: Both scenarios, defined once, before any candidate is evaluated.
SCENARIOS: tuple = (
    ("delay0", delay_only_scenario(0)),
    ("delay1", delay_only_scenario(1)),
)


def _load_windowed_dataset(session):
    df = build_dataset(session, ASSET, TIMEFRAME, feature_set_version=FEATURE_SET_VERSION, horizons=(HORIZON,))
    df = df[(df["timestamp"] >= WINDOW_START) & (df["timestamp"] < WINDOW_END)].reset_index(drop=True)
    df = df.dropna(subset=[f"call_wins_{HORIZON}", f"put_wins_{HORIZON}"]).reset_index(drop=True)
    return df


def _walk_forward_folds(session, backtest_config, *, n_folds=N_WALK_FORWARD_FOLDS):
    """Delegates to ``backtest.walkforward.compute_walk_forward_folds``,
    which bounds folds to the ROW-based TRAIN+VALIDATION boundary (via
    ``backtest.engine.compute_split_windows``) rather than a calendar-time
    proportion estimate -- the estimate is what let a fold overflow ~2h
    into TEST in the original run (see the ML_1M5M split-mismatch
    investigation). Bounded to ``[WINDOW_START, WINDOW_END)`` so it can
    never see the unrelated, already-used EUR_USD/1m block sitting
    elsewhere in the same table either.
    """
    return compute_walk_forward_folds(
        session, ASSET, TIMEFRAME, backtest_config,
        start=WINDOW_START, end=WINDOW_END, n_folds=n_folds,
    )


def _run_discovery(session, train_df, run_id_prefix):
    all_significant = []
    for target_col in (f"call_wins_{HORIZON}", f"put_wins_{HORIZON}"):
        run_id = f"{run_id_prefix}:{ASSET}:{TIMEFRAME}:{target_col}"
        results = run_discovery(
            session, train_df, list(ML1M5M_FEATURES), target_col,
            asset=ASSET, timeframe=TIMEFRAME, feature_set_version=FEATURE_SET_VERSION,
            combo_sizes=(2,), n_bins=4, min_sample_size=50, fdr_q=0.05, run_id=run_id,
        )
        significant = [
            r for r in results
            if r.fdr_significant and r.stat.n >= 100 and r.stat.win_rate is not None and r.stat.win_rate > 0.5
        ]
        logger.info("discovery %-14s trials=%-6d fdr_significant_winning=%d", target_col, len(results), len(significant))
        direction = "CALL" if target_col.startswith("call_wins") else "PUT"
        for r in significant[:TOP_N_CANDIDATES]:
            all_significant.append({
                "kind": "condition", "target_col": target_col, "direction": direction,
                "condition": r.condition, "p_value": r.p_value,
            })
    return all_significant


def _run_ml_fit(train_df, validation_df):
    break_even = break_even_win_rate(DEFAULT_PAYOUT)
    promising = []
    for target_col in (f"call_wins_{HORIZON}", f"put_wins_{HORIZON}"):
        direction = "CALL" if target_col.startswith("call_wins") else "PUT"
        try:
            fits = [
                fit_and_evaluate(
                    train_df, validation_df, list(ML1M5M_FEATURES), target_col,
                    family=family, probability_threshold=MODEL_PROBABILITY_THRESHOLD, seed=0,
                )
                for family in MODEL_FAMILIES
            ]
        except ValueError as exc:
            logger.info("ML %-14s skipped: %s", target_col, exc)
            continue

        best_promising = None
        for fit in fits:
            stat = fit.validation_stat_at_threshold
            logger.info(
                "ML %-20s %-14s val_n=%-5d val_wr=%s coverage=%.3f",
                fit.family, target_col, stat.n,
                f"{stat.win_rate:.4f}" if stat.win_rate is not None else "n/a",
                fit.validation_coverage,
            )
            is_promising = stat.n >= MODEL_MIN_VALIDATION_N and stat.ci_low is not None and stat.ci_low > break_even
            if is_promising and (best_promising is None or stat.win_rate > best_promising.validation_stat_at_threshold.win_rate):
                best_promising = fit
        if best_promising is not None:
            promising.append({"kind": "model", "fit": best_promising, "target_col": target_col, "direction": direction})
    return promising


def _evaluate_condition_both_scenarios(session, entry, backtest_config, folds, thresholds, run_id_prefix, rng_seed, *, allow_test=True):
    out = []
    label = f"{run_id_prefix}:{entry['target_col']}:{entry['condition'].label()}"
    for scenario_name, scenario in SCENARIOS:
        verdict = evaluate_condition_candidacy(
            session, entry["condition"], entry["direction"], EXPIRY_SECONDS, ASSET, TIMEFRAME,
            backtest_config, feature_set_version=FEATURE_SET_VERSION, payout=DEFAULT_PAYOUT,
            walk_forward_folds=folds, thresholds=thresholds, scenario=scenario, rng_seed=rng_seed,
            label=f"{label}:{scenario_name}",
            start=WINDOW_START, end=WINDOW_END, allow_test=allow_test,
        )
        out.append((scenario_name, verdict, label))
    return out


def _evaluate_model_both_scenarios(session, entry, backtest_config, folds, thresholds, run_id_prefix, rng_seed, *, allow_test=True):
    fit = entry["fit"]
    label = f"{run_id_prefix}:{entry['target_col']}:model_{fit.family}"
    out = []
    for scenario_name, scenario in SCENARIOS:
        base_strategy = ModelStrategy(
            fit.model, ML1M5M_FEATURES, entry["direction"], EXPIRY_SECONDS,
            probability_threshold=MODEL_PROBABILITY_THRESHOLD, label=f"{label}:{scenario_name}",
        )

        def strategy_factory(probability_threshold: float, _fit=fit, _direction=entry["direction"], _label=base_strategy.label) -> ModelStrategy:
            return ModelStrategy(
                _fit.model, ML1M5M_FEATURES, _direction, EXPIRY_SECONDS,
                probability_threshold=probability_threshold, label=_label,
            )

        param_grid = [{"probability_threshold": t} for t in MODEL_THRESHOLD_PERTURBATION_GRID]
        verdict = evaluate_candidacy(
            session, base_strategy, ASSET, TIMEFRAME, backtest_config,
            feature_set_version=FEATURE_SET_VERSION, payout=DEFAULT_PAYOUT,
            walk_forward_folds=folds, strategy_factory=strategy_factory,
            perturbation_param_grid=param_grid, thresholds=thresholds, scenario=scenario, rng_seed=rng_seed,
            start=WINDOW_START, end=WINDOW_END, allow_test=allow_test,
        )
        out.append((scenario_name, verdict, label))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id-prefix", default=None)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--output", default="data/ml1m5m_experiment_results.jsonl")
    parser.add_argument(
        "--no-test", action="store_true",
        help="Stop every candidate after gate 3 (walk-forward); TEST is never touched "
             "(evaluate_candidacy(allow_test=False)). Use for a baseline/debugging re-run.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()
    thresholds = CandidacyThresholds()

    run_id_prefix = args.run_id_prefix or dt.datetime.now(dt.timezone.utc).strftime("ml1m5m-%Y%m%dT%H%M%S")
    logger.info("ML_1M5M experiment starting -- run_id_prefix=%s", run_id_prefix)
    logger.info("Window: %s -> %s", WINDOW_START, WINDOW_END)

    df = _load_windowed_dataset(session)
    logger.info("dataset rows in window: %d", len(df))
    split = compute_temporal_split(len(df), config.backtest.train_fraction, config.backtest.validation_fraction)
    train_df = df.iloc[split.train_slice]
    validation_df = df.iloc[split.validation_slice]
    logger.info(
        "TRAIN: %s -> %s (n=%d)", train_df["timestamp"].min(), train_df["timestamp"].max(), len(train_df)
    )
    logger.info(
        "VALIDATION: %s -> %s (n=%d)", validation_df["timestamp"].min(), validation_df["timestamp"].max(), len(validation_df)
    )
    test_df = df.iloc[split.test_slice]
    logger.info(
        "TEST: %s -> %s (n=%d)", test_df["timestamp"].min(), test_df["timestamp"].max(), len(test_df)
    )

    folds = _walk_forward_folds(session, config.backtest)

    candidates: list[dict] = []
    candidates.extend(_run_discovery(session, train_df, run_id_prefix))
    candidates.extend(_run_ml_fit(train_df, validation_df))
    logger.info("total candidates to evaluate (both scenarios each): %d", len(candidates))

    rows = []
    with open(args.output, "a") as out:
        for entry in candidates:
            if entry["kind"] == "condition":
                results = _evaluate_condition_both_scenarios(
                    session, entry, config.backtest, folds, thresholds, run_id_prefix, args.rng_seed,
                    allow_test=not args.no_test,
                )
            else:
                results = _evaluate_model_both_scenarios(
                    session, entry, config.backtest, folds, thresholds, run_id_prefix, args.rng_seed,
                    allow_test=not args.no_test,
                )
            for scenario_name, verdict, label in results:
                logger.info(
                    "CANDIDACY[%s] %s %s -- accepted=%s rejected_at=%s",
                    scenario_name, label, entry["direction"], verdict.accepted, verdict.rejected_at_gate,
                )
                row = {
                    "kind": entry["kind"], "label": label, "scenario": scenario_name,
                    "direction": entry["direction"], "target_col": entry["target_col"],
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
    logger.info("ML_1M5M SUMMARY (n=%d scenario-evaluations, %d candidates x %d scenarios)", len(rows), len(candidates), len(SCENARIOS))
    for scenario_name, _ in SCENARIOS:
        scen_rows = [r for r in rows if r["scenario"] == scenario_name]
        counts = Counter("ACCEPTED" if r["accepted"] else (r["rejected_at_gate"] or "unknown") for r in scen_rows)
        logger.info("--- scenario=%s (n=%d) ---", scenario_name, len(scen_rows))
        for label, count in counts.most_common():
            logger.info("  %-28s %d", label, count)

    by_label: dict[str, dict] = {}
    for r in rows:
        by_label.setdefault(r["label"], {})[r["scenario"]] = r["accepted"]
    both = [lbl for lbl, d in by_label.items() if all(d.get(s[0]) for s in SCENARIOS)]
    one_only = [lbl for lbl, d in by_label.items() if any(d.get(s[0]) for s in SCENARIOS) and lbl not in both]
    logger.info("Accepted under BOTH scenarios: %d -- %s", len(both), both)
    logger.info("Accepted under ONE scenario only (latency-sensitive): %d -- %s", len(one_only), one_only)


if __name__ == "__main__":
    main()
