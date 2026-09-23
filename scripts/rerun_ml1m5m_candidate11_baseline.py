#!/usr/bin/env python
"""Clean baseline re-run of ML_1M5M candidate #11 (gradient_boosting)
ONLY, after fixing the split-mismatch bug the original run had (see
``ML1M5M_EXPERIMENT_REPORT.md`` and the follow-up investigation): gate 1
(TRAIN), gate 2 (robustness), and gate 4 (TEST) previously defaulted to
the FULL EUR_USD/1m candle history instead of the intended
2026-07-21 -> 2026-08-04 window, and the walk-forward fold boundary was a
calendar-time-proportion *estimate* that could (and did) drift past the
true row-based TRAIN+VALIDATION/TEST boundary.

This script changes NOTHING about candidate #11 itself: same 30
features, same GradientBoostingClassifier(random_state=0) with default
hyperparameters, same P(call_wins_5) > 0.50 -> CALL rule, same 5-minute
expiry. It re-evaluates that exact, already-fitted candidate through the
corrected candidacy funnel, with TEST deliberately never touched
(``allow_test=False``) -- this is a baseline read of gates 1-3 only, not
a validation.
"""

from __future__ import annotations

import sys

from otc_research.backtest.metrics import break_even_win_rate, wilson_confidence_interval
from otc_research.backtest.splits import compute_temporal_split
from otc_research.config import load_config
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.research.candidacy import CandidacyThresholds, evaluate_candidacy
from otc_research.research.model_strategy import ModelStrategy
from otc_research.research.models import fit_and_evaluate

sys.path.insert(0, "scripts")
import run_ml1m5m_experiment as exp  # noqa: E402

from otc_research.backtest.engine import compute_split_windows  # noqa: E402


def _fmt(n, w, l, wr, ci_low=None):
    ci_txt = f", CI_low={ci_low:.4f}" if ci_low is not None else ""
    return f"n={n} WIN={w} LOSS={l} WR={wr:.4f}{ci_txt}"


def main() -> None:
    config = load_config(None)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()
    thresholds = CandidacyThresholds()
    break_even = break_even_win_rate(exp.DEFAULT_PAYOUT)

    print("=" * 90)
    print("SECTION B/C: split windows (row-based, corrected)")
    print("=" * 90)
    windows = compute_split_windows(
        session, exp.ASSET, exp.TIMEFRAME, config.backtest, start=exp.WINDOW_START, end=exp.WINDOW_END
    )
    print(f"TRAIN      : {windows.train[0]} -> {windows.train[1]}")
    print(f"VALIDATION : {windows.validation[0]} -> {windows.validation[1]}")
    print(f"TEST       : {windows.test[0]} -> {windows.test[1]}")
    print(f"n_candles (within window): {windows.n_candles}")
    assert windows.train[1] < windows.validation[0], "TRAIN overlaps VALIDATION!"
    assert windows.validation[1] < windows.test[0], "VALIDATION overlaps TEST!"
    print("Integrity check: TRAIN < VALIDATION < TEST, no overlap -- OK")

    # --- refit the exact original model (deterministic, seed=0) ------------
    df = exp._load_windowed_dataset(session)
    split = compute_temporal_split(len(df), config.backtest.train_fraction, config.backtest.validation_fraction)
    train_df = df.iloc[split.train_slice]
    validation_df = df.iloc[split.validation_slice]
    print()
    print(f"Model-fit TRAIN (windowed dataset, feature/target-complete rows): "
          f"{train_df['timestamp'].min()} -> {train_df['timestamp'].max()} (n={len(train_df)})")
    print(f"Model-fit VALIDATION: {validation_df['timestamp'].min()} -> {validation_df['timestamp'].max()} "
          f"(n={len(validation_df)})")

    fit = fit_and_evaluate(
        train_df, validation_df, list(exp.ML1M5M_FEATURES), "call_wins_5",
        family="gradient_boosting", probability_threshold=exp.MODEL_PROBABILITY_THRESHOLD, seed=0,
    )
    val_stat = fit.validation_stat_at_threshold
    print(f"Refit VALIDATION check: n={val_stat.n} WR={val_stat.win_rate:.4f} "
          f"coverage={fit.validation_coverage:.4f} (expect n=1703 WR=0.5890 coverage=0.449 -- matches original)")

    folds = exp._walk_forward_folds(session, config.backtest)
    print(f"\nWalk-forward folds (corrected, bounded to TRAIN+VALIDATION only): {len(folds)}")
    for f in folds:
        print(f"  fold {f.fold_index}: train={f.train_window} test={f.test_window}")
        assert f.test_window[1] <= windows.test_start, f"fold {f.fold_index} invades TEST!"
    print("Integrity check: no walk-forward fold reaches TEST -- OK")

    print()
    print("=" * 90)
    print("SECTION D/E: candidate #11 baseline, corrected infra, allow_test=False")
    print("=" * 90)

    for scenario_name, scenario in exp.SCENARIOS:
        print()
        print(f"--- scenario={scenario_name} ({scenario.name}) ---")
        base_strategy = ModelStrategy(
            fit.model, exp.ML1M5M_FEATURES, "CALL", exp.EXPIRY_SECONDS,
            probability_threshold=exp.MODEL_PROBABILITY_THRESHOLD,
            label=f"ml1m5m-baseline-corrected:call_wins_5:model_gradient_boosting:{scenario_name}",
        )

        def strategy_factory(probability_threshold: float, _fit=fit, _label=base_strategy.label) -> ModelStrategy:
            return ModelStrategy(
                _fit.model, exp.ML1M5M_FEATURES, "CALL", exp.EXPIRY_SECONDS,
                probability_threshold=probability_threshold, label=_label,
            )

        param_grid = [{"probability_threshold": t} for t in exp.MODEL_THRESHOLD_PERTURBATION_GRID]
        verdict = evaluate_candidacy(
            session, base_strategy, exp.ASSET, exp.TIMEFRAME, config.backtest,
            feature_set_version=exp.FEATURE_SET_VERSION, payout=exp.DEFAULT_PAYOUT,
            walk_forward_folds=folds, strategy_factory=strategy_factory,
            perturbation_param_grid=param_grid, thresholds=thresholds, scenario=scenario, rng_seed=0,
            start=exp.WINDOW_START, end=exp.WINDOW_END, allow_test=False,
        )

        n = verdict.train_sample_size
        wr = verdict.train_win_rate
        w = round(wr * n) if wr is not None else None
        l = (n - w) if w is not None else None
        print(f"TRAIN: {_fmt(n, w, l, wr)} margin_over_break_even={verdict.train_margin_over_break_even}")
        print(f"rejected_at_gate={verdict.rejected_at_gate}")
        print(f"reason: {verdict.reason}")

        if verdict.robustness is not None:
            rb = verdict.robustness
            print(f"ROBUSTNESS: classification={rb.classification} "
                  f"n_points={rb.n_points} n_sufficiently_sampled={rb.n_sufficiently_sampled} "
                  f"n_with_edge={rb.n_with_edge} fraction_with_edge={rb.fraction_with_edge}")

        if verdict.walk_forward is not None:
            wf = verdict.walk_forward
            print(f"WALK-FORWARD SUMMARY: n_folds={wf.n_folds} "
                  f"n_folds_sufficiently_sampled={wf.n_folds_sufficiently_sampled} "
                  f"mean_win_rate={wf.mean_win_rate} stdev_win_rate={wf.stdev_win_rate} "
                  f"worst_fold_win_rate={wf.worst_fold_win_rate} worst_fold_index={wf.worst_fold_index} "
                  f"n_folds_with_edge={wf.n_folds_with_edge} fraction_folds_with_edge={wf.fraction_folds_with_edge}")

        assert verdict.test_sample_size is None, "TEST was touched -- this must never happen in this script!"
        print("TEST: not touched (allow_test=False) -- OK")


if __name__ == "__main__":
    main()
