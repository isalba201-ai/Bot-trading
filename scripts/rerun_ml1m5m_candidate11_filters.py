#!/usr/bin/env python
"""Second controlled phase of ML_1M5M candidate #11 investigation
(user-requested, 2026-09-23): tests whether two PRE-SPECIFIED filter
hypotheses -- never optimized here, never combined -- change candidate
#11's stability, specifically Fold 3's weak walk-forward performance.
Does NOT modify the model, its features, its hyperparameters, its
target, or its base threshold in any way.

Three strategies, built once, from the SAME refit (deterministic,
seed=0) GradientBoostingClassifier:
  1. Baseline  -- ModelStrategy(..., probability_threshold=0.5)  (unchanged #11)
  2. P>0.65    -- ModelStrategy(..., probability_threshold=0.65) (same model object)
  3. CCI>=-60  -- FilteredStrategy(baseline, cci_extreme_oversold_filter(-60.0))

Each is evaluated on TRAIN (the corrected gate-1 range) and the 4
corrected walk-forward folds, under both delay0 and delay1 -- via direct
``simulate()`` calls, never ``evaluate_candidacy``'s gate 2/4 (no
robustness sweep, no TEST attempt is even meaningful for a hard
CCI-exclusion filter, and gate 4 must never run in this phase at all).

TEST safety: every candle load in this script is bounded by
``windows.test_start`` (from ``backtest.engine.compute_split_windows``)
as an EXCLUSIVE upper bound -- ``_load_pretest_candles`` below is the
only candle-loading function this script uses, and it structurally
cannot return a TEST-range candle. This is asserted at runtime (not just
assumed) after every load.
"""

from __future__ import annotations

import sys

from otc_research.backtest.engine import SplitWindows, _load_candles, _load_features, compute_split_windows
from otc_research.backtest.execution import ExecutionScenario
from otc_research.backtest.metrics import break_even_win_rate, payout_adjusted_expectancy
from otc_research.backtest.simulator import simulate
from otc_research.backtest.splits import compute_temporal_split
from otc_research.config import load_config
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.research.filtered_strategy import FilteredStrategy, cci_extreme_oversold_filter
from otc_research.research.model_strategy import ModelStrategy
from otc_research.research.models import fit_and_evaluate

sys.path.insert(0, "scripts")
import run_ml1m5m_experiment as exp  # noqa: E402

BASELINE_THRESHOLD = 0.50
P_FILTER_THRESHOLD = 0.65  # pre-specified, fixed for this phase -- never swept
CCI_FILTER_THRESHOLD = -60.0  # pre-specified, fixed for this phase -- never swept

# The exact baseline numbers from the prior corrected re-run -- this
# script refuses to proceed with the filter comparisons if its own
# baseline reproduction doesn't match these (see main()).
EXPECTED_BASELINE_TRAIN_DELAY0 = {"n": 5612, "wins": 3298, "losses": 2314, "voided": 2}
EXPECTED_BASELINE_TRAIN_DELAY1 = {"n": 5612, "wins": 3202, "losses": 2410, "voided": 2}
EXPECTED_BASELINE_FOLD_WR_DELAY0 = [0.5797198132088058, 0.5630872483221476, 0.5707317073170731, 0.5149359886201992]


def _load_pretest_candles(session, windows: SplitWindows, *, start=None, end=None):
    """The ONLY candle-loading function this script uses. ``end`` is
    clamped to ``windows.test_start`` no matter what the caller passes --
    it is structurally impossible for this function to return a
    TEST-range candle. Asserted (not just relied upon) on every call.
    """
    safe_end = windows.test_start if end is None else min(end, windows.test_start)
    candles = _load_candles(session, exp.ASSET, exp.TIMEFRAME, start=start, end=safe_end)
    for c in candles:
        assert c.timestamp < windows.test_start, (
            f"TEST candle leaked into a pre-test load: {c.timestamp} >= {windows.test_start}"
        )
    return candles


def _stats_for(strategy, candles, features_by_ts, scenario: ExecutionScenario):
    trades = simulate(strategy, candles, features_by_ts, 60, scenario, rng=None)
    resolved = [t for t in trades if t.result in ("WIN", "LOSS")]
    wins = sum(1 for t in resolved if t.result == "WIN")
    losses = sum(1 for t in resolved if t.result == "LOSS")
    voided = sum(1 for t in trades if t.result == "VOID")
    n = wins + losses
    wr = wins / n if n else None
    return {"n": n, "wins": wins, "losses": losses, "voided": voided, "win_rate": wr}


def _wilson(wins, n):
    from otc_research.backtest.metrics import wilson_confidence_interval
    if n == 0:
        return (None, None)
    return wilson_confidence_interval(wins, n)


def _print_row(name, stats, break_even, baseline_n=None):
    n, w, l, v, wr = stats["n"], stats["wins"], stats["losses"], stats["voided"], stats["win_rate"]
    ci_low, ci_high = _wilson(w, n)
    margin = (wr - break_even) if wr is not None else None
    coverage = (n / baseline_n) if baseline_n else None
    parts = [
        f"{name:14s} n={n:5d} WIN={w:5d} LOSS={l:5d} VOID={v:2d}",
        f"WR={wr:.4f}" if wr is not None else "WR=n/a",
        f"CI_low={ci_low:.4f}" if ci_low is not None else "CI_low=n/a",
        f"margin={margin:+.4f}" if margin is not None else "margin=n/a",
    ]
    if coverage is not None:
        parts.append(f"coverage={coverage:.4f} ({coverage*100:.1f}%)")
    print("  " + " ".join(parts))
    return {"n": n, "wins": w, "losses": l, "voided": v, "win_rate": wr,
            "ci_low": ci_low, "ci_high": ci_high, "margin": margin, "coverage": coverage}


def main() -> None:
    config = load_config(None)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()
    break_even = break_even_win_rate(exp.DEFAULT_PAYOUT)
    features_by_ts = _load_features(session, exp.ASSET, exp.TIMEFRAME, FEATURE_SET_VERSION)

    windows = compute_split_windows(
        session, exp.ASSET, exp.TIMEFRAME, config.backtest, start=exp.WINDOW_START, end=exp.WINDOW_END
    )
    print(f"TRAIN={windows.train} VALIDATION={windows.validation} TEST={windows.test}")
    assert windows.train[1] < windows.validation[0]
    assert windows.validation[1] < windows.test[0]
    print("Integrity: TRAIN < VALIDATION < TEST, no overlap -- OK\n")

    # --- refit the exact original model (deterministic, seed=0) ------------
    df = exp._load_windowed_dataset(session)
    split = compute_temporal_split(len(df), config.backtest.train_fraction, config.backtest.validation_fraction)
    train_df, validation_df = df.iloc[split.train_slice], df.iloc[split.validation_slice]
    fit = fit_and_evaluate(
        train_df, validation_df, list(exp.ML1M5M_FEATURES), "call_wins_5",
        family="gradient_boosting", probability_threshold=BASELINE_THRESHOLD, seed=0,
    )
    val_stat = fit.validation_stat_at_threshold
    print(f"Refit VALIDATION check: n={val_stat.n} WR={val_stat.win_rate:.4f} "
          f"(expect n=1703 WR=0.5890 -- matches original candidate #11)\n")

    folds = exp._walk_forward_folds(session, config.backtest)
    for f in folds:
        assert f.test_window[1] <= windows.test_start, f"fold {f.fold_index} invades TEST!"

    baseline_strategy = ModelStrategy(
        fit.model, exp.ML1M5M_FEATURES, "CALL", exp.EXPIRY_SECONDS,
        probability_threshold=BASELINE_THRESHOLD, label="ml1m5m-filters-phase2:baseline",
    )
    p_filter_strategy = ModelStrategy(
        fit.model, exp.ML1M5M_FEATURES, "CALL", exp.EXPIRY_SECONDS,
        probability_threshold=P_FILTER_THRESHOLD, label="ml1m5m-filters-phase2:p_gt_065",
    )
    cci_filter_strategy = FilteredStrategy(
        baseline_strategy, cci_extreme_oversold_filter(CCI_FILTER_THRESHOLD),
        filter_label="cci20_lt_neg60", filter_required_features=frozenset({"cci_20"}),
    )
    strategies = [("Baseline", baseline_strategy), ("P>0.65", p_filter_strategy), ("CCI>=-60", cci_filter_strategy)]

    # TRAIN proper is the gate-1 range: [windows.train[0], windows.validation[0])
    # -- i.e. up to (not including) VALIDATION.
    train_candles = _load_pretest_candles(session, windows, start=windows.train[0], end=windows.validation[0])

    results = {}  # (name, scenario_name) -> {"train": {...}, "folds": [...]}
    for scenario_name, scenario in exp.SCENARIOS:
        print("=" * 100)
        print(f"SCENARIO = {scenario_name} ({scenario.name})")
        print("=" * 100)

        print("--- TRAIN ---")
        baseline_train_stats = None
        for name, strategy in strategies:
            raw = _stats_for(strategy, train_candles, features_by_ts, scenario)
            baseline_n = raw["n"] if name == "Baseline" else baseline_train_stats["n"]
            row = _print_row(name, raw, break_even, baseline_n=baseline_n if name != "Baseline" else None)
            if name == "Baseline":
                baseline_train_stats = raw
            results.setdefault((name, scenario_name), {})["train"] = row

        print("--- WALK-FORWARD (4 folds) ---")
        for name, strategy in strategies:
            fold_rows = []
            for f in folds:
                fold_candles = _load_pretest_candles(session, windows, start=f.test_window[0], end=f.test_window[1])
                raw = _stats_for(strategy, fold_candles, features_by_ts, scenario)
                fold_rows.append(raw)
            print(f"  {name}:")
            baseline_fold_ns = None
            for i, raw in enumerate(fold_rows):
                bn = raw["n"] if name == "Baseline" else None
                _print_row(f"    fold{i}", raw, break_even, baseline_n=bn)
            results[(name, scenario_name)]["folds"] = fold_rows

            n_folds_sampled = sum(1 for r in fold_rows if r["n"] >= 20)
            edges = [r["win_rate"] is not None and r["win_rate"] > break_even for r in fold_rows if r["n"] >= 20]
            n_with_edge = sum(edges)
            wrs = [r["win_rate"] for r in fold_rows if r["win_rate"] is not None]
            agg_wins = sum(r["wins"] for r in fold_rows)
            agg_losses = sum(r["losses"] for r in fold_rows)
            agg_n = agg_wins + agg_losses
            agg_wr = agg_wins / agg_n if agg_n else None
            print(f"    AGGREGATE: folds_sampled={n_folds_sampled} folds_with_edge={n_with_edge} "
                  f"fraction={n_with_edge/len(edges) if edges else None} "
                  f"worst_fold_wr={min(wrs) if wrs else None} best_fold_wr={max(wrs) if wrs else None} "
                  f"pooled_wr={agg_wr} pooled_n={agg_n}")
            print()

    # --- baseline reproducibility gate (section 11) -------------------------
    b0 = results[("Baseline", "delay0")]["train"]
    b1 = results[("Baseline", "delay1")]["train"]
    reproduced = (
        b0["n"] == EXPECTED_BASELINE_TRAIN_DELAY0["n"] and b0["wins"] == EXPECTED_BASELINE_TRAIN_DELAY0["wins"]
        and b1["n"] == EXPECTED_BASELINE_TRAIN_DELAY1["n"] and b1["wins"] == EXPECTED_BASELINE_TRAIN_DELAY1["wins"]
    )
    print("=" * 100)
    if reproduced:
        print("BASELINE REPRODUCIBILITY: OK -- matches the prior corrected run exactly.")
    else:
        print("BASELINE REPRODUCIBILITY: *** MISMATCH *** -- STOP. Do not trust the filter comparison above.")
        print(f"  expected delay0={EXPECTED_BASELINE_TRAIN_DELAY0} got={b0}")
        print(f"  expected delay1={EXPECTED_BASELINE_TRAIN_DELAY1} got={b1}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
