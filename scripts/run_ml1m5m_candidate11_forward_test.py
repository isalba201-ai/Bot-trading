#!/usr/bin/env python
"""Forward test of the three FROZEN ML_1M5M candidate #11 variants
(user-requested, 2026-09-23), against a genuinely new, never-before-used
block of EUR_USD 1-minute candles.

This is NOT another exploratory pass: the model is fit exactly once (on
TRAIN only, 2026-07-21->07-29, seed=0, unchanged since Phase 1), the
three strategies' thresholds are hardcoded constants copied verbatim from
the Phase-2 filter investigation, and every metric below is computed in
ONE single pass over the ENTIRE forward block before any of it is
inspected -- there is no code path here that runs part of the block,
looks at the result, and then changes anything before running the rest.

Frozen strategies (identical to Phase 2, no tuning here):
  A. Baseline    -- ModelStrategy(..., probability_threshold=0.50)
  B. P>0.65      -- ModelStrategy(..., probability_threshold=0.65) (same model object)
  C. CCI>=-60    -- FilteredStrategy(A, cci_extreme_oversold_filter(-60.0))

Forward block: EUR_USD 1m, 2026-09-19 09:43 -> 2026-09-22 21:02 (5,000
candles), fetched specifically for this phase, strictly after the last
previously-used EUR_USD/1m candle (2026-09-18 21:01) and disjoint from
TRAIN/VALIDATION/TEST and the unrelated Step10/11 block (2026-09-15 ->
09-18).

Safety, enforced with runtime asserts (not just comments) -- see also
tests/test_ml1m5m_candidate11_forward_test.py:
  - every forward candle's timestamp > LAST_HISTORICAL_CUTOFF
  - the forward block never overlaps TRAIN/VALIDATION/TEST
  - the model is fit exactly once, only on TRAIN
  - the three thresholds are the exact frozen constants below
  - no filter is combined with another
"""

from __future__ import annotations

import datetime as dt
import hashlib
import sys
from collections import defaultdict

import joblib

from otc_research.backtest.engine import _load_candles, _load_features, compute_split_windows
from otc_research.backtest.execution import delay_only_scenario
from otc_research.backtest.metrics import break_even_win_rate, wilson_confidence_interval
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

# --- frozen constants (verbatim from Phase 1/2, never changed here) -------
BASELINE_THRESHOLD = 0.50
P_FILTER_THRESHOLD = 0.65
CCI_FILTER_THRESHOLD = -60.0

# One second after the last EUR_USD/1m candle used by ANY prior phase
# (TRAIN/VALIDATION/TEST or the unrelated Step10/11 09-15->09-18 block).
LAST_HISTORICAL_CUTOFF = dt.datetime(2026, 9, 18, 21, 1, 1)

FROZEN_MODEL_PATH = "data/forward_test_models/ml1m5m_candidate11_gradient_boosting_frozen.joblib"

EXPECTED_VALIDATION_N = 1703
EXPECTED_VALIDATION_WR = 0.5890


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _freeze_model(session, config):
    df = exp._load_windowed_dataset(session)
    split = compute_temporal_split(len(df), config.backtest.train_fraction, config.backtest.validation_fraction)
    train_df, validation_df = df.iloc[split.train_slice], df.iloc[split.validation_slice]

    fit = fit_and_evaluate(
        train_df, validation_df, list(exp.ML1M5M_FEATURES), "call_wins_5",
        family="gradient_boosting", probability_threshold=BASELINE_THRESHOLD, seed=0,
    )
    val_stat = fit.validation_stat_at_threshold
    print(f"Model fit TRAIN: {train_df['timestamp'].min()} -> {train_df['timestamp'].max()} (n={len(train_df)})")
    print(f"Refit VALIDATION check: n={val_stat.n} WR={val_stat.win_rate:.4f} "
          f"(expect n={EXPECTED_VALIDATION_N} WR={EXPECTED_VALIDATION_WR})")
    assert val_stat.n == EXPECTED_VALIDATION_N, "model refit does not match the frozen candidate -- STOP"
    assert abs(val_stat.win_rate - EXPECTED_VALIDATION_WR) < 1e-3, "model refit does not match the frozen candidate -- STOP"

    payload = {
        "model": fit.model,
        "feature_cols": list(exp.ML1M5M_FEATURES),
        "target_col": "call_wins_5",
        "family": "gradient_boosting",
        "seed": 0,
        "train_range": (str(train_df["timestamp"].min()), str(train_df["timestamp"].max())),
        "train_n": len(train_df),
    }
    joblib.dump(payload, FROZEN_MODEL_PATH)
    model_hash = _sha256(FROZEN_MODEL_PATH)
    print(f"Frozen model saved: {FROZEN_MODEL_PATH}")
    print(f"Model SHA-256: {model_hash}")
    return fit.model, model_hash


def _stats(trades):
    resolved = [t for t in trades if t.result in ("WIN", "LOSS")]
    wins = sum(1 for t in resolved if t.result == "WIN")
    losses = sum(1 for t in resolved if t.result == "LOSS")
    voided = sum(1 for t in trades if t.result == "VOID")
    n = wins + losses
    wr = wins / n if n else None
    lo, hi = wilson_confidence_interval(wins, n) if n else (None, None)
    return {"n": n, "wins": wins, "losses": losses, "voided": voided, "win_rate": wr,
            "ci_low": lo, "ci_high": hi, "total_signals": len(trades)}


def main() -> None:
    config = load_config(None)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()
    break_even = break_even_win_rate(exp.DEFAULT_PAYOUT)

    # --- integrity: reconfirm the historical windows haven't moved ---------
    windows = compute_split_windows(session, exp.ASSET, exp.TIMEFRAME, config.backtest,
                                     start=exp.WINDOW_START, end=exp.WINDOW_END)
    print(f"Historical TRAIN={windows.train} VALIDATION={windows.validation} TEST={windows.test}")
    assert windows.train[1] < windows.validation[0] < windows.test[0]

    # --- load the forward block: everything strictly after the cutoff ------
    forward_candles = _load_candles(session, exp.ASSET, exp.TIMEFRAME, start=LAST_HISTORICAL_CUTOFF, end=None)
    assert len(forward_candles) > 0, "no forward data found -- did the fetch/ingest step fail?"
    fwd_start, fwd_end = forward_candles[0].timestamp, forward_candles[-1].timestamp
    print(f"\nForward block: {fwd_start} -> {fwd_end}  (n={len(forward_candles)} candles)")
    assert fwd_start > LAST_HISTORICAL_CUTOFF
    assert fwd_start > windows.test[1], "forward block must start after the historical TEST split"
    assert fwd_start > dt.datetime(2026, 9, 18, 21, 1), "forward block overlaps the Step10/11 Sept block"

    features_by_ts = _load_features(session, exp.ASSET, exp.TIMEFRAME, FEATURE_SET_VERSION)

    model, model_hash = _freeze_model(session, config)

    baseline_strategy = ModelStrategy(
        model, exp.ML1M5M_FEATURES, "CALL", exp.EXPIRY_SECONDS,
        probability_threshold=BASELINE_THRESHOLD, label="ml1m5m-fwd:baseline",
    )
    p_filter_strategy = ModelStrategy(
        model, exp.ML1M5M_FEATURES, "CALL", exp.EXPIRY_SECONDS,
        probability_threshold=P_FILTER_THRESHOLD, label="ml1m5m-fwd:p_gt_065",
    )
    cci_filter_strategy = FilteredStrategy(
        baseline_strategy, cci_extreme_oversold_filter(CCI_FILTER_THRESHOLD),
        filter_label="cci20_lt_neg60", filter_required_features=frozenset({"cci_20"}),
    )
    strategies = [("Baseline", baseline_strategy), ("P>0.65", p_filter_strategy), ("CCI>=-60", cci_filter_strategy)]
    assert len(strategies) == 3
    assert baseline_strategy.probability_threshold == BASELINE_THRESHOLD == 0.50
    assert p_filter_strategy.probability_threshold == P_FILTER_THRESHOLD == 0.65
    assert cci_filter_strategy.base is baseline_strategy  # no combination: wraps the baseline alone

    print(f"\nModel hash: {model_hash}")
    print(f"Frozen thresholds: baseline=P>{BASELINE_THRESHOLD}, B=P>{P_FILTER_THRESHOLD}, C=CCI_20>={CCI_FILTER_THRESHOLD}")
    print(f"Expiry: {exp.EXPIRY_SECONDS}s ({exp.HORIZON} candles)")

    # --- ONE blind pass: simulate every strategy under both delays ---------
    all_trades = {}  # (name, scenario_name) -> list[Trade]
    for scenario_name, scenario in exp.SCENARIOS:
        for name, strategy in strategies:
            trades = simulate(strategy, forward_candles, features_by_ts, 60, scenario, rng=None)
            all_trades[(name, scenario_name)] = trades

    # ============================ AGGREGATE =================================
    print("\n" + "=" * 100)
    print("AGGREGATE FORWARD-TEST RESULTS")
    print("=" * 100)
    baseline_totals = {}
    agg_rows = []
    for scenario_name, _ in exp.SCENARIOS:
        for name, _ in strategies:
            trades = all_trades[(name, scenario_name)]
            s = _stats(trades)
            if name == "Baseline":
                baseline_totals[scenario_name] = s["total_signals"]
            coverage = s["total_signals"] / baseline_totals[scenario_name] if baseline_totals.get(scenario_name) else None
            discarded = (baseline_totals[scenario_name] - s["total_signals"]) if name != "Baseline" else 0
            margin = (s["win_rate"] - break_even) if s["win_rate"] is not None else None
            row = {"strategy": name, "delay": scenario_name, **s, "coverage": coverage,
                   "discarded_by_filter": discarded, "margin": margin}
            agg_rows.append(row)
            print(f"{name:10s} delay={scenario_name:6s} total_signals={s['total_signals']:5d} "
                  f"discarded={discarded:5d} n={s['n']:5d} WIN={s['wins']:5d} LOSS={s['losses']:5d} "
                  f"VOID={s['voided']:3d} WR={s['win_rate']:.4f}" if s["win_rate"] is not None else
                  f"{name:10s} delay={scenario_name:6s} total_signals={s['total_signals']:5d} n=0 (no resolved trades)")
            if s["win_rate"] is not None:
                print(f"           CI=({s['ci_low']:.4f},{s['ci_high']:.4f}) margin={margin:+.4f} "
                      f"coverage={coverage:.4f} ({coverage*100:.1f}%)" if coverage is not None else
                      f"           CI=({s['ci_low']:.4f},{s['ci_high']:.4f}) margin={margin:+.4f}")

    # ============================ PER-DAY ====================================
    print("\n" + "=" * 100)
    print("PER-DAY RESULTS")
    print("=" * 100)
    for scenario_name, _ in exp.SCENARIOS:
        for name, _ in strategies:
            trades = all_trades[(name, scenario_name)]
            by_day = defaultdict(list)
            for t in trades:
                if t.result in ("WIN", "LOSS"):
                    by_day[t.signal_time.date()].append(t)
            for day in sorted(by_day):
                day_trades = by_day[day]
                wins = sum(1 for t in day_trades if t.result == "WIN")
                losses = sum(1 for t in day_trades if t.result == "LOSS")
                n = wins + losses
                wr = wins / n if n else None
                print(f"{day} {name:10s} delay={scenario_name:6s} WIN={wins:4d} LOSS={losses:4d} n={n:4d} "
                      f"WR={wr:.4f}" if wr is not None else f"{day} {name:10s} delay={scenario_name:6s} n=0")

    # ============================ 4 SEQUENTIAL BLOCKS ========================
    print("\n" + "=" * 100)
    print("4 SEQUENTIAL TEMPORAL BLOCKS (equal time quartiles, descriptive only)")
    print("=" * 100)
    total_span = fwd_end - fwd_start
    block_span = total_span / 4
    block_bounds = [(fwd_start + i * block_span, fwd_start + (i + 1) * block_span) for i in range(4)]
    print(f"Block boundaries: {block_bounds}")
    for scenario_name, _ in exp.SCENARIOS:
        for name, _ in strategies:
            trades = all_trades[(name, scenario_name)]
            print(f"-- {name} delay={scenario_name} --")
            for i, (b_start, b_end) in enumerate(block_bounds):
                block_trades = [t for t in trades if b_start <= t.signal_time < b_end and t.result in ("WIN", "LOSS")]
                wins = sum(1 for t in block_trades if t.result == "WIN")
                losses = sum(1 for t in block_trades if t.result == "LOSS")
                n = wins + losses
                wr = wins / n if n else None
                margin = (wr - break_even) if wr is not None else None
                edge = (margin is not None and margin > 0)
                print(f"  block{i} [{b_start} -> {b_end}] WIN={wins:4d} LOSS={losses:4d} n={n:4d} "
                      + (f"WR={wr:.4f} margin={margin:+.4f} edge={edge}" if wr is not None else "n=0"))

    print("\nDone. See scripts/run_ml1m5m_candidate11_forward_test.py's docstring for reproduction details.")


if __name__ == "__main__":
    main()
