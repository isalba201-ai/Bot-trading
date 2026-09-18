#!/usr/bin/env python
"""Step 9 addendum: the "backtest masivo" — evaluates every hand-designed
H1-H20 strategy, plus H9's own (hour, direction) family via a restricted
discovery run, against every real ingested (asset, timeframe) dataset and
every h-derived expiry, through the exact same corrected (delay-only)
four-gate ``research.candidacy.evaluate_candidacy`` funnel already used
for discovered conditions. Writes one JSON line per evaluation to
``--output`` (default ``data/binary_options_backtest_results.jsonl``) —
structured, not a text log to be regex-parsed later (see
``EXPIRY_UNIVERSE_AUDIT.md``'s own build_expiry_matrix.py for why that was
fragile the first time).

Does NOT re-run the already-corrected discovered-condition evaluations
(the 260-evaluation rerun from ``run_candidacy_corrected_rerun.py``) or
the triple-barrier auxiliary numbers — those are REUSED as-is in the
report step (``build_binary_options_report.py``), specifically so the
already-TESTed EUR/USD 15m session-conditioned candidate's TEST split is
never touched a second time, matching the TEST-once discipline verbatim.

No new strategy logic is invented here — every H1-H20 class is used
exactly as already documented in STRATEGIES.md; the only things new are
(a) the corrected execution model, (b) sweeping expiry across h={1,2,3,5}
instead of assuming each strategy's own hardcoded 300s default, and (c)
optionally, a NO_TRADE (volatility-contraction) filtered variant for
whatever clears gate 2 unfiltered.

Example:
    python scripts/run_binary_options_backtest.py --include-h9 --include-no-trade-filter
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

from otc_research.backtest.engine import run_backtest
from otc_research.backtest.execution import delay_only_scenario
from otc_research.backtest.splits import compute_temporal_split
from otc_research.backtest.walkforward import generate_folds
from otc_research.config import load_config
from otc_research.db.models import Candle
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.research.candidacy import (
    DEFAULT_ENTRY_DELAY_CANDLES,
    CandidacyThresholds,
    CandidacyVerdict,
    evaluate_candidacy,
    evaluate_condition_candidacy,
)
from otc_research.research.dataset import build_dataset
from otc_research.research.discovery import Condition, run_discovery
from otc_research.research.filtered_strategy import FilteredStrategy, volatility_contraction_filter
from otc_research.research.performance_report import build_performance_report
from otc_research.research.strategy_candidacy import SUPPORTED_CODES, build_candidacy_inputs
from otc_research.utils.logging import get_logger
from otc_research.utils.timeframes import timeframe_to_seconds

logger = get_logger(__name__)

ASSET_TIMEFRAMES: tuple[tuple[str, str], ...] = (
    ("EUR_USD", "1m"), ("EUR_USD", "5m"), ("EUR_USD", "15m"), ("EUR_USD", "1h"),
    ("GBP_USD", "5m"), ("GBP_USD", "1h"),
    ("USD_JPY", "5m"), ("USD_JPY", "1h"),
)
HORIZONS: tuple[int, ...] = (1, 2, 3, 5)  # no a priori expiry assumption -- see EXPIRY_UNIVERSE_AUDIT.md

DEFAULT_PAYOUT = 0.85  # gating payout; report step sweeps {70,75,80,85,90}% post-hoc
N_WALK_FORWARD_FOLDS = 5
DEFAULT_OUTPUT = "data/binary_options_backtest_results.jsonl"


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


def _classify(verdict: CandidacyVerdict, thresholds: CandidacyThresholds) -> str:
    """REJECTED / EXPLORATORY / PROMISING_BUT_UNPROVEN / ACCEPTED, derived
    mechanically from CandidacyVerdict fields -- see the Step 9 addendum
    plan section 5 for the exact mapping and its rationale.
    """
    if verdict.accepted:
        return "ACCEPTED"
    if verdict.rejected_at_gate == "test":
        return "PROMISING_BUT_UNPROVEN"
    if verdict.rejected_at_gate == "sample_size_and_margin":
        if verdict.train_sample_size is not None and verdict.train_sample_size < thresholds.min_sample_size:
            return "EXPLORATORY"
        return "REJECTED"
    if verdict.rejected_at_gate == "walk_forward":
        wf = verdict.walk_forward
        if wf is not None and wf.n_folds_sufficiently_sampled < thresholds.min_folds_sampled:
            return "EXPLORATORY"
        return "REJECTED"
    return "REJECTED"  # robustness gate, or any other rejection


def _verdict_to_row(verdict: CandidacyVerdict, classification: str, **meta) -> dict:
    row = dict(meta)
    row["classification"] = classification
    row["accepted"] = verdict.accepted
    row["rejected_at_gate"] = verdict.rejected_at_gate
    row["reason"] = verdict.reason
    row["train_sample_size"] = verdict.train_sample_size
    row["train_win_rate"] = verdict.train_win_rate
    row["train_margin_over_break_even"] = verdict.train_margin_over_break_even
    if verdict.robustness is not None:
        row["robustness_classification"] = verdict.robustness.classification
        row["robustness_fraction_with_edge"] = verdict.robustness.fraction_with_edge
        row["robustness_n_sufficiently_sampled"] = verdict.robustness.n_sufficiently_sampled
    if verdict.walk_forward is not None:
        row["wf_fraction_folds_with_edge"] = verdict.walk_forward.fraction_folds_with_edge
        row["wf_worst_fold_win_rate"] = verdict.walk_forward.worst_fold_win_rate
        row["wf_n_folds_sufficiently_sampled"] = verdict.walk_forward.n_folds_sufficiently_sampled
    row["test_sample_size"] = verdict.test_sample_size
    row["test_win_rate"] = verdict.test_win_rate
    row["test_win_rate_ci_low"] = verdict.test_win_rate_ci_low
    return row


def _maybe_performance_row(session, strategy, asset, timeframe, backtest_config, payout, rng_seed):
    """Only for gate-2+ survivors (per the user's "not every strategy"
    instruction) -- re-runs TRAIN once more (cheap, bounded set) purely to
    get the Trade list, since evaluate_candidacy doesn't return it.
    """
    result = run_backtest(
        session, strategy, asset, timeframe, backtest_config,
        feature_set_version=FEATURE_SET_VERSION, split="train",
        scenarios=[delay_only_scenario(DEFAULT_ENTRY_DELAY_CANDLES)], rng_seed=rng_seed,
    )[0]
    report = build_performance_report(result.trades, payout)
    return {
        "perf_n_trades": report.n_trades,
        "perf_win_rate": report.win_rate,
        "perf_profit_factor": report.profit_factor,
        "perf_cumulative_return_r": report.cumulative_return_r,
        "perf_max_drawdown_r": report.max_drawdown_r,
        "perf_max_consecutive_losses": report.max_consecutive_losses,
        "perf_trades_per_day": report.trades_per_day,
    }


def _run_h1_h20(session, config, args, thresholds, fold_cache, out):
    for code in SUPPORTED_CODES:
        for asset, timeframe in ASSET_TIMEFRAMES:
            fold_key = (asset, timeframe)
            if fold_key not in fold_cache:
                fold_cache[fold_key] = _walk_forward_folds(session, asset, timeframe, config.backtest)
            folds = fold_cache[fold_key]
            for h in HORIZONS:
                expiry_seconds = h * timeframe_to_seconds(timeframe)
                base_strategy, strategy_factory, param_grid = build_candidacy_inputs(code, expiry_seconds)
                verdict = evaluate_candidacy(
                    session, base_strategy, asset, timeframe, config.backtest,
                    feature_set_version=FEATURE_SET_VERSION, payout=args.payout,
                    walk_forward_folds=folds, strategy_factory=strategy_factory,
                    perturbation_param_grid=param_grid, thresholds=thresholds,
                    rng_seed=args.rng_seed,
                )
                classification = _classify(verdict, thresholds)
                row = _verdict_to_row(
                    verdict, classification,
                    family="H1-H20", code=code, label=base_strategy.label, direction="MIXED",
                    asset=asset, timeframe=timeframe, h=h, expiry_seconds=expiry_seconds,
                    payout_used=args.payout,
                )
                out.write(json.dumps(row) + "\n")
                logger.info(
                    "H1-H20 %s %s/%s h=%d -- n=%s wr=%s %s",
                    code, asset, timeframe, h, verdict.train_sample_size,
                    f"{verdict.train_win_rate:.3f}" if verdict.train_win_rate is not None else "n/a",
                    classification,
                )

                if args.include_no_trade_filter and verdict.robustness is not None:
                    _run_no_trade_variant(
                        session, config, args, thresholds, folds, base_strategy, strategy_factory,
                        param_grid, asset, timeframe, h, expiry_seconds,
                        family="H1-H20", code=code, out=out,
                    )

                if verdict.robustness is not None:
                    perf = _maybe_performance_row(
                        session, base_strategy, asset, timeframe, config.backtest, args.payout, args.rng_seed
                    )
                    out.write(json.dumps({
                        "family": "H1-H20-PERFORMANCE", "code": code, "label": base_strategy.label,
                        "asset": asset, "timeframe": timeframe, "h": h, "expiry_seconds": expiry_seconds,
                        **perf,
                    }) + "\n")


def _run_no_trade_variant(
    session, config, args, thresholds, folds, base_strategy, strategy_factory, param_grid,
    asset, timeframe, h, expiry_seconds, *, family, code, out, direction=None,
):
    filt = volatility_contraction_filter()
    filtered_base = FilteredStrategy(base_strategy, filt, filter_label="vol_contraction")

    def filtered_factory(**kwargs):
        return FilteredStrategy(strategy_factory(**kwargs), filt, filter_label="vol_contraction")

    verdict = evaluate_candidacy(
        session, filtered_base, asset, timeframe, config.backtest,
        feature_set_version=FEATURE_SET_VERSION, payout=args.payout,
        walk_forward_folds=folds, strategy_factory=filtered_factory,
        perturbation_param_grid=param_grid, thresholds=thresholds, rng_seed=args.rng_seed,
    )
    classification = _classify(verdict, thresholds)
    row = _verdict_to_row(
        verdict, classification,
        family=f"{family}-NOTRADE", code=code, label=filtered_base.label,
        direction=direction or "MIXED", asset=asset, timeframe=timeframe, h=h,
        expiry_seconds=expiry_seconds, payout_used=args.payout,
    )
    out.write(json.dumps(row) + "\n")
    logger.info(
        "NOTRADE-variant %s %s %s/%s h=%d -- n=%s %s",
        family, code, asset, timeframe, h, verdict.train_sample_size, classification,
    )


def _hour_conditions() -> list[Condition]:
    # One bin per literal hour value, 0..23 -- exactly H9SessionBias's own
    # decide() logic (int(hour_utc) == fixed hour), reused via discovery.py
    # (see discovery.py's `conditions=` parameter, added for this) rather
    # than re-implemented. hour_utc==0 needs the (-1, 0] half-open bin
    # since Condition.matches_row is `low < x <= high`.
    return [Condition(parts=(("hour_utc", float(h - 1), float(h)),)) for h in range(24)]


def _run_h9(session, config, args, thresholds, fold_cache, out):
    hour_conditions = _hour_conditions()
    run_stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    for asset, timeframe in ASSET_TIMEFRAMES:
        fold_key = (asset, timeframe)
        if fold_key not in fold_cache:
            fold_cache[fold_key] = _walk_forward_folds(session, asset, timeframe, config.backtest)
        folds = fold_cache[fold_key]
        for h in HORIZONS:
            df = build_dataset(session, asset, timeframe, feature_set_version=FEATURE_SET_VERSION, horizons=(h,))
            if df.empty:
                continue
            split = compute_temporal_split(
                len(df), config.backtest.train_fraction, config.backtest.validation_fraction
            )
            train_df = df.iloc[split.train_slice]
            expiry_seconds = h * timeframe_to_seconds(timeframe)

            for target_col in (f"call_wins_{h}", f"put_wins_{h}"):
                run_id = f"h9-restricted-{run_stamp}:{asset}:{timeframe}:{target_col}"
                results = run_discovery(
                    session, train_df, ["hour_utc"], target_col,
                    asset=asset, timeframe=timeframe, feature_set_version=FEATURE_SET_VERSION,
                    min_sample_size=50, fdr_q=0.05, run_id=run_id, conditions=hour_conditions,
                )
                significant = [
                    r for r in results
                    if r.fdr_significant and r.stat.n >= 100 and r.stat.win_rate is not None and r.stat.win_rate > 0.5
                ]
                logger.info(
                    "H9 restricted discovery %s/%s %s h=%d -- trials=24 fdr_significant_winning=%d",
                    asset, timeframe, target_col, h, len(significant),
                )
                direction = "CALL" if target_col.startswith("call_wins") else "PUT"
                for r in significant:
                    verdict = evaluate_condition_candidacy(
                        session, r.condition, direction, expiry_seconds, asset, timeframe,
                        config.backtest, feature_set_version=FEATURE_SET_VERSION, payout=args.payout,
                        walk_forward_folds=folds, thresholds=thresholds, rng_seed=args.rng_seed,
                        label=f"H9:{run_id}:{r.condition.label()}",
                    )
                    classification = _classify(verdict, thresholds)
                    row = _verdict_to_row(
                        verdict, classification,
                        family="H9", code="H9", label=f"H9_session_bias_{r.condition.label()}",
                        direction=direction, asset=asset, timeframe=timeframe, h=h,
                        expiry_seconds=expiry_seconds, payout_used=args.payout,
                    )
                    out.write(json.dumps(row) + "\n")
                    logger.info(
                        "H9 %s/%s %s h=%d %s -- n=%s %s",
                        asset, timeframe, direction, h, r.condition.label(),
                        verdict.train_sample_size, classification,
                    )

                    if args.include_no_trade_filter and verdict.robustness is not None:
                        from otc_research.research.condition_strategy import ConditionStrategy

                        base_strategy = ConditionStrategy(r.condition, direction, expiry_seconds)

                        def strategy_factory(edge_perturbation_pct: float, _c=r.condition, _d=direction, _e=expiry_seconds):
                            return ConditionStrategy(_c, _d, _e, edge_perturbation_pct=edge_perturbation_pct)

                        param_grid = [
                            {"edge_perturbation_pct": pct} for pct in thresholds.edge_perturbation_grid
                        ]
                        _run_no_trade_variant(
                            session, config, args, thresholds, folds, base_strategy, strategy_factory,
                            param_grid, asset, timeframe, h, expiry_seconds,
                            family="H9", code="H9", out=out, direction=direction,
                        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--payout", type=float, default=DEFAULT_PAYOUT)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--include-h9", action="store_true")
    parser.add_argument("--include-no-trade-filter", action="store_true")
    parser.add_argument("--skip-h1-h20", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()
    thresholds = CandidacyThresholds()
    fold_cache: dict[tuple, list] = {}

    with open(args.output, "a") as out:
        if not args.skip_h1_h20:
            _run_h1_h20(session, config, args, thresholds, fold_cache, out)
        if args.include_h9:
            _run_h9(session, config, args, thresholds, fold_cache, out)

    logger.info("done -- results appended to %s", args.output)


if __name__ == "__main__":
    main()
