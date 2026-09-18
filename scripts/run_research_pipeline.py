#!/usr/bin/env python
"""Step 7 (approved plan, implementation order point 7): runs the full
statistical-discovery pipeline -- research.dataset -> research.baseline ->
research.discovery (FDR-corrected interaction search, TRAIN only) ->
research.candidacy (the fixed accept/reject bar, TEST touched at most
once per candidate) -- against already-ingested REAL Forex data, and
prints a run summary ending in one of the plan's four scientific
conclusions:

  A: ROBUST EDGE       -- at least one condition passed every candidacy
                           gate, including TEST.
  B: PROMISING         -- discovery found an FDR-significant condition,
                           but nothing reached/passed the TEST gate (e.g.
                           rejected on margin, robustness, or walk-forward).
  C: EDGE VANISHES      -- at least one condition passed gates 1-3 (looks
     OUT-OF-SAMPLE          like a real, robust, replicated edge in-sample)
                           but was rejected specifically at the TEST gate.
  D: NO EVIDENCE        -- discovery found no FDR-significant condition
                           at all, across every asset/timeframe/horizon
                           searched.

Every condition trial (win or lose) is logged to ConditionTrial before
FDR correction, and every candidacy verdict is logged to this run's
console/log output in full -- nothing found here is filtered before
being reported, per the approved plan's explicit "even/especially a null
result must be recorded" requirement (point 7 of
/root/.claude/plans/sequential-sparking-candle.md).

Deliberately scoped for runtime: a fixed, deliberately-chosen 10-feature
subset (not every v4 feature at once -- discovery.py's own docstring asks
callers to restrict this), 2-way interactions only (3-way is the natural
next step if 2-way finds something and there is budget to extend the
search -- not run automatically here), and candidacy is only evaluated
for the top few FDR-significant conditions per asset/timeframe (by
p-value) to keep the number of TEST-touching runs small and deliberate.
"""

from __future__ import annotations

import argparse
import datetime as dt

from otc_research.backtest.metrics import break_even_win_rate
from otc_research.backtest.splits import compute_temporal_split
from otc_research.backtest.walkforward import generate_folds
from otc_research.config import load_config
from otc_research.db.models import Candle
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.research.baseline import unconditional_baseline
from otc_research.research.candidacy import CandidacyThresholds, evaluate_candidacy
from otc_research.research.dataset import DEFAULT_HORIZONS, build_dataset
from otc_research.research.discovery import run_discovery
from otc_research.research.models import run_model_comparison, train_val_frames
from otc_research.utils.logging import get_logger
from otc_research.utils.timeframes import timeframe_to_seconds

logger = get_logger(__name__)

#: A deliberately restricted subset of v4 features spanning momentum,
#: trend, mean-reversion, volatility, and position-in-range -- see this
#: module's docstring for why the search isn't run over all ~40 v4
#: features at once.
CORE_DISCOVERY_FEATURES: tuple[str, ...] = (
    "return_1",
    "return_5",
    "rsi_14",
    "adx_14",
    "atr_expansion_ratio",
    "cci_20",
    "rci_9",
    "bb_pct_b_20",
    "move_size_atr",
    "pct_position_in_range_20",
)

#: Real Forex data already ingested and validated in earlier phases (see
#: STRATEGIES.md's H1-H20 runs) -- this script only reads it, it never
#: fetches anything itself.
ASSET_TIMEFRAMES: tuple[tuple[str, str], ...] = (
    ("EUR_USD", "1m"),
    ("EUR_USD", "5m"),
    ("EUR_USD", "15m"),
    ("GBP_USD", "5m"),
    ("USD_JPY", "5m"),
)

DEFAULT_PAYOUT = 0.85  # typical binary-options broker payout, per BACKTESTING.md
TOP_N_CANDIDATES_PER_DATASET = 3
N_WALK_FORWARD_FOLDS = 5


def _history_bounds(session, asset: str, timeframe: str) -> tuple[dt.datetime, dt.datetime]:
    first_ts = (
        session.query(Candle.timestamp)
        .filter(Candle.asset == asset, Candle.timeframe == timeframe)
        .order_by(Candle.timestamp.asc())
        .first()
    )
    last_ts = (
        session.query(Candle.timestamp)
        .filter(Candle.asset == asset, Candle.timeframe == timeframe)
        .order_by(Candle.timestamp.desc())
        .first()
    )
    if first_ts is None or last_ts is None:
        raise ValueError(f"no candles ingested for {asset}/{timeframe}")
    return first_ts[0], last_ts[0]


def _walk_forward_folds(
    session, asset: str, timeframe: str, backtest_config, *, n_folds: int = N_WALK_FORWARD_FOLDS
):
    """Folds spanning only the train+validation region of history (the
    same fraction ``run_backtest``'s own split reserves for TEST is never
    touched here) -- ``n_folds`` equal-width, non-overlapping windows.
    """
    history_start, history_end = _history_bounds(session, asset, timeframe)
    total_span = history_end - history_start
    train_val_span = total_span * (backtest_config.train_fraction + backtest_config.validation_fraction)
    train_val_end = history_start + train_val_span
    fold_span = train_val_span / n_folds
    return generate_folds(
        history_start, train_val_end, train_span=fold_span, test_span=fold_span
    )


def _run_asset_timeframe(session, asset: str, timeframe: str, config, run_id_prefix: str) -> dict:
    logger.info("=== %s / %s ===", asset, timeframe)
    backtest_config = config.backtest
    timeframe_seconds = timeframe_to_seconds(timeframe)

    df = build_dataset(
        session, asset, timeframe, feature_set_version=FEATURE_SET_VERSION, horizons=DEFAULT_HORIZONS
    )
    if df.empty:
        logger.warning("no dataset rows for %s/%s -- skipping", asset, timeframe)
        return {"asset": asset, "timeframe": timeframe, "skipped": True}

    split = compute_temporal_split(
        len(df), backtest_config.train_fraction, backtest_config.validation_fraction
    )
    train_df = df.iloc[split.train_slice]
    validation_df = df.iloc[split.validation_slice]
    logger.info(
        "dataset rows=%d (train=%d, validation=%d, test=%d)",
        len(df), len(train_df), len(validation_df), len(df) - split.validation_end,
    )

    all_significant: list[dict] = []
    for horizon in DEFAULT_HORIZONS:
        for target_col in (f"call_wins_{horizon}", f"put_wins_{horizon}"):
            baseline = unconditional_baseline(train_df, target_col)
            logger.info(
                "  baseline %-14s n=%-6d win_rate=%s",
                target_col,
                baseline.n,
                f"{baseline.win_rate:.4f}" if baseline.win_rate is not None else "n/a",
            )

            run_id = f"{run_id_prefix}:{asset}:{timeframe}:{target_col}"
            results = run_discovery(
                session,
                train_df,
                list(CORE_DISCOVERY_FEATURES),
                target_col,
                asset=asset,
                timeframe=timeframe,
                feature_set_version=FEATURE_SET_VERSION,
                combo_sizes=(2,),
                n_bins=4,
                min_sample_size=50,
                fdr_q=0.05,
                run_id=run_id,
            )
            significant = [r for r in results if r.fdr_significant and r.stat.n >= 100]
            logger.info(
                "  discovery %-14s trials=%-6d fdr_significant=%d (n>=100: %d)",
                target_col, len(results), sum(1 for r in results if r.fdr_significant), len(significant),
            )
            for r in significant[:TOP_N_CANDIDATES_PER_DATASET]:
                all_significant.append(
                    {
                        "target_col": target_col,
                        "horizon": horizon,
                        "direction": "CALL" if target_col.startswith("call_wins") else "PUT",
                        "condition": r.condition,
                        "stat": r.stat,
                        "p_value": r.p_value,
                    }
                )

    candidacy_verdicts: list[dict] = []
    if all_significant:
        folds = _walk_forward_folds(session, asset, timeframe, backtest_config)
        for entry in all_significant:
            expiry_seconds = entry["horizon"] * timeframe_seconds
            verdict = evaluate_candidacy(
                session,
                entry["condition"],
                entry["direction"],
                expiry_seconds,
                asset,
                timeframe,
                backtest_config,
                feature_set_version=FEATURE_SET_VERSION,
                payout=DEFAULT_PAYOUT,
                walk_forward_folds=folds,
                thresholds=CandidacyThresholds(),
                label=f"{entry['target_col']}:{entry['condition'].label()}",
            )
            logger.info(
                "  CANDIDACY %s direction=%s h=%d train_n=%s train_wr=%s -> "
                "accepted=%s rejected_at=%s (%s)",
                entry["condition"].label(),
                entry["direction"],
                entry["horizon"],
                verdict.train_sample_size,
                f"{verdict.train_win_rate:.4f}" if verdict.train_win_rate is not None else "n/a",
                verdict.accepted,
                verdict.rejected_at_gate,
                verdict.reason,
            )
            candidacy_verdicts.append({"entry": entry, "verdict": verdict})

    model_results = []
    if all_significant:
        # Escalate to models.py (plan point 8) using the single most
        # significant target/condition's target column as the modeling
        # target -- ML is interpretive/confirmatory here, not a second
        # independent search.
        best = min(all_significant, key=lambda e: e["p_value"] if e["p_value"] is not None else 1.0)
        try:
            comparisons = run_model_comparison(
                train_df, validation_df, list(CORE_DISCOVERY_FEATURES), best["target_col"],
                probability_threshold=0.6, seed=0,
            )
            for m in comparisons:
                logger.info(
                    "  MODEL %-20s target=%-14s val_n=%-5d val_win_rate=%s coverage=%.3f",
                    m.family, m.target_col, m.validation_stat_at_threshold.n,
                    f"{m.validation_stat_at_threshold.win_rate:.4f}"
                    if m.validation_stat_at_threshold.win_rate is not None else "n/a",
                    m.validation_coverage,
                )
            model_results = comparisons
        except ValueError as exc:
            logger.info("  MODEL skipped: %s", exc)

    return {
        "asset": asset,
        "timeframe": timeframe,
        "skipped": False,
        "n_significant": len(all_significant),
        "candidacy_verdicts": candidacy_verdicts,
        "model_results": model_results,
    }


def _classify_conclusion(all_results: list[dict]) -> str:
    any_significant = any(r.get("n_significant", 0) > 0 for r in all_results)
    if not any_significant:
        return "D"

    any_accepted = any(
        cv["verdict"].accepted
        for r in all_results
        for cv in r.get("candidacy_verdicts", [])
    )
    if any_accepted:
        return "A"

    any_reached_test = any(
        cv["verdict"].rejected_at_gate == "test"
        for r in all_results
        for cv in r.get("candidacy_verdicts", [])
    )
    if any_reached_test:
        return "C"

    return "B"


CONCLUSION_TEXT = {
    "A": "ROBUST EDGE: at least one discovered condition passed every candidacy "
         "gate (TRAIN sample size/margin, parameter-sensitivity robustness, "
         "walk-forward, and out-of-sample TEST).",
    "B": "PROMISING BUT INSUFFICIENT: discovery found FDR-significant condition(s), "
         "but none reached the TEST gate with an intact edge (rejected earlier, on "
         "margin, robustness, or walk-forward consistency).",
    "C": "EDGE VANISHES OUT-OF-SAMPLE: at least one condition looked like a real, "
         "robust, replicated edge through walk-forward, but failed specifically "
         "at the held-out TEST split.",
    "D": "NO EVIDENCE: no condition, across every asset/timeframe/horizon searched, "
         "survived Benjamini-Hochberg FDR correction at all.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--run-id-prefix", default=None, help="Defaults to a UTC timestamp for this run."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    run_id_prefix = args.run_id_prefix or dt.datetime.now(dt.timezone.utc).strftime("step7-%Y%m%dT%H%M%S")
    logger.info("Step 7 research run starting -- run_id_prefix=%s", run_id_prefix)
    logger.info("Break-even win rate at payout=%.2f: %.4f", DEFAULT_PAYOUT, break_even_win_rate(DEFAULT_PAYOUT))

    all_results = []
    for asset, timeframe in ASSET_TIMEFRAMES:
        result = _run_asset_timeframe(session, asset, timeframe, config, run_id_prefix)
        all_results.append(result)

    conclusion = _classify_conclusion(all_results)
    logger.info("=" * 78)
    logger.info("STEP 7 CONCLUSION: %s", conclusion)
    logger.info(CONCLUSION_TEXT[conclusion])
    logger.info("=" * 78)

    for r in all_results:
        if r.get("skipped"):
            continue
        for cv in r.get("candidacy_verdicts", []):
            entry, verdict = cv["entry"], cv["verdict"]
            logger.info(
                "%s/%s %s h=%d %s -- accepted=%s rejected_at=%s",
                r["asset"], r["timeframe"], entry["direction"], entry["horizon"],
                entry["condition"].label(), verdict.accepted, verdict.rejected_at_gate,
            )


if __name__ == "__main__":
    main()
