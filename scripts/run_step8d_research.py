#!/usr/bin/env python
"""Step 8d (audit+research addendum): two bounded, literature-informed
re-runs of discovery + candidacy, each with its own run_id and its own
independent Benjamini-Hochberg correction (never pooled with Step 7's
run for significance purposes, no threshold relaxed):

1. **Session-conditioned run**: Step 7's own 10-feature core subset
   extended with hour_utc/trading_session_code/day_of_week -- the one
   concrete, peer-reviewed-adjacent hypothesis from the literature review
   (Gao/Han/Li/Zhou 2018's intraday-momentum finding), on the naive
   call_wins_h/put_wins_h targets exactly as Step 7 used.
2. **Triple-barrier-target run**: Step 7's same 10 core features, target
   replaced with research.targets.triple_barrier_labels (ATR-scaled
   barriers) -- the direct test of whether the naive-target failure mode
   was a labeling artifact, per the literature's strongest actionable
   lead (the triple-barrier/meta-labeling critique of fixed-horizon
   labels). upper_mult/lower_mult/max_horizon are fixed here, in code,
   BEFORE running -- never tuned after seeing a result.

Whatever clears FDR correction with n>=100 is re-evaluated through
research.candidacy's unchanged 4-gate bar, exactly like Step 7's own
candidates were.
"""

from __future__ import annotations

import argparse
import datetime as dt

from otc_research.backtest.walkforward import generate_folds
from otc_research.config import load_config
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.research.candidacy import CandidacyThresholds, evaluate_condition_candidacy
from otc_research.research.dataset import DEFAULT_HORIZONS, build_dataset
from otc_research.research.discovery import run_discovery
from otc_research.research.targets import triple_barrier_labels, triple_barrier_to_binary
from otc_research.utils.logging import get_logger
from otc_research.utils.timeframes import timeframe_to_seconds

logger = get_logger(__name__)

CORE_DISCOVERY_FEATURES: tuple[str, ...] = (
    "return_1", "return_5", "rsi_14", "adx_14", "atr_expansion_ratio",
    "cci_20", "rci_9", "bb_pct_b_20", "move_size_atr", "pct_position_in_range_20",
)
SESSION_FEATURES: tuple[str, ...] = ("hour_utc", "trading_session_code", "day_of_week")

ASSET_TIMEFRAMES: tuple[tuple[str, str], ...] = (
    ("EUR_USD", "1m"), ("EUR_USD", "5m"), ("EUR_USD", "15m"),
    ("GBP_USD", "5m"), ("USD_JPY", "5m"),
)

DEFAULT_PAYOUT = 0.85
TOP_N_CANDIDATES_PER_DATASET = 3
N_WALK_FORWARD_FOLDS = 5

# Triple-barrier parameters -- fixed here, before running, per the
# addendum's explicit "no post-hoc tuning" guardrail.
TB_UPPER_MULT = 1.0
TB_LOWER_MULT = 1.0
TB_MAX_HORIZON = 10  # candles


def _walk_forward_folds(session, asset, timeframe, backtest_config, *, n_folds=N_WALK_FORWARD_FOLDS):
    from otc_research.db.models import Candle

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


def _run_discovery_and_candidacy(session, df, feature_cols, target_cols, asset, timeframe, config, run_id_prefix):
    from otc_research.backtest.splits import compute_temporal_split

    split = compute_temporal_split(len(df), config.backtest.train_fraction, config.backtest.validation_fraction)
    train_df = df.iloc[split.train_slice]
    timeframe_seconds = timeframe_to_seconds(timeframe)

    all_significant = []
    for target_col, horizon in target_cols:
        run_id = f"{run_id_prefix}:{asset}:{timeframe}:{target_col}"
        results = run_discovery(
            session, train_df, list(feature_cols), target_col,
            asset=asset, timeframe=timeframe, feature_set_version=FEATURE_SET_VERSION,
            combo_sizes=(2,), n_bins=4, min_sample_size=50, fdr_q=0.05, run_id=run_id,
        )
        significant = [r for r in results if r.fdr_significant and r.stat.n >= 100 and r.stat.win_rate is not None and r.stat.win_rate > 0.5]
        logger.info(
            "  discovery %-30s trials=%-6d fdr_significant_winning=%d",
            target_col, len(results), len(significant),
        )
        for r in significant[:TOP_N_CANDIDATES_PER_DATASET]:
            direction = "CALL" if target_col.startswith("call_wins") or target_col.startswith("barrier_call") else "PUT"
            all_significant.append({"target_col": target_col, "horizon": horizon, "direction": direction,
                                     "condition": r.condition, "stat": r.stat, "p_value": r.p_value})

    if not all_significant:
        return []

    folds = _walk_forward_folds(session, asset, timeframe, config.backtest)
    verdicts = []
    for entry in all_significant:
        expiry_seconds = entry["horizon"] * timeframe_seconds
        verdict = evaluate_condition_candidacy(
            session, entry["condition"], entry["direction"], expiry_seconds, asset, timeframe,
            config.backtest, feature_set_version=FEATURE_SET_VERSION, payout=DEFAULT_PAYOUT,
            walk_forward_folds=folds, thresholds=CandidacyThresholds(),
            label=f"{run_id_prefix}:{entry['target_col']}:{entry['condition'].label()}",
        )
        logger.info(
            "  CANDIDACY %s %s h=%d train_n=%s train_wr=%s -> accepted=%s rejected_at=%s",
            entry["condition"].label(), entry["direction"], entry["horizon"],
            verdict.train_sample_size,
            f"{verdict.train_win_rate:.4f}" if verdict.train_win_rate is not None else "n/a",
            verdict.accepted, verdict.rejected_at_gate,
        )
        verdicts.append((entry, verdict))
    return verdicts


def run_session_conditioned(session, config, run_id_prefix):
    logger.info("### Session-conditioned run (Gao et al. 2018 intraday-momentum hypothesis) ###")
    features = list(CORE_DISCOVERY_FEATURES) + list(SESSION_FEATURES)
    all_verdicts = []
    for asset, timeframe in ASSET_TIMEFRAMES:
        logger.info("=== %s/%s ===", asset, timeframe)
        df = build_dataset(session, asset, timeframe, feature_set_version=FEATURE_SET_VERSION, horizons=DEFAULT_HORIZONS)
        if df.empty:
            continue
        target_cols = [(f"{d}_wins_{h}", h) for h in DEFAULT_HORIZONS for d in ("call", "put")]
        verdicts = _run_discovery_and_candidacy(session, df, features, target_cols, asset, timeframe, config, run_id_prefix)
        all_verdicts.extend(verdicts)
    return all_verdicts


def run_triple_barrier(session, config, run_id_prefix):
    logger.info("### Triple-barrier-target run (upper=%.1f*ATR lower=%.1f*ATR horizon=%d candles) ###",
                TB_UPPER_MULT, TB_LOWER_MULT, TB_MAX_HORIZON)
    all_verdicts = []
    for asset, timeframe in ASSET_TIMEFRAMES:
        logger.info("=== %s/%s ===", asset, timeframe)
        df = build_dataset(session, asset, timeframe, feature_set_version=FEATURE_SET_VERSION, horizons=DEFAULT_HORIZONS)
        if df.empty:
            continue
        labels = triple_barrier_labels(
            df, atr_col="atr_14", upper_mult=TB_UPPER_MULT, lower_mult=TB_LOWER_MULT, max_horizon=TB_MAX_HORIZON
        )
        call_wins, put_wins = triple_barrier_to_binary(labels)
        df = df.copy()
        df["barrier_call_wins"] = call_wins
        df["barrier_put_wins"] = put_wins
        n_resolved = call_wins.notna().sum()
        logger.info("  triple-barrier labels resolved (not timed out): %d/%d", n_resolved, len(df))

        target_cols = [("barrier_call_wins", TB_MAX_HORIZON), ("barrier_put_wins", TB_MAX_HORIZON)]
        verdicts = _run_discovery_and_candidacy(
            session, df, CORE_DISCOVERY_FEATURES, target_cols, asset, timeframe, config, run_id_prefix
        )
        all_verdicts.extend(verdicts)
    return all_verdicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")

    session_verdicts = run_session_conditioned(session, config, f"step8d-session-{timestamp}")
    tb_verdicts = run_triple_barrier(session, config, f"step8d-triplebarrier-{timestamp}")

    logger.info("=" * 78)
    logger.info("STEP 8D SUMMARY")
    logger.info("session-conditioned run: %d candidacy evaluations, %d accepted",
                len(session_verdicts), sum(1 for _, v in session_verdicts if v.accepted))
    logger.info("triple-barrier run: %d candidacy evaluations, %d accepted",
                len(tb_verdicts), sum(1 for _, v in tb_verdicts if v.accepted))


if __name__ == "__main__":
    main()
