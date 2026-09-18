#!/usr/bin/env python
"""Re-evaluates the FDR-significant, winning-direction conditions already
logged by Step 7 (``step7-...``) and Step 8d (``step8d-session-...``,
``step8d-triplebarrier-...``) through the CORRECTED
``research.candidacy.evaluate_candidacy`` — which now defaults to a
delay-only execution scenario (no slippage, no signal-drop) and enforces
a minimum walk-forward folds-sampled floor, per
``BINARY_OPTIONS_REFRAME_AUDIT.md`` Section 8.

Never re-runs discovery (the naive-label win rates and FDR significance
flags are execution-model-independent, and are already correctly stored)
-- only the candidacy funnel, which is what the correction changes.

Example:
    python scripts/run_candidacy_corrected_rerun.py \\
        --run-id-prefix step7-20260918T161113 --top-n 3
"""

from __future__ import annotations

import argparse
from collections import Counter

from otc_research.backtest.walkforward import generate_folds
from otc_research.config import load_config
from otc_research.db.models import Candle, ConditionTrial
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.research.candidacy import CandidacyThresholds, evaluate_candidacy
from otc_research.research.discovery import Condition
from otc_research.utils.logging import get_logger
from otc_research.utils.timeframes import timeframe_to_seconds

logger = get_logger(__name__)

DEFAULT_PAYOUT = 0.85
N_WALK_FORWARD_FOLDS = 5


def _select_candidates(session, run_id_prefix: str, top_n: int, min_sample_size: int):
    rows = (
        session.query(ConditionTrial)
        .filter(
            ConditionTrial.run_id.like(f"{run_id_prefix}%"),
            ConditionTrial.fdr_significant.is_(True),
            ConditionTrial.sample_size >= min_sample_size,
            ConditionTrial.win_rate > 0.5,  # winning direction only -- see PREDICTABILITY_AUDIT.md Section 6
        )
        .all()
    )
    by_key: dict[tuple, list[ConditionTrial]] = {}
    for row in rows:
        key = (row.asset, row.timeframe, row.target_col)
        by_key.setdefault(key, []).append(row)

    selected: list[ConditionTrial] = []
    for group in by_key.values():
        group.sort(key=lambda r: (r.p_value if r.p_value is not None else 1.0))
        selected.extend(group[:top_n])
    return selected


def _walk_forward_folds(session, asset: str, timeframe: str, backtest_config, *, n_folds: int = N_WALK_FORWARD_FOLDS):
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
    parser.add_argument(
        "--run-id-prefix", action="append", required=True,
        help="Repeatable. e.g. --run-id-prefix step7-20260918T161113 --run-id-prefix step8d-session-",
    )
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--min-sample-size", type=int, default=100)
    parser.add_argument("--payout", type=float, default=DEFAULT_PAYOUT)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    fold_cache: dict[tuple, list] = {}
    all_verdicts = []

    for prefix in args.run_id_prefix:
        candidates = _select_candidates(session, prefix, args.top_n, args.min_sample_size)
        logger.info("prefix=%s: %d candidates selected", prefix, len(candidates))

        for row in candidates:
            condition = Condition.from_json(row.condition_json)
            direction = "CALL" if row.win_rate is not None and (
                row.target_col.startswith("call_wins") or row.target_col.startswith("barrier_call")
            ) else "PUT"
            # horizon in candles is the trailing integer for naive targets
            # (call_wins_5 -> 5); for barrier targets it's TB_MAX_HORIZON,
            # not encoded in the column name, so fall back to a fixed
            # value consistent with scripts/run_step8d_research.py.
            try:
                horizon = int(row.target_col.rsplit("_", 1)[-1])
            except ValueError:
                horizon = 10  # matches run_step8d_research.py's TB_MAX_HORIZON
            expiry_seconds = horizon * timeframe_to_seconds(row.timeframe)

            fold_key = (row.asset, row.timeframe)
            if fold_key not in fold_cache:
                fold_cache[fold_key] = _walk_forward_folds(session, row.asset, row.timeframe, config.backtest)
            folds = fold_cache[fold_key]

            verdict = evaluate_candidacy(
                session, condition, direction, expiry_seconds, row.asset, row.timeframe,
                config.backtest, feature_set_version=FEATURE_SET_VERSION, payout=args.payout,
                walk_forward_folds=folds, thresholds=CandidacyThresholds(),
                rng_seed=args.rng_seed,
                label=f"corrected:{prefix}:{row.target_col}:{condition.label()}",
            )
            logger.info(
                "CORRECTED %s/%s %s h=%d %s -- train_n=%s train_wr=%s accepted=%s rejected_at=%s",
                row.asset, row.timeframe, direction, horizon, condition.label(),
                verdict.train_sample_size,
                f"{verdict.train_win_rate:.4f}" if verdict.train_win_rate is not None else "n/a",
                verdict.accepted, verdict.rejected_at_gate,
            )
            all_verdicts.append((prefix, row, direction, horizon, condition.label(), verdict))

    logger.info("=" * 78)
    logger.info("CORRECTED CANDIDACY SUMMARY (n=%d evaluations)", len(all_verdicts))
    counts: Counter = Counter(v.rejected_at_gate or "ACCEPTED" for *_x, v in all_verdicts)
    for label, count in counts.most_common():
        logger.info("  %-28s %d (%.1f%%)", label, count, 100.0 * count / len(all_verdicts))

    accepted = [t for t in all_verdicts if t[-1].accepted]
    if accepted:
        logger.info("ACCEPTED CANDIDATES:")
        for prefix, row, direction, horizon, condition_label, verdict in accepted:
            logger.info(
                "  %s %s/%s %s h=%d %s test_wr=%.4f test_ci_low=%.4f",
                prefix, row.asset, row.timeframe, direction, horizon, condition_label,
                verdict.test_win_rate, verdict.test_win_rate_ci_low,
            )
    else:
        logger.info("ACCEPTED CANDIDATES: none")


if __name__ == "__main__":
    main()
