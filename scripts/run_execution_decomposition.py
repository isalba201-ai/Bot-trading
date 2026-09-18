#!/usr/bin/env python
"""Step 8 (audit+research addendum): runs the execution-decomposition
ladder (research/decomposition.py) against a bounded, deliberately chosen
set of Step 7's own FDR-significant candidates — the one that reached
candidacy gate 2, plus the top few per asset/timeframe by p-value — to
answer "where exactly does the edge disappear" and classify each one
A/B/C (or survives/insufficient-data). Never re-derives raw win rates:
reads them straight from the already-stored ``condition_trials`` rows.

Example:
    python scripts/run_execution_decomposition.py --run-id-prefix step7-20260918T161113 --top-n 3
"""

from __future__ import annotations

import argparse
from collections import Counter

from otc_research.config import load_config
from otc_research.db.models import ConditionTrial
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.research.decomposition import run_decomposition
from otc_research.research.discovery import Condition
from otc_research.utils.logging import get_logger
from otc_research.utils.timeframes import timeframe_to_seconds

logger = get_logger(__name__)

DEFAULT_PAYOUT = 0.85


def _select_candidates(session, run_id_prefix: str, top_n: int, min_sample_size: int):
    rows = (
        session.query(ConditionTrial)
        .filter(
            ConditionTrial.run_id.like(f"{run_id_prefix}%"),
            ConditionTrial.fdr_significant.is_(True),
            ConditionTrial.sample_size >= min_sample_size,
            # A significant LOW win rate for this target_col is not a
            # separate finding -- by construction (call_wins_h and
            # put_wins_h are near-perfect complements at the naive-label
            # level, see PREDICTABILITY_AUDIT.md), it's the same
            # underlying pattern already present, with a genuinely high
            # win rate, under the opposite target_col. Only the
            # winning-direction row is a meaningful decomposition
            # candidate -- including both would silently double-count
            # one market pattern as two "significant conditions".
            ConditionTrial.win_rate > 0.5,
        )
        .all()
    )
    by_key: dict[tuple, list[ConditionTrial]] = {}
    for row in rows:
        key = (row.asset, row.timeframe, row.target_col)
        by_key.setdefault(key, []).append(row)

    selected: list[ConditionTrial] = []
    for key, group in by_key.items():
        group.sort(key=lambda r: (r.p_value if r.p_value is not None else 1.0))
        selected.extend(group[:top_n])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id-prefix", required=True, help="e.g. 'step7-20260918T161113'")
    parser.add_argument("--top-n", type=int, default=3, help="Top N candidates per (asset, timeframe, target_col)")
    parser.add_argument("--min-sample-size", type=int, default=100)
    parser.add_argument("--payout", type=float, default=DEFAULT_PAYOUT)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    candidates = _select_candidates(session, args.run_id_prefix, args.top_n, args.min_sample_size)
    logger.info("selected %d candidates for decomposition", len(candidates))

    all_results = []
    for row in candidates:
        condition = Condition.from_json(row.condition_json)
        direction = "CALL" if row.target_col.startswith("call_wins") else "PUT"
        horizon = int(row.target_col.rsplit("_", 1)[-1])
        expiry_seconds = horizon * timeframe_to_seconds(row.timeframe)

        result = run_decomposition(
            session,
            condition,
            direction,
            expiry_seconds,
            row.asset,
            row.timeframe,
            config.backtest,
            feature_set_version=FEATURE_SET_VERSION,
            payout=args.payout,
            raw_win_rate=row.win_rate,
            raw_sample_size=row.sample_size,
            rng_seed=args.rng_seed,
            label=f"{row.target_col}:{condition.label()}",
        )

        logger.info(
            "=== %s/%s %s h=%d %s -- classification=%s ===",
            row.asset, row.timeframe, direction, horizon, condition.label(), result.classification,
        )
        for rung in result.rungs:
            logger.info(
                "  %-16s n=%-5d win_rate=%s margin=%s expectancy=%s",
                rung.name,
                rung.sample_size,
                f"{rung.win_rate:.4f}" if rung.win_rate is not None else "n/a",
                f"{rung.margin_over_break_even:+.4f}" if rung.margin_over_break_even is not None else "n/a",
                f"{rung.expectancy:+.4f}" if rung.expectancy is not None else "n/a",
            )
        all_results.append(result)

    logger.info("=" * 78)
    logger.info("CLASSIFICATION SUMMARY (n=%d candidates, winning direction only)", len(all_results))
    counts: Counter = Counter(r.classification for r in all_results)
    for label, count in counts.most_common():
        logger.info("  %-28s %d (%.1f%%)", label, count, 100.0 * count / len(all_results))

    logger.info("=" * 78)
    logger.info("DELAY-SENSITIVITY TABLE: fraction of candidates still clearing margin>=0 at each rung")
    rung_order = ["raw", "optimistic", "delay_only_1", "delay_only_2", "delay_only_3", "delay_only_5",
                  "slippage_only", "realistic", "pessimistic"]
    for rung_name in rung_order:
        clearing = 0
        total = 0
        for r in all_results:
            match = next((x for x in r.rungs if x.name == rung_name), None)
            if match is None or match.margin_over_break_even is None:
                continue
            total += 1
            if match.margin_over_break_even >= 0:
                clearing += 1
        frac = clearing / total if total else float("nan")
        logger.info("  %-16s %d/%d (%.1f%%) still clear break-even margin>=0", rung_name, clearing, total, 100.0 * frac)


if __name__ == "__main__":
    main()
