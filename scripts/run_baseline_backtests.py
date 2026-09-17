#!/usr/bin/env python
"""Run every Phase 5 baseline strategy (H1-H8, H10) against one
asset/timeframe and record the result.

Deliberately restricted to ``--split train`` (default) or
``--split validation`` — never ``test``. BACKTESTING.md's out-of-sample
test split must be touched exactly once, deliberately, after every
development decision is final; a bulk multi-hypothesis sweep like this
one is exactly the kind of run that should never be able to touch it by
accident. Use scripts/run_backtest.py directly (one strategy at a time)
for that, when it's actually time.

H9 (session bias) is excluded — it needs an explicit (hour, direction)
pair, not a zero-argument instance; see
otc_research/strategies/h9_session_bias.py.

After each run, if the hypothesis's current status in the database is
still "registered", it's advanced to "tested" — a factual statement that
it has now actually been run through the engine, NOT a claim of edge
(BACKTESTING.md's edge classification needs walk-forward + Monte Carlo +
robustness, none of which exist yet — see BACKTESTING.md's "what's
implemented" table). Nothing here fabricates data: this only does
anything useful once real candles have actually been fetched
(scripts/fetch_market_data.py) and features computed
(scripts/compute_features.py) for the given pair/timeframe.

Example:
    python scripts/run_baseline_backtests.py --pair EUR_USD --timeframe 5m
"""

from __future__ import annotations

import argparse
import datetime as dt

from otc_research.backtest.engine import run_backtest
from otc_research.config import load_config
from otc_research.db.models import Hypothesis
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.strategies import BASELINE_STRATEGIES
from otc_research.utils.logging import get_logger

logger = get_logger(__name__)


def _advance_hypothesis_status(session, hypothesis_code: str, run_ids: list[int]) -> None:
    hypothesis = session.query(Hypothesis).filter_by(code=hypothesis_code).one_or_none()
    if hypothesis is None:
        logger.warning(
            "No Hypothesis row for %s — run scripts/seed_hypotheses.py first; "
            "skipping status update.",
            hypothesis_code,
        )
        return

    note_line = (
        f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] backtest_run_ids={run_ids}"
    )
    hypothesis.notes = f"{hypothesis.notes}\n{note_line}" if hypothesis.notes else note_line

    if hypothesis.status == "registered":
        hypothesis.status = "tested"

    session.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", required=True, help='e.g. "EUR_USD"')
    parser.add_argument("--timeframe", required=True, help='"1m", "5m", "15m", ...')
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--feature-set-version", default=FEATURE_SET_VERSION)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    for code, strategy_cls in BASELINE_STRATEGIES.items():
        strategy = strategy_cls()
        try:
            results = run_backtest(
                session,
                strategy,
                asset=args.pair,
                timeframe=args.timeframe,
                backtest_config=config.backtest,
                feature_set_version=args.feature_set_version,
                split=args.split,
                rng_seed=args.rng_seed,
            )
        except ValueError as exc:
            logger.warning("%s: skipped (%s)", code, exc)
            continue

        for result in results:
            s = result.stats
            logger.info(
                "%s [%s/%s] n=%d win_rate=%s ci=[%s, %s] expectancy_pct=%s voided=%d run_id=%d",
                code,
                args.split,
                result.scenario,
                s.sample_size,
                f"{s.win_rate:.3f}" if s.win_rate is not None else "n/a",
                f"{s.win_rate_ci_low:.3f}" if s.win_rate_ci_low is not None else "n/a",
                f"{s.win_rate_ci_high:.3f}" if s.win_rate_ci_high is not None else "n/a",
                f"{s.expectancy_pct:.4f}" if s.expectancy_pct is not None else "n/a",
                s.voided,
                result.backtest_run_id,
            )

        _advance_hypothesis_status(session, code, [r.backtest_run_id for r in results])


if __name__ == "__main__":
    main()
