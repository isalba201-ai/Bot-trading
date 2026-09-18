#!/usr/bin/env python
"""H21 backtest (user-requested, 2026-09-18): CCI(20)+RSI(14) overbought
confluence confirmed by a bearish MACD histogram AND a red candle -> PUT
(``strategies/h21_cci_rsi_macd_reversal.py``), tested across every real
ingested dataset at both requested expiries (h=2 and h=4 candles).

Reports two things per (asset, timeframe, h), kept explicitly separate:

1. **RAW tally** — exactly what was asked for directly: how many times
   did this signal fire, and of the ones that resolved, how many WON vs
   LOST. Computed over TRAIN+VALIDATION (80% of each dataset's real
   history) — TEST is intentionally excluded here so it stays untouched
   for the gated evaluation below, never touched twice.
2. **Gated evaluation** — the exact same four-gate candidacy funnel
   already applied to every other strategy in this project (sample/margin
   -> robustness -> walk-forward -> TEST), so "is this reliable" is
   answered by the established discipline, not the raw win rate alone.

No new indicator/strategy variant is invented here beyond exactly what
h21_cci_rsi_macd_reversal.py implements; results are printed as JSON for
direct transcription into the write-up, nothing is summarized away.
"""

from __future__ import annotations

import argparse
import json

from otc_research.backtest.engine import run_backtest
from otc_research.backtest.execution import delay_only_scenario
from otc_research.backtest.walkforward import generate_folds
from otc_research.config import load_config
from otc_research.db.models import Candle
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.research.candidacy import (
    DEFAULT_ENTRY_DELAY_CANDLES,
    CandidacyThresholds,
    evaluate_candidacy,
)
from otc_research.research.strategy_candidacy import build_candidacy_inputs
from otc_research.utils.logging import get_logger
from otc_research.utils.timeframes import timeframe_to_seconds

logger = get_logger(__name__)

ASSET_TIMEFRAMES: tuple[tuple[str, str], ...] = (
    ("EUR_USD", "1m"), ("EUR_USD", "5m"), ("EUR_USD", "15m"), ("EUR_USD", "1h"),
    ("GBP_USD", "5m"), ("GBP_USD", "1h"),
    ("USD_JPY", "5m"), ("USD_JPY", "1h"),
)
HORIZONS: tuple[int, ...] = (2, 4)  # exactly what the user requested
DEFAULT_PAYOUT = 0.85
N_WALK_FORWARD_FOLDS = 5


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


def _raw_tally(session, strategy, asset, timeframe, config, rng_seed):
    """TRAIN + VALIDATION combined (80% of history) -- TEST untouched."""
    n = wins = losses = 0
    for split in ("train", "validation"):
        result = run_backtest(
            session, strategy, asset, timeframe, config,
            feature_set_version=FEATURE_SET_VERSION, split=split,
            scenarios=[delay_only_scenario(DEFAULT_ENTRY_DELAY_CANDLES)], rng_seed=rng_seed,
        )[0]
        for trade in result.trades:
            if trade.result == "WIN":
                n += 1
                wins += 1
            elif trade.result == "LOSS":
                n += 1
                losses += 1
            # VOID (signal dropped / insufficient data at series end) excluded
    return {"opportunities_resolved": n, "wins": wins, "losses": losses,
            "win_rate": wins / n if n else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--payout", type=float, default=DEFAULT_PAYOUT)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()
    thresholds = CandidacyThresholds()
    fold_cache: dict[tuple, list] = {}

    results = []
    for asset, timeframe in ASSET_TIMEFRAMES:
        fold_key = (asset, timeframe)
        if fold_key not in fold_cache:
            fold_cache[fold_key] = _walk_forward_folds(session, asset, timeframe, config.backtest)
        folds = fold_cache[fold_key]

        for h in HORIZONS:
            expiry_seconds = h * timeframe_to_seconds(timeframe)
            base_strategy, strategy_factory, param_grid = build_candidacy_inputs("H21", expiry_seconds)

            raw = _raw_tally(session, base_strategy, asset, timeframe, config.backtest, args.rng_seed)

            verdict = evaluate_candidacy(
                session, base_strategy, asset, timeframe, config.backtest,
                feature_set_version=FEATURE_SET_VERSION, payout=args.payout,
                walk_forward_folds=folds, strategy_factory=strategy_factory,
                perturbation_param_grid=param_grid, thresholds=thresholds,
                rng_seed=args.rng_seed,
            )

            row = {
                "asset": asset, "timeframe": timeframe, "h": h, "expiry_seconds": expiry_seconds,
                "raw_tally": raw,
                "gate1_train_n": verdict.train_sample_size,
                "gate1_train_win_rate": verdict.train_win_rate,
                "gate1_train_margin": verdict.train_margin_over_break_even,
                "accepted": verdict.accepted,
                "rejected_at_gate": verdict.rejected_at_gate,
                "reason": verdict.reason,
                "robustness_classification": verdict.robustness.classification if verdict.robustness else None,
                "wf_fraction_folds_with_edge": verdict.walk_forward.fraction_folds_with_edge if verdict.walk_forward else None,
                "wf_worst_fold_win_rate": verdict.walk_forward.worst_fold_win_rate if verdict.walk_forward else None,
                "test_sample_size": verdict.test_sample_size,
                "test_win_rate": verdict.test_win_rate,
                "test_win_rate_ci_low": verdict.test_win_rate_ci_low,
            }
            results.append(row)
            logger.info("H21 %s/%s h=%d -- raw: n=%s W=%s L=%s wr=%s | gated: accepted=%s rejected_at=%s",
                         asset, timeframe, h, raw["opportunities_resolved"], raw["wins"], raw["losses"],
                         f"{raw['win_rate']:.3f}" if raw["win_rate"] is not None else "n/a",
                         verdict.accepted, verdict.rejected_at_gate)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
