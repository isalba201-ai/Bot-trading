#!/usr/bin/env python
"""Point-16 report for the ML_1M5M candidate #11 manual-live test:
theoretical (every CALL the model generated) vs. manual (only the ones
the user actually executed) performance, plus a CSV export of the full
evaluation/signal/decision history. Read-only -- never modifies a Signal
or LiveEvaluation row.

Usage:
    python scripts/report_live_candidate11.py
    python scripts/report_live_candidate11.py --export history.csv
"""

from __future__ import annotations

import argparse
import csv

from otc_research.backtest.metrics import break_even_win_rate, wilson_confidence_interval
from otc_research.config import load_config
from otc_research.db.models import LiveEvaluation, Signal, SignalDecision
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.live import candidate as cand
from otc_research.signals.decisions import TOOK_TRADE


def _print_report(session) -> None:
    evaluations = (
        session.query(LiveEvaluation)
        .filter(LiveEvaluation.strategy_code == cand.CODE)
        .order_by(LiveEvaluation.candle_timestamp.asc())
        .all()
    )
    if not evaluations:
        print("No evaluations recorded yet for", cand.CODE)
        return

    n_total = len(evaluations)
    n_call = sum(1 for e in evaluations if e.signal == "CALL")
    n_no_trade = sum(1 for e in evaluations if e.signal == "NO_TRADE")
    n_data_error = sum(1 for e in evaluations if e.signal == "DATA_ERROR")

    signals = session.query(Signal).filter(Signal.strategy_code == cand.CODE).all()
    resolved = [s for s in signals if s.result in ("WIN", "LOSS")]
    n_win = sum(1 for s in resolved if s.result == "WIN")
    n_loss = sum(1 for s in resolved if s.result == "LOSS")
    n_void = sum(1 for s in signals if s.result == "VOID")
    n_pending = sum(1 for s in signals if s.result is None)
    n_resolved = n_win + n_loss
    wr = n_win / n_resolved if n_resolved else None
    be = break_even_win_rate(cand.PAYOUT)
    margin = (wr - be) if wr is not None else None
    ci = wilson_confidence_interval(n_win, n_resolved) if n_resolved else None

    decisions_by_signal = {d.signal_id: d for d in session.query(SignalDecision).all()}
    executed = [s for s in signals if decisions_by_signal.get(s.id) is not None
                and decisions_by_signal[s.id].decision == TOOK_TRADE]
    n_executed = len(executed)
    manual_resolved = [
        s for s in executed
        if decisions_by_signal[s.id].result in ("WIN", "LOSS")
        or s.result in ("WIN", "LOSS")
    ]
    n_manual_win = sum(
        1 for s in executed
        if (decisions_by_signal[s.id].result or s.result) == "WIN"
    )
    n_manual_loss = sum(
        1 for s in executed
        if (decisions_by_signal[s.id].result or s.result) == "LOSS"
    )
    n_skipped = sum(
        1 for s in signals
        if decisions_by_signal.get(s.id) is None or decisions_by_signal[s.id].decision != TOOK_TRADE
    )
    manual_n = n_manual_win + n_manual_loss
    manual_wr = n_manual_win / manual_n if manual_n else None

    print("=" * 72)
    print("Forward Manual Test -- ML_1M5M candidate #11 (baseline, P>0.50, delay=1)")
    print("=" * 72)
    print(f"Período: {evaluations[0].candle_timestamp} -> {evaluations[-1].candle_timestamp}")
    print(f"Total evaluaciones: {n_total}")
    print(f"CALL: {n_call}   NO_TRADE: {n_no_trade}   DATA_ERROR: {n_data_error}")
    print()
    print("Resultado teórico (todas las señales CALL del modelo)")
    print(f"  WIN: {n_win}  LOSS: {n_loss}  VOID: {n_void}  PENDING: {n_pending}")
    print(f"  WR: {f'{wr:.2%}' if wr is not None else 'n/a'}   "
          f"CI95: {f'({ci[0]:.2%}, {ci[1]:.2%})' if ci else 'n/a'}")
    print(f"  Margin vs {be:.2%}: {f'{margin:+.2%}' if margin is not None else 'n/a'}")
    print()
    print("Resultado manual (solo lo que el usuario ejecutó)")
    print(f"  Ejecutadas: {n_executed}  WIN: {n_manual_win}  LOSS: {n_manual_loss}  Skipped: {n_skipped}")
    print(f"  WR: {f'{manual_wr:.2%}' if manual_wr is not None else 'n/a'}")


def _export_csv(session, path: str) -> None:
    evaluations = (
        session.query(LiveEvaluation)
        .filter(LiveEvaluation.strategy_code == cand.CODE)
        .order_by(LiveEvaluation.candle_timestamp.asc())
        .all()
    )
    signals_by_id = {s.id: s for s in session.query(Signal).filter(Signal.strategy_code == cand.CODE)}
    decisions_by_signal = {d.signal_id: d for d in session.query(SignalDecision).all()}

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candle_timestamp", "signal_time", "probability_call", "signal",
            "entry_time", "expiry_time", "signal_ref", "entry_price", "expiry_price",
            "theoretical_result", "manual_decision", "manual_result", "notes",
        ])
        for e in evaluations:
            sig = signals_by_id.get(e.signal_id) if e.signal_id else None
            decision = decisions_by_signal.get(e.signal_id) if e.signal_id else None
            writer.writerow([
                e.candle_timestamp.isoformat(), e.signal_time.isoformat(),
                e.probability_call, e.signal,
                e.entry_time.isoformat() if e.entry_time else "",
                e.expiry_time.isoformat() if e.expiry_time else "",
                sig.signal_ref if sig else "",
                sig.entry_price if sig else "",
                sig.expiry_price if sig else "",
                sig.result if sig else "",
                decision.decision if decision else "",
                decision.result if decision else "",
                e.notes or "",
            ])
    print(f"Exported {len(evaluations)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--export", default=None, help="write the full history to this CSV path")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    _print_report(session)
    if args.export:
        _export_csv(session, args.export)


if __name__ == "__main__":
    main()
