"""Human-readable rendering for the manual-live candidate #11 signal
generator -- deliberately its own format (not
``signals.formatting.format_signal_message``, which serves the general
multi-strategy paper-test family) because the user specified this exact
layout. Language is explicit that this is a proposal for manual review,
never an instruction or an executed trade -- the same non-imperative
discipline the general formatter already follows.
"""

from __future__ import annotations

import datetime as dt

from otc_research.live import candidate as cand
from otc_research.live.evaluation import Evaluation


def _render(
    *, signal_time: dt.datetime, candle_timestamp: dt.datetime, entry_time: dt.datetime,
    probability_call: float, signal_ref: str | None,
) -> str:
    ref = f" ({signal_ref})" if signal_ref else ""
    return (
        "━" * 22 + "\n"
        f"\U0001F6A8 SEÑAL CALL{ref}\n\n"
        f"{cand.ASSET.replace('_', '/')}\n"
        f"Timeframe: {cand.TIMEFRAME_SECONDS // 60} minuto\n\n"
        f"Señal generada: {signal_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        f"Entrada prevista: {entry_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        f"Expiry: {cand.EXPIRY_SECONDS // 60} minutos\n\n"
        f"Probabilidad CALL: {probability_call:.1%}\n\n"
        f"Modelo: ML_1M5M #11\n"
        f"Threshold: >{cand.PROBABILITY_THRESHOLD:.0%}\n"
        f"Delay: {cand.ENTRY_DELAY_CANDLES} candle\n\n"
        f"Signal candle: {candle_timestamp.strftime('%H:%M')} (close {signal_time.strftime('%H:%M')})\n"
        f"Entry candle: {entry_time.strftime('%H:%M')}\n\n"
        + "━" * 22 + "\n"
        "ESTO NO ES UNA ORDEN AUTOMÁTICA.\n"
        "El modelo está generando una señal CALL y debes decidir/hacer "
        "manualmente la entrada.\n"
        "Todos los timestamps en UTC.\n"
        + "━" * 22
    )


def format_call_signal(evaluation: Evaluation, *, signal_ref: str | None = None) -> str:
    assert evaluation.signal == "CALL"
    return _render(
        signal_time=evaluation.signal_time, candle_timestamp=evaluation.candle_timestamp,
        entry_time=evaluation.entry_time, probability_call=evaluation.probability_call,
        signal_ref=signal_ref,
    )


def format_call_signal_from_row(signal) -> str:
    """Same rendering as ``format_call_signal``, built from a persisted
    ``Signal`` ORM row instead of an ``Evaluation`` -- what every
    ``NotificationProvider`` for this candidate actually receives
    (``NotificationProvider.notify(signal)``'s fixed signature). Derives
    the display fields from ``signal.generated_at`` (the candle-open
    timestamp, this project's standing convention) and the frozen
    candidate's own constants -- never from a value that could drift
    from what was actually used to create the signal.
    """
    candle_timestamp = signal.generated_at
    signal_time = candle_timestamp + dt.timedelta(seconds=cand.TIMEFRAME_SECONDS)
    entry_time = candle_timestamp + dt.timedelta(
        seconds=(1 + cand.ENTRY_DELAY_CANDLES) * cand.TIMEFRAME_SECONDS
    )
    return _render(
        signal_time=signal_time, candle_timestamp=candle_timestamp, entry_time=entry_time,
        probability_call=signal.score, signal_ref=signal.signal_ref,
    )


def format_status_line(
    *,
    last_candle_timestamp: dt.datetime | None,
    last_close: float | None,
    last_probability: float | None,
    last_signal: str | None,
) -> str:
    return (
        "\U0001F7E2 MARKET MONITORING\n"
        f"{cand.ASSET.replace('_', '/')} — {cand.TIMEFRAME}\n\n"
        f"Última vela recibida: {last_candle_timestamp.isoformat() if last_candle_timestamp else 'n/a'}\n"
        f"Último precio: {last_close if last_close is not None else 'n/a'}\n\n"
        f"Última evaluación -- Probabilidad CALL: "
        f"{f'{last_probability:.1%}' if last_probability is not None else 'n/a'}\n"
        f"Señal: {last_signal or 'n/a'}"
    )


def format_stats_block(*, n_call: int, n_win: int, n_loss: int, n_void: int) -> str:
    resolved = n_win + n_loss
    wr = (n_win / resolved) if resolved else None
    return (
        "Estadísticas de la prueba\n"
        f"Señales CALL: {n_call}\n"
        f"WIN: {n_win}\n"
        f"LOSS: {n_loss}\n"
        f"VOID: {n_void}\n"
        f"WR: {f'{wr:.2%}' if wr is not None else 'n/a'}"
    )
