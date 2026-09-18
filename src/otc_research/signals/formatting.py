"""One human-readable rendering of a ``Signal`` row, reused by every
``NotificationProvider`` (approved plan point 9: "one
format_signal_message() function, not duplicated per provider") --
adding a new channel (desktop, Telegram, ...) never means re-deriving
this text.

Language is deliberately descriptive, never imperative or pressuring: a
signal is a statistically-grounded proposal for a person to evaluate, not
an instruction ("EJECUTAR AHORA" or similar is never generated here) --
approved plan's explicit requirement for notification language.
"""

from __future__ import annotations

from otc_research.db.models import Signal


def format_signal_message(signal: Signal) -> str:
    ref = signal.signal_ref or f"id={signal.id}"
    probability = (
        f"{signal.historical_win_rate:.1%}" if signal.historical_win_rate is not None else "n/a"
    )
    ci = (
        f"[{signal.win_rate_ci_low:.1%}, {signal.win_rate_ci_high:.1%}]"
        if signal.win_rate_ci_low is not None and signal.win_rate_ci_high is not None
        else "n/a"
    )
    margin = (
        f"{signal.margin_over_break_even:+.1%}" if signal.margin_over_break_even is not None else "n/a"
    )
    payout_note = " (estimado)" if signal.payout_is_estimated else ""
    payout = f"{signal.payout:.0%}{payout_note}" if signal.payout is not None else "n/a"

    lines = [
        f"SEÑAL {ref} -- {signal.asset} {signal.direction} ({signal.timeframe}, "
        f"expira en {signal.expiry_seconds}s)",
        f"Confianza: {signal.confidence_label or 'n/a'} "
        f"(probabilidad {probability}, IC95 {ci}, muestra n={signal.historical_sample_size or 'n/a'})",
        f"Payout {payout} -- margen sobre punto de equilibrio: {margin}",
    ]
    if signal.conditions_met:
        lines.append(f"Condiciones cumplidas: {signal.conditions_met}")
    if signal.reasons_rejected:
        lines.append(f"Condiciones NO cumplidas: {signal.reasons_rejected}")
    if signal.valid_until is not None:
        lines.append(f"Ventana de entrada válida hasta: {signal.valid_until.isoformat()}")
    lines.append(
        "Esta es una propuesta estadística para tu revisión, no una instrucción -- "
        "la decisión de operar es siempre tuya."
    )
    return "\n".join(lines)
