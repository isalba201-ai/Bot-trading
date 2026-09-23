"""Combines several ``NotificationProvider``s so a signal can go out over
multiple channels at once (console + sound + Telegram, etc.) -- the
user's explicit request for a hard-to-miss alert, "sonido, notificación
de escritorio, consola, Telegram, o una combinación."

Deliberately relaxes ``NotificationProvider``'s "raise on any failure"
contract: it raises only if EVERY sub-provider failed, so an optional,
best-effort channel (desktop notification, Telegram) failing does not
mask a channel that actually got through (console/sound, which never
fail). Every individual failure is still logged, never swallowed
silently.
"""

from __future__ import annotations

from otc_research.db.models import Signal
from otc_research.notifications.base import NotificationProvider
from otc_research.utils.logging import get_logger

logger = get_logger(__name__)


class CompositeNotificationProvider(NotificationProvider):
    def __init__(self, providers: list[NotificationProvider]):
        if not providers:
            raise ValueError("CompositeNotificationProvider needs at least one provider")
        self.providers = providers

    def notify(self, signal: Signal) -> None:
        failures: list[tuple[NotificationProvider, Exception]] = []
        for provider in self.providers:
            try:
                provider.notify(signal)
            except Exception as exc:  # noqa: BLE001 -- deliberately broad, see module docstring
                logger.warning(
                    "notification channel %s failed for signal %s: %s",
                    type(provider).__name__, signal.signal_ref or signal.id, exc,
                )
                failures.append((provider, exc))

        if len(failures) == len(self.providers):
            names = ", ".join(type(p).__name__ for p, _ in failures)
            raise RuntimeError(f"all notification channels failed for this signal: {names}")
