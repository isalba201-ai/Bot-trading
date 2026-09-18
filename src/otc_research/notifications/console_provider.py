"""The first concrete ``NotificationProvider`` (approved plan's "Open
decisions" #1): stdout, no external service, no credentials -- the
provider every other channel (desktop, Telegram, ...) will eventually be
compared against, since it can never fail to "deliver" in a way that
hides a real bug elsewhere in the pipeline.
"""

from __future__ import annotations

from otc_research.db.models import Signal
from otc_research.notifications.base import NotificationProvider
from otc_research.signals.formatting import format_signal_message
from otc_research.utils.logging import get_logger

logger = get_logger(__name__)


class ConsoleNotificationProvider(NotificationProvider):
    def notify(self, signal: Signal) -> None:
        message = format_signal_message(signal)
        print(message)
        logger.info("signal %s delivered via console", signal.signal_ref or signal.id)
