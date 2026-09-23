"""A channel that is hard to miss even without any external service:
prints the message plus the ASCII bell character (most terminals beep or
flash on it) -- no configuration, no external dependency, always
succeeds like ``ConsoleNotificationProvider``.
"""

from __future__ import annotations

import sys
from typing import Callable

from otc_research.db.models import Signal
from otc_research.notifications.base import NotificationProvider
from otc_research.signals.formatting import format_signal_message
from otc_research.utils.logging import get_logger

logger = get_logger(__name__)

BELL = "\a"


class SoundNotificationProvider(NotificationProvider):
    def __init__(self, message_formatter: Callable[[Signal], str] = format_signal_message):
        self._format = message_formatter

    def notify(self, signal: Signal) -> None:
        message = self._format(signal)
        sys.stdout.write(BELL * 3 + "\n" + message + "\n")
        sys.stdout.flush()
        logger.info("signal %s delivered via sound/console bell", signal.signal_ref or signal.id)
