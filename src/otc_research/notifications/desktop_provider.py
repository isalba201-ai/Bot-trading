"""Best-effort OS desktop notification (Linux ``notify-send`` or macOS
``osascript``). Raises if neither is available or the call fails -- per
``NotificationProvider``'s contract, a channel must never claim delivery
it didn't actually make. Use via ``CompositeNotificationProvider`` if you
want a guaranteed-successful channel (console/sound) alongside this one.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Callable

from otc_research.db.models import Signal
from otc_research.notifications.base import NotificationProvider
from otc_research.signals.formatting import format_signal_message
from otc_research.utils.logging import get_logger

logger = get_logger(__name__)


class DesktopNotificationProvider(NotificationProvider):
    def __init__(self, message_formatter: Callable[[Signal], str] = format_signal_message):
        self._format = message_formatter

    def notify(self, signal: Signal) -> None:
        title = f"CALL — {signal.asset} — {signal.timeframe}"
        body = self._format(signal)

        if shutil.which("notify-send"):
            result = subprocess.run(["notify-send", title, body], capture_output=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError(f"notify-send failed: {result.stderr.decode(errors='replace')}")
        elif shutil.which("osascript"):
            script = f'display notification {body!r} with title {title!r}'
            result = subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError(f"osascript failed: {result.stderr.decode(errors='replace')}")
        else:
            raise RuntimeError("no desktop notification backend available (notify-send/osascript not found)")

        logger.info("signal %s delivered via desktop notification", signal.signal_ref or signal.id)
