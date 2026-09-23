"""Telegram delivery via the Bot API's plain ``sendMessage`` endpoint --
no SDK dependency, just ``requests`` (already a project dependency via
``TwelveDataSource``). Credentials are never hardcoded or stored in
config.yaml (same rule as ``TWELVEDATA_API_KEY``): both
``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` come from the
environment. This is prepared, optional infrastructure -- the user does
not need to configure it to use the console/sound channels, and this
class is never instantiated unless Telegram is explicitly requested (see
``run_live_signal_monitor.py --notify``).
"""

from __future__ import annotations

import os
from typing import Callable

import requests

from otc_research.db.models import Signal
from otc_research.notifications.base import NotificationProvider
from otc_research.signals.formatting import format_signal_message
from otc_research.utils.logging import get_logger

logger = get_logger(__name__)

API_BASE = "https://api.telegram.org"


class TelegramNotificationProvider(NotificationProvider):
    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        message_formatter: Callable[[Signal], str] = format_signal_message,
    ):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        if not self.bot_token or not self.chat_id:
            raise ValueError(
                "Telegram notifications require TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID "
                "(env vars, or pass them explicitly) -- see LIVE_MANUAL_TEST.md for how "
                "to create a bot with @BotFather and find your chat id."
            )
        self._format = message_formatter

    def notify(self, signal: Signal) -> None:
        message = self._format(signal)
        url = f"{API_BASE}/bot{self.bot_token}/sendMessage"
        response = requests.post(url, json={"chat_id": self.chat_id, "text": message}, timeout=15)
        if response.status_code != 200:
            raise RuntimeError(f"Telegram sendMessage failed ({response.status_code}): {response.text}")
        logger.info("signal %s delivered via Telegram", signal.signal_ref or signal.id)
