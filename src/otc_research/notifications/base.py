"""The interface every notification channel implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from otc_research.db.models import Signal


class NotificationProvider(ABC):
    """A delivery channel for a generated ``Signal``. ``notify`` must
    raise on failure to deliver rather than fail silently --
    ``signals.service.send_notification`` only stamps ``sent_at`` after
    ``notify`` returns without raising, so a swallowed error here would
    make a signal look delivered when it never reached anyone.
    """

    @abstractmethod
    def notify(self, signal: Signal) -> None: ...
