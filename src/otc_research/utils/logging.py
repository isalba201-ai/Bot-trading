"""Minimal shared logging setup.

Everything logs to both stderr and a rotating-by-run log file under
``logs/`` so signal generation / ingestion runs can be reconstructed later
(see AUDITORÍA / section 28 of the project brief).
"""

from __future__ import annotations

import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[3] / "logs"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_DIR / "otc_research.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
