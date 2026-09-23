"""Frozen definition of the ONE manual-live candidate for this phase:
ML_1M5M candidate #11 baseline, ``P(CALL) > 0.50``, delay=1, EUR/USD 1m,
300s (5-candle) expiry -- exactly the configuration reported in
``ML1M5M_CANDIDATE11_FORWARD_TEST_REPORT.md``. Nothing here may be
changed while an observation period is running (user's explicit
instruction): not the model, not the threshold, not the delay, not the
expiry, not the features.

Only ever loads the already-fitted model from
``data/forward_test_models/ml1m5m_candidate11_gradient_boosting_frozen.joblib``
via ``joblib.load`` -- this module never calls ``.fit()`` on anything.
The SHA-256 check in ``load_frozen_model_payload`` is not just a test
fixture: it runs every time this module is used, live or in replay, and
refuses to proceed if the artifact on disk doesn't match the exact bytes
frozen for the forward test.
"""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
from pathlib import Path

from otc_research.research.model_strategy import ModelStrategy

CODE = "ML1M5M_11_LIVE"
ASSET = "EUR_USD"
TIMEFRAME = "1m"
EXPIRY_SECONDS = 300  # 5 candles -- unchanged from every prior phase
PROBABILITY_THRESHOLD = 0.50  # baseline #11 only -- no P>0.65, no CCI filter, in this phase
ENTRY_DELAY_CANDLES = 1  # project standard "delay1" -- see backtest.execution.delay_only_scenario
TIMEFRAME_SECONDS = 60

#: Real broker payout at signal time is unknown -- this is the project's
#: documented default, always shown as estimated (Signal.payout_is_estimated).
PAYOUT = 0.85
PAYOUT_IS_ESTIMATED = True

FROZEN_MODEL_PATH = Path("data/forward_test_models/ml1m5m_candidate11_gradient_boosting_frozen.joblib")
EXPECTED_MODEL_SHA256 = "eaa78365572b6e16f1f716d217951e064cb4dd5c8746e58cf9222091ec2296c7"

#: One second after the last candle used by the blind forward test (see
#: ML1M5M_CANDIDATE11_FORWARD_TEST_REPORT.md: forward block ends
#: 2026-09-22 21:02:00) -- this manual-live test only ever evaluates
#: candles strictly after this, so it can never silently re-consume data
#: that phase already reported on, on top of the historical
#: TRAIN/VALIDATION/TEST window (ends 2026-08-04) it in turn never touched.
LIVE_TEST_START_AFTER = dt.datetime(2026, 9, 22, 21, 2, 1)

#: Historical reference ONLY, for display/context in a notification or
#: report -- never used to compute the live P(CALL) itself, which always
#: comes fresh from the frozen model's own predict_proba on the live
#: feature row. This is the most recent, most relevant out-of-sample read
#: for this EXACT configuration (baseline, delay=1).
HISTORICAL_REFERENCE_NOTE = (
    "Forward-test delay1 (blind, 2026-09-19..22): WR=48.08% n=2373, "
    "CI=(46.08%,50.09%) -- below break-even 54.05%. See "
    "ML1M5M_CANDIDATE11_FORWARD_TEST_REPORT.md. This manual live test "
    "exists to see whether that result continues to hold on further new data."
)


class FrozenModelIntegrityError(RuntimeError):
    """Raised when the on-disk model artifact does not match the exact
    bytes frozen for the forward test -- refuses to proceed rather than
    silently evaluate against a possibly-modified/retrained model.
    """


@functools.lru_cache(maxsize=1)
def load_frozen_model_payload() -> dict:
    import joblib

    if not FROZEN_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"frozen model artifact not found at {FROZEN_MODEL_PATH} -- see "
            "ML1M5M_CANDIDATE11_FORWARD_TEST_REPORT.md section 1 for how it was produced."
        )
    digest = hashlib.sha256(FROZEN_MODEL_PATH.read_bytes()).hexdigest()
    if digest != EXPECTED_MODEL_SHA256:
        raise FrozenModelIntegrityError(
            f"frozen model hash mismatch: expected {EXPECTED_MODEL_SHA256}, got {digest} -- "
            "refusing to load a model artifact that does not match the one frozen for the "
            "forward test. Do not replace this file; if it was intentionally regenerated, "
            "that is itself a change to the frozen strategy and must not happen during an "
            "observation period."
        )
    return joblib.load(FROZEN_MODEL_PATH)


def model_version() -> str:
    """Short, stable identifier for what produced a given evaluation --
    the frozen model's own hash, not a mutable version string.
    """
    return EXPECTED_MODEL_SHA256[:16]


def build_strategy() -> ModelStrategy:
    """Wraps the frozen, already-fitted model exactly as the forward test
    did -- same feature columns (read from the frozen payload itself, not
    re-derived), same direction (CALL only), same expiry, same threshold.
    """
    payload = load_frozen_model_payload()
    return ModelStrategy(
        payload["model"],
        payload["feature_cols"],
        "CALL",
        EXPIRY_SECONDS,
        probability_threshold=PROBABILITY_THRESHOLD,
        label="ml1m5m_candidate11_live_baseline",
    )
