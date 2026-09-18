"""Combines the pure indicators in indicators.py into one point-in-time
feature set for a candle series.

``FEATURE_SET_VERSION`` must be bumped (never overwritten in place)
whenever the computation logic here or in indicators.py changes, so that
past Feature rows and any backtest run against them stay reproducible
against the version they were actually computed with — see DATA.md's
``features`` table description and Feature.feature_set_version.
"""

from __future__ import annotations

import pandas as pd

from otc_research.features import indicators

FEATURE_SET_VERSION = "v4"

RETURN_LAGS: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 30)

FEATURE_NAMES: list[str] = [
    "ema_12",
    "ema_26",
    "ema_slope_12_3",
    "rsi_14",
    "atr_14",
    "adx_14",
    "bb_mid_20",
    "bb_upper_20",
    "bb_lower_20",
    "bb_pct_b_20",
    "roc_10",
    *[f"return_{lag}" for lag in RETURN_LAGS],
    "cumulative_return_10",
    "return_acceleration",
    "macd_line",
    "macd_signal_line",
    "macd_histogram",
    "macd_cross_signal",
    "cci_20",
    "rci_9",
    "same_color_streak",
    "range_ratio_20",
    "move_size_atr",
    "rolling_std_return_20",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "body_ratio",
    "engulfing_signal",
    "inside_bar_breakout_signal",
    "donchian_high_20",
    "donchian_low_20",
    "dist_to_high_atr_20",
    "dist_to_low_atr_20",
    "pct_position_in_range_20",
    "atr_expansion_ratio",
    "hour_utc",
    "day_of_week",
    "trading_session_code",
    "structure_bias",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` must have columns ``timestamp``/``open``/``high``/``low``/
    ``close``, one row per candle, sorted ascending by timestamp, integer
    positional index (0..n-1, e.g. via ``reset_index(drop=True)``).

    Returns a DataFrame with the same index and one column per name in
    ``FEATURE_NAMES``. Every value at row ``i`` is computed using only
    rows ``<= i`` — the point-in-time contract described in FEATURES.md —
    which tests/test_feature_engine.py verifies by recomputing on
    truncated input and checking the last row is unchanged. Early rows are
    NaN until enough history exists for a given indicator's lookback;
    pipeline.py never stores those as a fabricated value.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    out = pd.DataFrame(index=df.index)

    out["ema_12"] = indicators.ema(c, 12)
    out["ema_26"] = indicators.ema(c, 26)
    out["ema_slope_12_3"] = indicators.ema_slope(c, span=12, lookback=3)
    out["rsi_14"] = indicators.rsi(c, period=14)
    out["atr_14"] = indicators.atr(h, l, c, period=14)
    out["adx_14"] = indicators.adx(h, l, c, period=14)

    mid, upper, lower, pct_b = indicators.bollinger_bands(c, period=20, num_std=2.0)
    out["bb_mid_20"] = mid
    out["bb_upper_20"] = upper
    out["bb_lower_20"] = lower
    out["bb_pct_b_20"] = pct_b

    out["roc_10"] = indicators.roc(c, period=10)

    for lag in RETURN_LAGS:
        out[f"return_{lag}"] = indicators.roc(c, period=lag)
    out["cumulative_return_10"] = indicators.cumulative_return(c, window=10)
    out["return_acceleration"] = indicators.return_acceleration(c)

    macd_line, macd_signal_line, macd_histogram = indicators.macd(c)
    out["macd_line"] = macd_line
    out["macd_signal_line"] = macd_signal_line
    out["macd_histogram"] = macd_histogram
    out["macd_cross_signal"] = indicators.macd_cross_signal(macd_line, macd_signal_line)

    out["cci_20"] = indicators.cci(h, l, c, period=20)
    out["rci_9"] = indicators.rci(c, period=9)

    out["same_color_streak"] = indicators.same_color_streak(o, c)
    out["range_ratio_20"] = indicators.range_ratio(o, h, l, c, period=20)
    out["move_size_atr"] = indicators.move_size_atr(c, h, l, period=14)
    out["rolling_std_return_20"] = indicators.rolling_std_return(c, period=20)

    upper_wick_ratio, lower_wick_ratio, body_ratio = indicators.wick_ratios(o, h, l, c)
    out["upper_wick_ratio"] = upper_wick_ratio
    out["lower_wick_ratio"] = lower_wick_ratio
    out["body_ratio"] = body_ratio

    out["engulfing_signal"] = indicators.engulfing_signal(o, c)
    out["inside_bar_breakout_signal"] = indicators.inside_bar_breakout_signal(h, l, c)

    out["donchian_high_20"] = indicators.donchian_high(h, period=20)
    out["donchian_low_20"] = indicators.donchian_low(l, period=20)

    atr_14 = out["atr_14"]
    out["dist_to_high_atr_20"] = indicators.dist_to_high_atr(c, h, l, atr_14, period=20)
    out["dist_to_low_atr_20"] = indicators.dist_to_low_atr(c, h, l, atr_14, period=20)
    out["pct_position_in_range_20"] = indicators.pct_position_in_range(c, h, l, period=20)

    out["atr_expansion_ratio"] = indicators.atr_expansion_ratio(h, l, c)

    hour = indicators.hour_of_day_utc(df["timestamp"])
    out["hour_utc"] = hour
    out["day_of_week"] = indicators.day_of_week(df["timestamp"])
    out["trading_session_code"] = indicators.trading_session_code(hour)

    out["structure_bias"] = indicators.structure_bias(h, l)

    assert list(out.columns) == FEATURE_NAMES  # keeps this list and the code in lockstep
    return out
