"""Pure, point-in-time-safe indicator functions.

Every function here takes one or more ``pandas.Series`` (already sorted
ascending by timestamp, one row per candle) and returns a Series aligned to
the same index. The hard rule enforced by every implementation, per
FEATURES.md / ARCHITECTURE.md's point-in-time contract: the value at
position ``i`` may only depend on input values at positions ``<= i``.
Concretely that means:

* rolling/ewm windows are never centered and never use a negative shift
  that would land in the future relative to the position being computed,
* any intermediate calculation that *does* look at a future position (the
  fractal/pivot detector below) is only ever exposed in the returned
  series after being shifted forward by exactly the lag needed to make it
  knowable at that position, never earlier.

tests/test_feature_engine.py enforces this for the combined feature set by
recomputing on truncated input and asserting the last row doesn't change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Trend / momentum -------------------------------------------------


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average. NaN until ``span`` candles exist."""
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def ema_slope(close: pd.Series, span: int, lookback: int = 3) -> pd.Series:
    """Change in the EMA over ``lookback`` candles (H3, H10)."""
    e = ema(close, span)
    return (e - e.shift(lookback)) / lookback


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change over ``period`` candles, in percent (H3)."""
    prev = close.shift(period)
    return (close - prev) / prev * 100.0


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI (H7). 100 when there have been no losses at all in
    the lookback (not a division by zero), 0 when there have been no gains
    at all.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    return result.where(avg_loss != 0.0, 100.0)


# --- Volatility ---------------------------------------------------------


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style average true range (H8)."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Returns (mid, upper, lower, pct_b) (H4). ``pct_b`` is where close
    sits within the bands: 0 = at the lower band, 1 = at the upper band.
    """
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    band_width = (upper - lower).replace(0.0, np.nan)
    pct_b = (close - lower) / band_width
    return mid, upper, lower, pct_b


def range_ratio(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """Current candle's range vs. the average range of the PRECEDING
    ``period`` candles (H2 extreme-range detection). Excludes the current
    candle from its own baseline on purpose, so a single huge candle can't
    inflate the average it's being compared against.
    """
    rng = high - low
    baseline = rng.shift(1).rolling(period, min_periods=period).mean()
    return rng / baseline.replace(0.0, np.nan)


# --- Candle shape / price action ----------------------------------------


def same_color_streak(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Signed count of consecutive same-color candles ending at this bar
    (H1): +3 = three bullish candles in a row, -2 = two bearish in a row,
    0 = a doji (open == close) resets the streak. Inherently sequential,
    computed with a plain backward loop (still O(n), still only looks at
    candles <= i).
    """
    colors = np.sign((close - open_).to_numpy())
    streak = np.zeros(len(colors), dtype=float)
    running = 0.0
    for i, color in enumerate(colors):
        if color == 0:
            running = 0.0
        elif running != 0.0 and np.sign(running) == color:
            running += color
        else:
            running = color
        streak[i] = running
    return pd.Series(streak, index=close.index)


def wick_ratios(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper_wick_ratio, lower_wick_ratio, body_ratio), each as a
    fraction of the candle's full range (H6 level rejection). The three
    always sum to 1 (upper wick + lower wick + body == high - low exactly).
    """
    upper_edge = pd.concat([open_, close], axis=1).max(axis=1)
    lower_edge = pd.concat([open_, close], axis=1).min(axis=1)
    rng = (high - low).replace(0.0, np.nan)

    upper_wick_ratio = (high - upper_edge) / rng
    lower_wick_ratio = (lower_edge - low) / rng
    body_ratio = (upper_edge - lower_edge) / rng
    return upper_wick_ratio, lower_wick_ratio, body_ratio


def donchian_high(high: pd.Series, period: int = 20) -> pd.Series:
    """Highest high of the PRECEDING ``period`` candles, excluding the
    current one (H5 breakout baseline)."""
    return high.shift(1).rolling(period, min_periods=period).max()


def donchian_low(low: pd.Series, period: int = 20) -> pd.Series:
    """Lowest low of the PRECEDING ``period`` candles, excluding the
    current one (H5 breakout baseline)."""
    return low.shift(1).rolling(period, min_periods=period).min()


# --- Time / session -------------------------------------------------------


def hour_of_day_utc(timestamp: pd.Series) -> pd.Series:
    return timestamp.dt.hour.astype(float)


def day_of_week(timestamp: pd.Series) -> pd.Series:
    """Monday=0 ... Sunday=6 (H9)."""
    return timestamp.dt.dayofweek.astype(float)


# Approximate UTC session hours (H9), not adjusted for DST. Overlap of
# London/New York is called out separately since liquidity/volatility
# there is empirically different from either session alone.
def trading_session_code(hour: pd.Series) -> pd.Series:
    """Numeric session bucket for the given UTC hour: 0=sydney, 1=tokyo,
    2=london, 3=london_ny_overlap, 4=new_york. Kept numeric (not a string
    column) so it stores cleanly in Feature.value like every other
    feature; see FEATURES.md for the label mapping.
    """

    def _bucket(h: float) -> float:
        h = int(h)
        if 12 <= h < 16:
            return 3.0  # london_ny_overlap
        if 0 <= h < 9:
            return 1.0  # tokyo (also covers the 07:00-09:00 tokyo/london overlap)
        if 7 <= h < 16:
            return 2.0  # london
        if 12 <= h < 21:
            return 4.0  # new_york
        return 0.0  # sydney

    return hour.apply(_bucket)


# --- Market structure -----------------------------------------------------


def _confirmed_fractal(series: pd.Series, wing: int, is_high: bool) -> pd.Series:
    """Bill Williams-style fractal: position ``i`` is a pivot if its value
    is strictly the extreme of the ``2*wing + 1``-wide window centered on
    it. That comparison needs ``wing`` future candles, so the boolean is
    computed unshifted first and then shifted forward by ``wing`` places
    before being returned — i.e. the fact "position i was a pivot" only
    ever appears in the output at position ``i + wing``, the first point
    where it is actually knowable. See the no-lookahead test in
    tests/test_feature_engine.py.
    """
    cond = pd.Series(True, index=series.index)
    for offset in range(1, wing + 1):
        shifted_back = series.shift(offset)
        shifted_fwd = series.shift(-offset)
        if is_high:
            cond &= series > shifted_back
            cond &= series > shifted_fwd
        else:
            cond &= series < shifted_back
            cond &= series < shifted_fwd
    return cond.shift(wing).fillna(False).astype(bool)


def structure_bias(high: pd.Series, low: pd.Series, wing: int = 2) -> pd.Series:
    """+1 while the most recent two confirmed fractal pivots form a
    higher-high + higher-low sequence (uptrend structure), -1 for
    lower-high + lower-low (downtrend structure), 0 otherwise/unclear
    (H10). First version — deliberately simple; only kept in a
    hypothesis's final rule set if it measurably helps out-of-sample per
    BACKTESTING.md's ablation approach.
    """
    fractal_high = _confirmed_fractal(high, wing, is_high=True)
    fractal_low = _confirmed_fractal(low, wing, is_high=False)

    h = high.to_numpy()
    l = low.to_numpy()
    fh = fractal_high.to_numpy()
    fl = fractal_low.to_numpy()

    last_high = prev_high = last_low = prev_low = np.nan
    bias = 0.0
    out = np.zeros(len(high), dtype=float)

    for i in range(len(high)):
        if fh[i]:
            confirmed_value = h[i - wing]
            prev_high, last_high = last_high, confirmed_value
        if fl[i]:
            confirmed_value = l[i - wing]
            prev_low, last_low = last_low, confirmed_value

        have_all = not (
            np.isnan(last_high) or np.isnan(prev_high) or np.isnan(last_low) or np.isnan(prev_low)
        )
        if have_all:
            if last_high > prev_high and last_low > prev_low:
                bias = 1.0
            elif last_high < prev_high and last_low < prev_low:
                bias = -1.0
            else:
                bias = 0.0
        out[i] = bias

    return pd.Series(out, index=high.index)
