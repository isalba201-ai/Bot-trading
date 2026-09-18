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


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram) (H11). Self-contained
    (computes its own fast/slow EMAs rather than reusing ema_12/ema_26) so
    changing MACD's own periods never silently changes those other
    features.
    """
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def macd_cross_signal(macd_line: pd.Series, signal_line: pd.Series) -> pd.Series:
    """+1 on the bar where the MACD line crosses from at/below to above
    its signal line, -1 on a cross from at/above to below, 0 otherwise
    (H11) — a discrete one-bar trigger, not a sustained state. Sequential
    (like same_color_streak/structure_bias), still causal: bar i's value
    only compares bar i to bar i-1.
    """
    diff = (macd_line - signal_line).to_numpy()
    out = np.zeros(len(diff))
    prev_sign = 0.0
    for i, d in enumerate(diff):
        if np.isnan(d):
            prev_sign = 0.0
            continue
        sign = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
        if prev_sign != 0.0 and sign != 0.0 and sign != prev_sign:
            out[i] = sign
        if sign != 0.0:
            prev_sign = sign
    return pd.Series(out, index=macd_line.index)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index (H12): how far the typical price sits from
    its own recent moving average, scaled by mean absolute deviation.
    Conventionally ranges roughly -100..+100 but is not hard-bounded.
    """
    typical_price = (high + low + close) / 3.0
    sma_tp = typical_price.rolling(period, min_periods=period).mean()
    mean_deviation = typical_price.rolling(period, min_periods=period).apply(
        lambda window: np.mean(np.abs(window - window.mean())), raw=True
    )
    return (typical_price - sma_tp) / (0.015 * mean_deviation.replace(0.0, np.nan))


def rci(close: pd.Series, period: int = 9) -> pd.Series:
    """Rank Correlation Index (H13): Spearman rank correlation between
    chronological order and price rank over the trailing ``period``
    candles, scaled to [-100, 100]. +100 = price rose on every candle in
    the window (in rank terms), -100 = fell on every candle. A popular
    short-term momentum/exhaustion indicator in Japanese retail trading,
    conceptually different from RSI/momentum (rank-based, not
    magnitude-based).

    Ties are broken by original (chronological) order rather than
    averaged — a deliberate simplification that's immaterial for
    continuous FX close prices, where an exact tie within one window is
    vanishingly rare.
    """

    def _rci_window(window: np.ndarray) -> float:
        n = len(window)
        time_rank = np.arange(1, n + 1, dtype=float)
        price_rank = np.empty(n, dtype=float)
        sorter = np.argsort(window, kind="mergesort")
        price_rank[sorter] = np.arange(1, n + 1, dtype=float)
        d_squared_sum = np.sum((time_rank - price_rank) ** 2)
        return (1.0 - 6.0 * d_squared_sum / (n * (n**2 - 1))) * 100.0

    return close.rolling(period, min_periods=period).apply(_rci_window, raw=True)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Average Directional Index: trend STRENGTH, direction-agnostic
    (a strong downtrend and a strong uptrend both score high). Used by
    research/regimes.py to classify trend-strength regime; ``ema_slope``
    already carries direction, so the two are meant to be used together,
    not as substitutes for each other.
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )
    tr = true_range(high, low, close)

    smoothed_tr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * smoothed_plus_dm / smoothed_tr.replace(0.0, np.nan)
    minus_di = 100.0 * smoothed_minus_dm / smoothed_tr.replace(0.0, np.nan)
    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum

    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


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


def atr_expansion_ratio(
    high: pd.Series, low: pd.Series, close: pd.Series, atr_period: int = 14, baseline_period: int = 20
) -> pd.Series:
    """Current ATR vs. the average ATR of the PRECEDING ``baseline_period``
    candles (H8 volatility-regime-change detection): a ratio well above 1
    means volatility has expanded relative to its own recent (calmer)
    baseline. Same "exclude the current value from its own baseline"
    principle as range_ratio/donchian, applied to ATR instead of raw
    range.
    """
    atr_series = atr(high, low, close, period=atr_period)
    baseline = atr_series.shift(1).rolling(baseline_period, min_periods=baseline_period).mean()
    return atr_series / baseline.replace(0.0, np.nan)


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


def rolling_std_return(close: pd.Series, period: int = 20) -> pd.Series:
    """Rolling (population) standard deviation of 1-candle percent returns
    over the trailing ``period`` candles — a volatility measure independent
    of ATR's true-range definition, used by research/regimes.py alongside
    ``atr_expansion_ratio`` for volatility-regime classification.
    """
    one_candle_return = close.pct_change() * 100.0
    return one_candle_return.rolling(period, min_periods=period).std(ddof=0)


def move_size_atr(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    """Current candle's close-to-close move, scaled by ATR: how large this
    move was relative to what's typical right now. Uses its own ATR call
    (period matches ``atr_14`` by default) rather than taking a precomputed
    series, for the same self-containment reason as ``macd``.
    """
    atr_series = atr(high, low, close, period=period)
    return (close - close.shift(1)).abs() / atr_series.replace(0.0, np.nan)


# --- Returns / distance to extremes ---------------------------------------


def cumulative_return(close: pd.Series, window: int = 10) -> pd.Series:
    """Sum of the trailing ``window`` individual 1-candle percent returns —
    approximately, but not exactly, equal to a single ``window``-candle
    return (``roc(close, window)``) due to compounding; this is the
    candle-by-candle accumulated version specifically.
    """
    one_candle_return = close.pct_change() * 100.0
    return one_candle_return.rolling(window, min_periods=window).sum()


def return_acceleration(close: pd.Series) -> pd.Series:
    """Change in the 1-candle return itself: is the move speeding up or
    slowing down, one candle at a time.
    """
    one_candle_return = roc(close, period=1)
    return one_candle_return - one_candle_return.shift(1)


def dist_to_high_atr(
    close: pd.Series, high: pd.Series, low: pd.Series, atr_close: pd.Series, period: int = 20
) -> pd.Series:
    """Distance from close to the highest high of the PRECEDING ``period``
    candles (current candle excluded, same convention as ``donchian_high``),
    scaled by ATR so it's comparable across volatility regimes.
    """
    recent_high = donchian_high(high, period=period)
    return (recent_high - close) / atr_close.replace(0.0, np.nan)


def dist_to_low_atr(
    close: pd.Series, high: pd.Series, low: pd.Series, atr_close: pd.Series, period: int = 20
) -> pd.Series:
    """Distance from close to the lowest low of the PRECEDING ``period``
    candles, scaled by ATR — see ``dist_to_high_atr``.
    """
    recent_low = donchian_low(low, period=period)
    return (close - recent_low) / atr_close.replace(0.0, np.nan)


def pct_position_in_range(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """Where close sits within the PRECEDING ``period`` candles' high-low
    range: 0 = at the recent low, 1 = at the recent high. Same
    current-candle-excluded baseline as ``donchian_high``/``donchian_low``.
    """
    recent_high = donchian_high(high, period=period)
    recent_low = donchian_low(low, period=period)
    span = (recent_high - recent_low).replace(0.0, np.nan)
    return (close - recent_low) / span


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


def engulfing_signal(open_: pd.Series, close: pd.Series) -> pd.Series:
    """+1 on a bullish engulfing bar (previous candle bearish, current
    bullish, current body fully contains the previous body), -1 on a
    bearish engulfing bar, 0 otherwise (H14). Classic two-candle price
    action reversal pattern; causal, only needs bar i and i-1.
    """
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    prev_bearish = prev_close < prev_open
    prev_bullish = prev_close > prev_open
    curr_bullish = close > open_
    curr_bearish = close < open_

    bullish_engulf = prev_bearish & curr_bullish & (open_ <= prev_close) & (close >= prev_open)
    bearish_engulf = prev_bullish & curr_bearish & (open_ >= prev_close) & (close <= prev_open)

    out = pd.Series(0.0, index=open_.index)
    out[bullish_engulf.fillna(False)] = 1.0
    out[bearish_engulf.fillna(False)] = -1.0
    return out


def inside_bar_breakout_signal(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Tracks the most recent "mother bar" — the candle immediately
    before an inside bar (a candle whose full range sits inside the
    previous one, a classic price-action consolidation signal). +1 on
    the bar where close first breaks above the mother bar's high, -1 on
    a break below its low, 0 while still consolidating inside or once a
    breakout has already fired/been superseded by a newer inside bar
    (H15). Sequential state (like structure_bias), still causal: bar i's
    value only depends on bars <= i.
    """
    h = high.to_numpy()
    l = low.to_numpy()
    c = close.to_numpy()
    n = len(h)
    out = np.zeros(n)
    mother_high = np.nan
    mother_low = np.nan
    active = False

    for i in range(1, n):
        is_inside = h[i] <= h[i - 1] and l[i] >= l[i - 1]
        if is_inside:
            mother_high = h[i - 1]
            mother_low = l[i - 1]
            active = True
            continue
        if active:
            if c[i] > mother_high:
                out[i] = 1.0
                active = False
            elif c[i] < mother_low:
                out[i] = -1.0
                active = False

    return pd.Series(out, index=high.index)


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
