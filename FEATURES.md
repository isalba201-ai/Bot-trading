# Feature engineering (Phase 3)

**Status: implemented.** `src/otc_research/features/` computes the feature
set below from stored, validated candles and writes it to the `features`
table (see DATA.md). Nothing in Phase 4+ (backtesting, strategies, the
signal engine) is allowed to compute an indicator itself or read candles
directly — it reads `Feature` rows produced here, so every strategy sees
exactly the same, auditable numbers.

## The point-in-time contract

> A feature at `timestamp` may only be computed from candles with
> `timestamp' <= timestamp`. Anything else is look-ahead bias.
> — DATA.md

This is the single most important property of this module, because a
backtest built on a feature that secretly used future data will look
profitable and will not be. It is enforced two ways:

- Every indicator in `indicators.py` is implemented with non-centered
  rolling windows / `ewm` / forward-only shifts. The one indicator that
  legitimately needs to look at future bars internally (`structure_bias`'s
  fractal/pivot detection, which by definition can't confirm a pivot until
  bars exist on both sides of it) only ever exposes that fact in its
  output `wing` bars later than the pivot itself — the first point at
  which it is actually knowable.
- `tests/test_feature_engine.py::test_features_are_point_in_time_safe`
  checks this directly rather than trusting the implementation: it
  recomputes the whole feature set on a truncated prefix of the same
  candle series and asserts the last row is bit-for-bit identical to what
  the full computation produced for that same timestamp. If any indicator
  ever regresses into using a future candle, this test fails.

`pipeline.py` also never stores a fabricated value: if an indicator's
lookback isn't satisfied yet (not enough history), that (asset,
timeframe, timestamp, name) row is simply not written — never a 0 or a
carried-forward previous value standing in for "unknown".

## Feature set (`feature_set_version = "v1"`)

| Feature | Formula / definition | Supports hypothesis |
|---|---|---|
| `ema_12`, `ema_26` | Exponential moving average, span 12 / 26 | H3, H10 |
| `ema_slope_12_3` | `(EMA12[t] - EMA12[t-3]) / 3` | H3, H10 |
| `rsi_14` | Wilder RSI, period 14 (100 when there were no losses at all in the lookback, 0 when there were no gains at all — not a division-by-zero artifact) | H7 |
| `atr_14` | Wilder average true range, period 14 | H8 |
| `bb_mid_20`, `bb_upper_20`, `bb_lower_20` | 20-period SMA ± 2 population std dev | H4 |
| `bb_pct_b_20` | `(close - lower) / (upper - lower)`; 0 = at the lower band, 1 = at the upper band | H4 |
| `roc_10` | `(close[t] - close[t-10]) / close[t-10] * 100` | H3 |
| `same_color_streak` | Signed count of consecutive same-color candles ending at `t` (e.g. `+3` = three bullish in a row); a doji (open == close) resets it to 0 | H1 |
| `range_ratio_20` | `(high[t]-low[t]) / mean(range[t-20..t-1])` — current range vs. the average of the **preceding** 20 candles, so one huge candle can't inflate its own baseline | H2 |
| `upper_wick_ratio`, `lower_wick_ratio`, `body_ratio` | Wick/body sizes as a fraction of the candle's full range; the three always sum to exactly 1 | H6 |
| `donchian_high_20`, `donchian_low_20` | Highest high / lowest low of the **preceding** 20 candles (current candle excluded from its own baseline) | H5 |
| `hour_utc` | UTC hour of the candle's open time (0-23) | H9 |
| `day_of_week` | Monday=0 ... Sunday=6 | H9 |
| `trading_session_code` | Approximate UTC session bucket: `0`=Sydney, `1`=Tokyo (also covers the Tokyo/London overlap), `2`=London, `3`=London/New York overlap, `4`=New York. Not adjusted for DST — a first version, see below | H9 |
| `structure_bias` | `+1` while the two most recent **confirmed** fractal pivots form a higher-high + higher-low sequence, `-1` for lower-high + lower-low, `0` otherwise. Pivot confirmation uses a 2-bar wing on each side (Bill Williams-style fractal) | H10 |

Every column is stored under `Feature.name`, one row per
`(asset, timeframe, timestamp, feature_set_version, name)` — see DATA.md's
schema description.

## Known simplifications in this first version

Per STRATEGIES.md's rule that supporting factors are only kept if they
measurably improve out-of-sample results, not because they're common
technical-analysis concepts, the following are intentionally simple and
are candidates for revision once Phase 4/5 backtesting can actually
measure whether a more elaborate version helps:

- `trading_session_code`'s hour boundaries are fixed UTC ranges, not
  adjusted for daylight saving time in any session's local timezone.
- `structure_bias` uses a single fixed fractal wing size (2) and only
  looks at the two most recent confirmed pivots — it does not yet model
  longer swing sequences, trendlines, or invalidation on a break of
  structure.

Changing either of the above (or any other indicator's formula) requires
bumping `FEATURE_SET_VERSION` in `features/engine.py`, never editing the
existing version's meaning in place — old `Feature` rows and any backtest
run against them must stay reproducible against the version they were
actually computed with.

## Running it

```bash
# after fetching/importing candles for a pair+timeframe:
python scripts/compute_features.py --pair EUR_USD --timeframe 5m
```

Safe to re-run: already-computed rows for the current
`feature_set_version` are skipped, never overwritten.

## What's next (Phase 4+)

`Feature` rows are the only input the backtesting engine (Phase 4) and
every hypothesis's evaluation logic are allowed to read — see
BACKTESTING.md and ARCHITECTURE.md's layering rule. Nothing past this
point exists yet.
