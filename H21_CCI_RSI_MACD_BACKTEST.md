# H21 backtest: CCI(20) + RSI(14) overbought, confirmed by a bearish MACD histogram + red candle

**Status: complete. Bottom line first, since it's the most important
number in this document: across every real dataset this project has
ingested (~40,000 candles total, spanning 3 days to 7 months of history
depending on timeframe), this exact signal fired 5 times.** Not 5% of
the time — five occurrences, total, everywhere. That is the honest answer
to "cuántas oportunidades se dieron" before any question of win rate even
applies. The rest of this document explains precisely why, with the exact
numbers, and what a data-grounded next step would look like.

## 1. Exact strategy tested (no ambiguity, per this project's own discipline)

Implemented literally as described, at each indicator's own conventional
default (`src/otc_research/strategies/h21_cci_rsi_macd_reversal.py`):

```
IF  cci_20  >= 100   (CCI(20) overbought — same default as H12)
AND rsi_14  >= 70    (RSI(14) overbought — same default as H7)
AND macd_histogram < 0   (MACD(12,26,9) histogram bar below zero — "red"
                           on any platform that colors histogram bars by
                           sign; this is the concrete meaning given to
                           "el MACD presenta una vela roja ... en el
                           volumen del MAC" for this test)
AND close < open      (the signal candle itself is bearish — "esa vela roja")
THEN → PUT
```

Entry: next candle's open (1-candle delay — this project's established
default reaction-time assumption, `delay_only_scenario(1)`, used for
every other strategy in this report family). Expiry: **both requested
values, h=2 and h=4 candles** (`expiry_seconds = h × timeframe_seconds`,
e.g. 10/20 min on 5m, 30/60 min on 15m). **PUT-only** — the user described
only the overbought/bearish case; no oversold/bullish mirror was
requested or added.

## 2. Signal frequency — the actual finding

| Activo/TF | Filas (historia completa) | CCI≥100 | RSI≥70 | CCI≥100 AND RSI≥70 | + MACD hist<0 | + vela roja (señal completa) |
|---|---:|---:|---:|---:|---:|---:|
| EUR_USD/1m | 4,966 | — | — | — | — | **0** |
| EUR_USD/5m | 4,967 | 886 (17.8%) | 64 (1.3%) | 64 | 0 | **0** |
| EUR_USD/15m | 4,967 | 942 (19.0%) | 126 (2.5%) | 111 | 1 | **1** |
| EUR_USD/1h | 4,964 | 911 (18.4%) | 176 (3.5%) | 147 | 4 | **1** |
| GBP_USD/5m | 4,967 | 914 (18.4%) | 108 (2.2%) | 93 | 0 | **0** |
| GBP_USD/1h | 4,967 | — | — | — | — | **0** |
| USD_JPY/5m | 4,967 | 964 (19.4%) | 179 (3.6%) | 164 | 11 | **2** |
| USD_JPY/1h | 4,967 | — | — | — | — | **1** |
| **TOTAL** | ~39,700 | | | | | **5** |

**Why it collapses to almost nothing, mechanically (from the data, not a
guess)**: CCI≥100 alone is common (~18-19% of candles — a loose filter).
RSI≥70 is already much stricter (1.3-3.6%) — when RSI is overbought, CCI
almost always is too (the "CCI AND RSI" column barely differs from "RSI
alone"), so those two are not independent, mechanistically-distinct
filters here, they're nearly the same filter twice. The real bottleneck
is **requiring the MACD histogram to already be negative AT THE SAME
CANDLE** as RSI/CCI are still at their overbought extreme: by
construction, RSI/CCI tend to peak DURING the fastest part of an
up-move, which is also when the MACD histogram is typically still
*rising* (positive) — the histogram only turns negative once the MACD
line crosses below its signal line, which normally happens several
candles *after* RSI/CCI have already dropped out of overbought
territory. Demanding all three simultaneously asks for a narrow,
late-stage coincidence, not a common pattern — and the data confirms it:
adding the MACD condition took 64-179 occurrences down to 0-11 on every
pair, before the red-candle filter even applies.

## 3. Raw tally per pair/expiry (TRAIN+VALIDATION, 80% of history —
TEST intentionally excluded so it stays untouched)

| Activo/TF | h | Oportunidades resueltas | Ganadas | Perdidas | Win % |
|---|---:|---:|---:|---:|---:|
| EUR_USD/15m | 2 | 1 | 1 | 0 | 100% |
| EUR_USD/15m | 4 | 1 | 1 | 0 | 100% |
| Todos los demás (14 combinaciones: 7 pares/TF × 2 vencimientos) | — | **0** | 0 | 0 | n/a |

This is the literal answer requested — "de esas 100 oportunidades, cuántas
gané/perdí" — except there were never 100 opportunities on any pair. The
largest count anywhere, on any pair or expiry, was **1**. A single win on
a single occurrence is not evidence of anything — it is one coin flip
that happened to land heads, full stop. It cannot be evaluated for
reliability, and this report does not claim it shows an edge.

## 4. Gated evaluation (the same 4-gate funnel every strategy in this
project goes through)

Every one of the 16 (pair, expiry) combinations was rejected at **gate 1
(sample size)** — `"TRAIN sample size 0 < required 100"` for 14 of them,
and for the 2 EUR_USD/15m cases, `n=1` is likewise far short of 100.
**None reached the robustness, walk-forward, or TEST gates at all** —
there isn't enough signal to even begin testing consistency, let alone
statistical significance.

## 5. Is this strategy reliable? — direct answer

**No, and it cannot currently be judged either way** — not "rejected for
lacking an edge" (that would require enough trades to measure an edge at
all), but "there is not enough data to have an opinion." A signal that
fires 5 times across 40,000 real candles is not usable as a trading
strategy regardless of what its win rate happens to be on those 5 cases.

## 6. Data-grounded ideas for improvement — not tested, explicitly flagged as such

Per your own instruction not to invent results: nothing below has been
run. These are the specific next steps the frequency breakdown in §2
points to, if you want to test them:

1. **Loosen the RSI threshold** (e.g. 70 → 65 or 60) — RSI≥70 is the
   tightest single filter (1.3-3.6%); a lower bar would materially
   increase the CCI+RSI joint count (though not necessarily fix #2 below).
2. **Stop requiring MACD histogram<0 at the exact same candle as the
   RSI/CCI overbought reading.** The mechanical mismatch in §2 suggests
   the natural fix is a short lookback window instead: "RSI/CCI were
   overbought within the last N candles, AND the MACD histogram has
   *since* turned negative" — the standard "confirmation follows the
   extreme by a few candles" pattern in discretionary trading, and a much
   better match to how these three indicators actually move relative to
   each other in time.
3. **Alternative to histogram<0**: use `macd_cross_signal` (the discrete
   bearish-cross event H11 already tracks) within a few candles *after*
   the RSI/CCI overbought reading, instead of the instantaneous
   histogram-sign condition — same idea as #2, different mechanism.

Any of these would be a **new, distinct hypothesis** (H21 as literally
specified stays exactly as tested above, unmodified) needing its own
pre-registration and fresh TRAIN/VALIDATION/TEST protocol, per this
project's standing discipline — happy to build and run whichever variant
you want to pursue next.
