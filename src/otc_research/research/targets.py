"""Step 8 (audit+research addendum, see
``/root/.claude/plans/sequential-sparking-candle.md``): alternative
look-forward TARGETS for the statistical-discovery pipeline, additive to
(never replacing) ``research.dataset.build_dataset``'s existing
``call_wins_h``/``put_wins_h`` fixed-horizon labels.

The literature review behind this step flagged a specific, well-evidenced
critique of a fixed-horizon "next candle" label (Step 7's naive target):
it ignores time-varying volatility and doesn't reflect what a real trade
would actually do (a real position would be stopped out or hit its
target before an arbitrary fixed horizon). The "triple-barrier method"
(López de Prado and the wider quant-ML literature) fixes this by sizing
the label's threshold to volatility at signal time (ATR) and letting the
label itself be "which threshold got touched first" rather than "where
is price at a fixed clock time later."

**Deliberate simplification, stated once here rather than buried**: these
labels use only CLOSE prices, the same convention every other target in
this codebase already uses (``call_wins_h``/``put_wins_h`` also compare
future closes, never intrabar high/low). A "true" triple-barrier
implementation checks intrabar high/low for a barrier touch, which this
project's ``Candle`` rows would support but
``research.dataset.build_dataset`` does not currently carry into its
output frame. Using close-only is a real precision loss (a barrier could
be touched intrabar without the candle's close ever crossing it) but
keeps this consistent with the rest of the pipeline's existing
close-based semantics and the backtest engine's own fixed-expiry (not
stop/target) execution model — extending to intrabar high/low is flagged
as future work in ``PREDICTABILITY_AUDIT.md``, not built here.

Like ``call_wins_h``/``put_wins_h``, everything in this module looks
INTO THE FUTURE relative to its own row by construction — these are
targets, never features. Never feed a column from this module back into
``feature_cols`` for ``research.baseline``/``research.discovery``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    close_col: str = "close_price",
    atr_col: str = "atr_14",
    upper_mult: float,
    lower_mult: float,
    max_horizon: int,
) -> pd.Series:
    """For each row ``i``, looks forward up to ``max_horizon`` candles for
    the first close that crosses ``close[i] + upper_mult * atr[i]``
    (label ``1.0``) or ``close[i] - lower_mult * atr[i]`` (label
    ``-1.0``); ``NaN`` if neither barrier is crossed before the
    horizon elapses ("timed out" / unresolved) or if there isn't enough
    future history left to know (the same "never fabricate a trailing
    label" convention ``research.dataset.build_dataset`` already
    follows), or if ``atr[i]`` is missing/non-positive (a barrier can't
    be sized from it). ``upper_mult``/``lower_mult``/``max_horizon`` must
    be fixed BEFORE looking at any result -- this module has no opinion
    on what a good value is, and picking one after seeing outcomes would
    reintroduce exactly the data-mining risk ``research.discovery``'s FDR
    correction exists to control.
    """
    if upper_mult <= 0 or lower_mult <= 0:
        raise ValueError("upper_mult and lower_mult must be positive")
    if max_horizon <= 0:
        raise ValueError("max_horizon must be positive")

    close = df[close_col].to_numpy(dtype=float)
    atr = df[atr_col].to_numpy(dtype=float)
    n = len(df)
    labels = np.full(n, np.nan)

    for i in range(n):
        atr_i = atr[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue
        entry = close[i]
        upper = entry + upper_mult * atr_i
        lower = entry - lower_mult * atr_i
        window_end = i + max_horizon
        if window_end >= n:
            continue  # not enough future history to know if it timed out or not

        for j in range(i + 1, window_end + 1):
            c = close[j]
            if c >= upper:
                labels[i] = 1.0
                break
            if c <= lower:
                labels[i] = -1.0
                break
        # else: neither barrier touched in the window -> stays NaN (timed out)

    return pd.Series(labels, index=df.index)


def triple_barrier_to_binary(labels: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Converts ``triple_barrier_labels``' ``{1.0, -1.0, NaN}`` output
    into the same ``{1.0, 0.0, NaN}`` "did CALL/PUT win" convention
    ``research.dataset.build_dataset``'s ``call_wins_h``/``put_wins_h``
    already use, so ``research.discovery.run_discovery`` and
    ``research.baseline`` can be reused against a triple-barrier target
    completely unchanged. Timed-out rows (``NaN``) stay ``NaN`` in both
    -- never counted as a win or a loss for either direction.
    """
    call_wins = labels.map({1.0: 1.0, -1.0: 0.0})
    put_wins = labels.map({1.0: 0.0, -1.0: 1.0})
    return call_wins, put_wins


@dataclass(frozen=True)
class MfeMaeColumns:
    """Column names ``mfe_mae_labels`` adds to its output frame.
    Maximum Favorable/Adverse Excursion for PUT is not stored separately
    -- for a close-price-only path, PUT's excursion is exactly the sign-
    flipped mirror of CALL's (``mfe_put == -mae_call``,
    ``mae_put == -mfe_call``), so storing both would just duplicate the
    same numbers under four names instead of two.
    """

    mfe_call: str = "mfe_call"
    mae_call: str = "mae_call"


COLUMNS = MfeMaeColumns()


def mfe_mae_labels(
    df: pd.DataFrame, *, close_col: str = "close_price", max_horizon: int
) -> pd.DataFrame:
    """For each row ``i``, the Maximum Favorable Excursion (best
    close-to-close % gain a CALL entered at ``close[i]`` would have seen)
    and Maximum Adverse Excursion (worst % loss, i.e. the most negative
    close-to-close return) over the next ``max_horizon`` candles --
    ``NaN`` for trailing rows without enough future history. A richer,
    continuous-valued per-row signal than a single win/loss bit; used by
    the no-trade-filter analysis (rows with a large adverse excursion in
    BOTH directions -- ``mae_call`` very negative AND ``mfe_call`` very
    small -- are exactly what a "don't trade here" filter should flag).
    """
    if max_horizon <= 0:
        raise ValueError("max_horizon must be positive")

    close = df[close_col].to_numpy(dtype=float)
    n = len(df)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)

    for i in range(n):
        window_end = i + max_horizon
        if window_end >= n:
            continue
        entry = close[i]
        future = close[i + 1 : window_end + 1]
        returns = (future - entry) / entry
        mfe[i] = returns.max()
        mae[i] = returns.min()

    return pd.DataFrame({COLUMNS.mfe_call: mfe, COLUMNS.mae_call: mae}, index=df.index)
