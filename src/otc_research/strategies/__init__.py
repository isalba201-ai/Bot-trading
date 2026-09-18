"""Phase 5: concrete strategies for the hypotheses registered in
STRATEGIES.md, implementing otc_research.backtest.strategy.Strategy so
each is directly runnable by the Phase 4 engine (backtest.engine.run_backtest).

**Implementing a hypothesis here is not the same as it having an edge.**
Every class below is a first, literal reading of its hypothesis's entry
trigger from STRATEGIES.md — nothing has been run against real market
data yet, so every one of them is still exactly what STRATEGIES.md calls
"registered", not "tested". Actually testing one means fetching real
candles (scripts/fetch_market_data.py), computing features
(scripts/compute_features.py), and running it through the engine
(scripts/run_baseline_backtests.py) — see that script for how the
resulting Hypothesis.status transition is recorded, and BACKTESTING.md
for why that's still not a final edge classification (NO EDGE / WEAK
EDGE / PROMISING / ROBUST EDGE) until walk-forward, Monte Carlo, and
robustness testing (Phases 6-8) also exist.

Every constructor parameter with a default (thresholds, lookback-adjacent
knobs) is intentionally a parameter, not a hardcoded constant, because
Phase 6's robustness/sensitivity sweeps need to vary them — a hypothesis
that only "works" at one exact threshold and breaks under small
perturbations is OVERFITTED/FRAGILE per BACKTESTING.md, and that can only
be checked if the threshold is actually a knob.
"""

from otc_research.strategies.h1_streak import H1StreakContinuation
from otc_research.strategies.h2_extreme_range import H2ExtremeRangeReversion
from otc_research.strategies.h3_momentum import H3MomentumContinuation
from otc_research.strategies.h4_bollinger import H4BollingerMeanReversion
from otc_research.strategies.h5_breakout import H5DonchianBreakout
from otc_research.strategies.h6_wick_rejection import H6WickRejection
from otc_research.strategies.h7_rsi_extreme import H7RsiExtremeConfirmed
from otc_research.strategies.h8_volatility_expansion import H8VolatilityExpansion
from otc_research.strategies.h9_session_bias import H9SessionBias
from otc_research.strategies.h10_combined import H10CombinedTrendStructureMomentum
from otc_research.strategies.h11_macd_cross import H11MacdCrossContinuation
from otc_research.strategies.h12_cci_extreme import H12CciExtremeReversion
from otc_research.strategies.h13_rci_extreme import H13RciExtremeReversion
from otc_research.strategies.h14_engulfing import H14EngulfingReversal
from otc_research.strategies.h15_inside_bar_breakout import H15InsideBarBreakout
from otc_research.strategies.h16_macd_rsi_confirmed import H16MacdRsiConfirmed
from otc_research.strategies.h17_bollinger_rci_confirmed import H17BollingerRciConfirmed
from otc_research.strategies.h18_cci_engulfing_confirmed import H18CciEngulfingConfirmed
from otc_research.strategies.h19_trend_pullback import H19TrendPullback
from otc_research.strategies.h20_breakout_volatility_confirmed import (
    H20BreakoutVolatilityConfirmed,
)
from otc_research.strategies.h21_cci_rsi_macd_reversal import H21CciRsiMacdBearishReversal

#: Hypothesis code -> strategy class. H9 is deliberately excluded: it
#: needs an (hour, direction) pair to test one specific bias hypothesis
#: at a time (see its module docstring for why), so it can't be
#: zero-argument constructed like the others.
BASELINE_STRATEGIES = {
    "H1": H1StreakContinuation,
    "H2": H2ExtremeRangeReversion,
    "H3": H3MomentumContinuation,
    "H4": H4BollingerMeanReversion,
    "H5": H5DonchianBreakout,
    "H6": H6WickRejection,
    "H7": H7RsiExtremeConfirmed,
    "H8": H8VolatilityExpansion,
    "H10": H10CombinedTrendStructureMomentum,
    "H11": H11MacdCrossContinuation,
    "H12": H12CciExtremeReversion,
    "H13": H13RciExtremeReversion,
    "H14": H14EngulfingReversal,
    "H15": H15InsideBarBreakout,
    "H16": H16MacdRsiConfirmed,
    "H17": H17BollingerRciConfirmed,
    "H18": H18CciEngulfingConfirmed,
    "H19": H19TrendPullback,
    "H20": H20BreakoutVolatilityConfirmed,
    "H21": H21CciRsiMacdBearishReversal,
}

__all__ = [
    "BASELINE_STRATEGIES",
    "H1StreakContinuation",
    "H2ExtremeRangeReversion",
    "H3MomentumContinuation",
    "H4BollingerMeanReversion",
    "H5DonchianBreakout",
    "H6WickRejection",
    "H7RsiExtremeConfirmed",
    "H8VolatilityExpansion",
    "H9SessionBias",
    "H10CombinedTrendStructureMomentum",
    "H11MacdCrossContinuation",
    "H12CciExtremeReversion",
    "H13RciExtremeReversion",
    "H14EngulfingReversal",
    "H15InsideBarBreakout",
    "H16MacdRsiConfirmed",
    "H17BollingerRciConfirmed",
    "H18CciEngulfingConfirmed",
    "H19TrendPullback",
    "H20BreakoutVolatilityConfirmed",
    "H21CciRsiMacdBearishReversal",
]
