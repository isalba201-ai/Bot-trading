"""The three execution-realism scenarios BACKTESTING.md requires every
backtest to run under. A strategy that only survives ``OPTIMISTIC`` is
classified fragile, not profitable — see BACKTESTING.md's "Execution
realism" section.
"""

from __future__ import annotations

from dataclasses import dataclass

from otc_research.config import ExecutionScenarioConfig


@dataclass(frozen=True)
class ExecutionScenario:
    name: str  # "optimistic" / "realistic" / "pessimistic"

    #: Candles of extra delay between the decision candle's close and the
    #: actual entry (modeling latency): 0 = enter at the very next candle's
    #: open, exactly on signal.
    entry_delay_candles: int

    #: Fraction of signals discarded outright (missed candle, execution
    #: error, order rejected) rather than ever becoming a trade.
    signal_drop_probability: float

    #: Adverse percent adjustment applied to the entry price, in the
    #: direction that hurts the trade (worse fill than the quoted price).
    slippage_pct: float

    def __post_init__(self) -> None:
        if self.entry_delay_candles < 0:
            raise ValueError("entry_delay_candles must be >= 0")
        if not 0.0 <= self.signal_drop_probability <= 1.0:
            raise ValueError("signal_drop_probability must be between 0 and 1")
        if self.slippage_pct < 0.0:
            raise ValueError("slippage_pct must be >= 0")


#: No latency, no slippage, every signal fills. Not configurable on
#: purpose — it is the fixed zero baseline every other scenario is
#: compared against (see BACKTESTING.md).
OPTIMISTIC = ExecutionScenario(
    name="optimistic", entry_delay_candles=0, signal_drop_probability=0.0, slippage_pct=0.0
)


def realistic_scenario(cfg: ExecutionScenarioConfig) -> ExecutionScenario:
    return ExecutionScenario(
        name="realistic",
        entry_delay_candles=cfg.entry_delay_candles,
        signal_drop_probability=cfg.signal_drop_probability,
        slippage_pct=cfg.slippage_pct,
    )


def pessimistic_scenario(cfg: ExecutionScenarioConfig) -> ExecutionScenario:
    return ExecutionScenario(
        name="pessimistic",
        entry_delay_candles=cfg.entry_delay_candles,
        signal_drop_probability=cfg.signal_drop_probability,
        slippage_pct=cfg.slippage_pct,
    )
