"""The three execution-realism scenarios BACKTESTING.md requires every
backtest to run under. A strategy that only survives ``OPTIMISTIC`` is
classified fragile, not profitable — see BACKTESTING.md's "Execution
realism" section.

``realistic_scenario``/``pessimistic_scenario`` bundle ``slippage_pct``
in with delay — appropriate for a hand-designed strategy meant to be
manually placed as a spot-FX trade, where crossing the bid/ask spread is
a real cost. ``delay_only_scenario``/``slippage_only_scenario`` below
exist to isolate that bundle into its parts — see
``BINARY_OPTIONS_REFRAME_AUDIT.md``: a binary option has no fill-price/
spread concept, so ``research.candidacy`` now evaluates a discovered
condition under ``delay_only_scenario`` (no slippage, no signal-drop) by
default, never the bundled ``realistic_scenario``. ``slippage_only_scenario``
is kept for diagnostic/forensic use (``research.decomposition``), not as
a default for any binary-options-relevant evaluation.
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


def delay_only_scenario(entry_delay_candles: int) -> ExecutionScenario:
    """No slippage, no signal-drop — isolates pure entry-timing delay
    (how much does waiting ``entry_delay_candles`` extra candles before
    entering cost you), the one friction dimension a binary option
    actually has (see this module's docstring).
    """
    return ExecutionScenario(
        name=f"delay_only_{entry_delay_candles}",
        entry_delay_candles=entry_delay_candles,
        signal_drop_probability=0.0,
        slippage_pct=0.0,
    )


def slippage_only_scenario(slippage_pct: float) -> ExecutionScenario:
    """Zero delay, just the adverse fill-price adjustment — a spot-FX
    bid/ask-crossing cost model. Diagnostic only: kept for
    ``research.decomposition``'s forensic ladder, not used as a default
    for any binary-options-relevant evaluation (a binary option has no
    fill price to slip).
    """
    return ExecutionScenario(
        name="slippage_only", entry_delay_candles=0, signal_drop_probability=0.0, slippage_pct=slippage_pct
    )
