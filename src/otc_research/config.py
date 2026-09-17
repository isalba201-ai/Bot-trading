"""Configuration loading.

Everything the system needs to run comes from a single YAML file (see
config/config.yaml for the shipped defaults and comments on each value).
No secrets/credentials belong in this file or anywhere in the repo — see
RISK_MANAGEMENT.md and section 22 of the project brief about live
execution and credential handling.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@dataclasses.dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float
    daily_loss_limit: float
    max_consecutive_losses: int
    max_trades_per_day: int
    max_drawdown: float
    cooldown_after_loss_minutes: int


@dataclasses.dataclass(frozen=True)
class SignalsConfig:
    default_min_quality_tier: str
    allow_live_execution: bool


@dataclasses.dataclass(frozen=True)
class DataValidationConfig:
    max_suspicious_move_pct: float


@dataclasses.dataclass(frozen=True)
class ExecutionScenarioConfig:
    entry_delay_candles: int
    signal_drop_probability: float
    slippage_pct: float


@dataclasses.dataclass(frozen=True)
class BacktestConfig:
    train_fraction: float
    validation_fraction: float
    realistic: ExecutionScenarioConfig
    pessimistic: ExecutionScenarioConfig

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("backtest.train_fraction must be between 0 and 1")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("backtest.validation_fraction must be between 0 and 1")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError(
                "backtest.train_fraction + validation_fraction must leave a "
                "nonzero test fraction"
            )


KNOWN_DATA_PROVIDERS = ("twelvedata", "oanda", "csv")


@dataclasses.dataclass(frozen=True)
class MarketConfig:
    data_provider: str  # "twelvedata", "oanda", or "csv"
    oanda_environment: str  # "practice" or "live" — the API token itself is never in config
    pairs: list[str]
    timeframes: list[str]

    def __post_init__(self) -> None:
        if self.data_provider not in KNOWN_DATA_PROVIDERS:
            raise ValueError(
                f"Unknown market.data_provider {self.data_provider!r}; "
                f"expected one of {KNOWN_DATA_PROVIDERS}"
            )


@dataclasses.dataclass(frozen=True)
class AppConfig:
    database_url: str
    risk: RiskConfig
    signals: SignalsConfig
    data_validation: DataValidationConfig
    market: MarketConfig
    backtest: BacktestConfig


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text())

    if raw["signals"].get("allow_live_execution"):
        raise ValueError(
            "allow_live_execution=true is not supported in this codebase yet. "
            "Live/automated execution is intentionally not implemented "
            "(see RISK_MANAGEMENT.md / project phase 22)."
        )

    market_raw = raw["market"]
    backtest_raw = raw["backtest"]
    execution_raw = backtest_raw["execution"]
    return AppConfig(
        database_url=raw["database"]["url"],
        risk=RiskConfig(**raw["risk"]),
        signals=SignalsConfig(**raw["signals"]),
        data_validation=DataValidationConfig(**raw["data_validation"]),
        market=MarketConfig(
            data_provider=market_raw["data_provider"],
            oanda_environment=market_raw["oanda"]["environment"],
            pairs=list(market_raw["pairs"]),
            timeframes=list(market_raw["timeframes"]),
        ),
        backtest=BacktestConfig(
            train_fraction=backtest_raw["train_fraction"],
            validation_fraction=backtest_raw["validation_fraction"],
            realistic=ExecutionScenarioConfig(**execution_raw["realistic"]),
            pessimistic=ExecutionScenarioConfig(**execution_raw["pessimistic"]),
        ),
    )
