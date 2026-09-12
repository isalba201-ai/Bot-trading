import pytest

from otc_research.config import load_config


def test_default_config_loads():
    config = load_config()
    assert config.risk.risk_per_trade == 0.04
    assert config.risk.max_consecutive_losses == 3
    assert config.signals.allow_live_execution is False
    assert config.signals.default_min_quality_tier == "A+"


def test_live_execution_flag_is_refused(tmp_path):
    bad_config = tmp_path / "config.yaml"
    bad_config.write_text(
        """
database:
  url: "sqlite:///:memory:"
risk:
  risk_per_trade: 0.04
  daily_loss_limit: 0.1
  max_consecutive_losses: 3
  max_trades_per_day: 10
  max_drawdown: 0.2
  cooldown_after_loss_minutes: 30
signals:
  default_min_quality_tier: "A+"
  allow_live_execution: true
data_validation:
  max_suspicious_move_pct: 5.0
"""
    )
    with pytest.raises(ValueError):
        load_config(bad_config)
