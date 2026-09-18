#!/usr/bin/env python
"""Step 8d (audit+research addendum), Section 7: is there a statistically
robust way to identify periods NOT to trade at all -- as opposed to
which direction to trade? Reuses research.baseline.conditional_by_category
(no new core code) over the EXISTING `regime` column
(research.regimes.classify_regime, already part of every dataset built by
research.dataset.build_dataset) on TRAIN data only, for every asset/
timeframe/horizon Step 7 covered. A regime is flagged as a candidate
"avoid" period when BOTH call_wins_h and put_wins_h show a below-50% win
rate simultaneously with a reasonable sample -- since call+put win rates
sum to ~1.0 at the naive-label level (see PREDICTABILITY_AUDIT.md), both
being below 50% at once can only happen from a heavier VOID/tie fraction
in that regime, which naive win_rate_stat already excludes from n --
i.e. this specifically surfaces regimes with elevated unresolved-outcome
mass, not just "a losing regime for one direction" (which is just the
other direction's edge).
"""

from __future__ import annotations

from otc_research.config import load_config
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.research.baseline import conditional_by_category
from otc_research.research.dataset import DEFAULT_HORIZONS, build_dataset
from otc_research.research.targets import mfe_mae_labels
from otc_research.utils.logging import get_logger

logger = get_logger(__name__)

ASSET_TIMEFRAMES: tuple[tuple[str, str], ...] = (
    ("EUR_USD", "1m"), ("EUR_USD", "5m"), ("EUR_USD", "15m"),
    ("GBP_USD", "5m"), ("USD_JPY", "5m"),
)
MIN_SAMPLE_SIZE = 100


def main() -> None:
    config = load_config()
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    from otc_research.backtest.splits import compute_temporal_split

    for asset, timeframe in ASSET_TIMEFRAMES:
        df = build_dataset(session, asset, timeframe, feature_set_version=FEATURE_SET_VERSION, horizons=DEFAULT_HORIZONS)
        if df.empty:
            continue
        split = compute_temporal_split(len(df), config.backtest.train_fraction, config.backtest.validation_fraction)
        train_df = df.iloc[split.train_slice]

        logger.info("=== %s/%s (regime distribution on TRAIN) ===", asset, timeframe)
        regime_counts = train_df["regime"].value_counts(dropna=True)
        for regime, count in regime_counts.items():
            logger.info("  regime=%-28s n=%d (%.1f%%)", regime, count, 100.0 * count / len(train_df))

        for horizon in DEFAULT_HORIZONS:
            call_stats = conditional_by_category(train_df, "regime", f"call_wins_{horizon}")
            put_stats = conditional_by_category(train_df, "regime", f"put_wins_{horizon}")
            put_by_regime = {s.category: s for s in put_stats}
            for cs in call_stats:
                ps = put_by_regime.get(cs.category)
                if ps is None or cs.stat.n < MIN_SAMPLE_SIZE or ps.stat.n < MIN_SAMPLE_SIZE:
                    continue
                if cs.stat.win_rate is None or ps.stat.win_rate is None:
                    continue
                if cs.stat.win_rate < 0.5 and ps.stat.win_rate < 0.5:
                    logger.info(
                        "  h=%d regime=%-28s BOTH below 50%%: call_wr=%.4f (n=%d) put_wr=%.4f (n=%d)",
                        horizon, cs.category, cs.stat.win_rate, cs.stat.n, ps.stat.win_rate, ps.stat.n,
                    )

        # Direction-independent "is there even enough movement here to
        # bother" read: per-regime average achievable excursion,
        # regardless of which way it goes.
        mfe_mae = mfe_mae_labels(train_df, max_horizon=10)
        train_with_excursion = train_df.copy()
        train_with_excursion["mfe_call"] = mfe_mae["mfe_call"]
        train_with_excursion["mae_call"] = mfe_mae["mae_call"]
        grouped = train_with_excursion.dropna(subset=["regime", "mfe_call", "mae_call"]).groupby("regime")
        logger.info("  -- 10-candle achievable excursion by regime (direction-independent) --")
        for regime_name, group in grouped:
            if len(group) < MIN_SAMPLE_SIZE:
                continue
            mean_mfe = group["mfe_call"].mean()
            mean_abs_mae = group["mae_call"].abs().mean()
            logger.info(
                "  regime=%-28s n=%-5d mean_mfe=%.4f%% mean_|mae|=%.4f%%",
                regime_name, len(group), mean_mfe * 100, mean_abs_mae * 100,
            )


if __name__ == "__main__":
    main()
