"""Step 6 (approved plan points 4 and 8): simple ML, gated behind
``research.discovery`` finding an FDR-significant condition — this module
is never a first-resort signal generator, only escalated to after steps
1-7 of the statistical methodology have already found something worth
modeling more richly. Three model families, in the plan's fixed order —
logistic regression, then random forest, then gradient boosting — each
trained on TRAIN and read on VALIDATION only. TEST is never touched
here; ``research.candidacy`` (and, for a model specifically, whatever
step 7's research run wires around this module) owns TEST-once
discipline. No neural networks, per explicit instruction.

Every model is interpreted, not treated as an opaque signal generator:
``ModelFitResult`` always carries ``feature_importance`` (logistic
regression's coefficients, or the tree ensembles' built-in impurity-based
importances) alongside its VALIDATION win-rate read — a model that
"worked" must be explainable, not just trusted, same requirement the
approved plan places on every discovered condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from otc_research.backtest.splits import compute_temporal_split
from otc_research.research.baseline import WinRateStat, win_rate_stat

#: Fixed order the approved plan specifies: simplest/most interpretable
#: first, escalating only as far as needed.
MODEL_FAMILIES: tuple[str, ...] = ("logistic_regression", "random_forest", "gradient_boosting")


def _make_model(family: str, *, seed: int = 0):
    if family == "logistic_regression":
        return LogisticRegression(max_iter=1000)
    if family == "random_forest":
        return RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed)
    if family == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    raise ValueError(f"unknown model family {family!r}; expected one of {MODEL_FAMILIES}")


@dataclass(frozen=True)
class ModelFitResult:
    family: str
    feature_cols: tuple[str, ...]
    target_col: str
    train_sample_size: int
    #: Win-rate stat over the VALIDATION rows the model would have traded
    #: (predicted P(win) > ``probability_threshold``) — never a generic
    #: accuracy score; this project's unit of success is always "does the
    #: win rate clear break-even", not classification accuracy.
    validation_stat_at_threshold: WinRateStat
    #: Fraction of VALIDATION rows the model would have traded at all.
    validation_coverage: float
    probability_threshold: float
    feature_importance: dict[str, float]
    #: The fitted classifier itself (fit on TRAIN only, never refit) —
    #: added for Step 10 (``research.model_strategy.ModelStrategy``), so a
    #: model that shows a promising VALIDATION read can be wrapped as a
    #: Strategy and pushed through the same candidacy funnel every other
    #: strategy family here goes through, instead of staying a purely
    #: informational VALIDATION-only read.
    model: object


def train_val_frames(
    df: pd.DataFrame, train_fraction: float, validation_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Row-position split matching ``backtest.splits.compute_temporal_split``
    exactly — the same 60/20/20-style convention every other split in this
    codebase uses. ``df`` must already be sorted ascending by timestamp,
    one row per candle (``research.dataset.build_dataset``'s own
    contract). Never returns the TEST slice; this module has no
    legitimate reason to see it.
    """
    split = compute_temporal_split(len(df), train_fraction, validation_fraction)
    return df.iloc[split.train_slice], df.iloc[split.validation_slice]


def _feature_importance(model, feature_cols: Sequence[str]) -> dict[str, float]:
    if hasattr(model, "coef_"):
        values = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        return {}
    return {feature: float(value) for feature, value in zip(feature_cols, values)}


def fit_and_evaluate(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    family: str,
    probability_threshold: float = 0.5,
    seed: int = 0,
) -> ModelFitResult:
    """Fits ``family`` on TRAIN rows with a resolved target (drops rows
    with a missing feature or an unresolved/tied target — the same NaN
    convention ``research.dataset.build_dataset`` already establishes for
    those targets), then reads VALIDATION performance as a win-rate stat
    over the rows it would have traded.
    """
    train_resolved = train_df.dropna(subset=[*feature_cols, target_col])
    if train_resolved.empty:
        raise ValueError("no resolved TRAIN rows with complete features for this target")

    y_train = train_resolved[target_col].to_numpy()
    if len(set(y_train)) < 2:
        raise ValueError(
            f"TRAIN target {target_col!r} has only one outcome class present "
            "-- cannot fit a classifier on it"
        )
    X_train = train_resolved[list(feature_cols)].to_numpy()

    model = _make_model(family, seed=seed)
    model.fit(X_train, y_train)

    validation_resolved = validation_df.dropna(subset=[*feature_cols, target_col])
    X_val = validation_resolved[list(feature_cols)].to_numpy()
    probabilities = model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.array([])

    would_trade = probabilities > probability_threshold
    traded_targets = validation_resolved.loc[would_trade, target_col]
    stat = win_rate_stat(traded_targets)
    coverage = (
        float(would_trade.sum()) / len(validation_resolved) if len(validation_resolved) > 0 else 0.0
    )

    return ModelFitResult(
        family=family,
        feature_cols=tuple(feature_cols),
        target_col=target_col,
        train_sample_size=len(train_resolved),
        validation_stat_at_threshold=stat,
        validation_coverage=coverage,
        probability_threshold=probability_threshold,
        feature_importance=_feature_importance(model, feature_cols),
        model=model,
    )


def run_model_comparison(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    probability_threshold: float = 0.5,
    seed: int = 0,
    families: Sequence[str] = MODEL_FAMILIES,
) -> list[ModelFitResult]:
    """Runs every family in the plan's fixed order and returns their
    results in that same order — callers decide what "worked" means (e.g.
    ``validation_stat_at_threshold.ci_low`` clearing break-even with
    enough coverage to be useful), this function only runs and reports,
    the same separation of concerns as ``research.discovery.run_discovery``.
    """
    return [
        fit_and_evaluate(
            train_df,
            validation_df,
            feature_cols,
            target_col,
            family=family,
            probability_threshold=probability_threshold,
            seed=seed,
        )
        for family in families
    ]
