"""
Phase 3 benchmark: baselines vs. advanced models, point accuracy + conformal intervals.

Protocol (identical for every model):
    fit on TRAIN races (VAL races only for early stopping)
    -> residuals on CALIB races -> split-conformal half-width q
    -> point predictions + intervals on TEST races -> metrics
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.models.advanced.gbdt import fit_catboost, make_catboost
from src.models.advanced.mlp import TorchMLPRegressor
from src.models.baselines.linear import make_ridge_pipeline
from src.models.baselines.naive import NaivePaceBaseline
from src.models.conformal import conformal_interval, interval_metrics, split_conformal_quantile
from src.models.dataset import RaceSplit, chronological_race_split, feature_matrix

# Row identifiers copied into the predictions table (for per-race / per-driver plots).
ID_COLUMNS: list[str] = ["RaceId", "RoundNumber", "Driver", "LapNumber", "Compound", "TyreLife"]


@dataclass
class BenchmarkResult:
    split: RaceSplit
    metrics: pd.DataFrame  # one row per model
    predictions: pd.DataFrame  # one row per (test lap, model)
    models: dict[str, RegressorMixin] = field(default_factory=dict)


def _fit_models(
    split: RaceSplit,
    catboost_params: dict[str, Any] | None,
    mlp_params: dict[str, Any] | None,
) -> dict[str, RegressorMixin]:
    X_train, y_train = feature_matrix(split.train)
    X_val, y_val = feature_matrix(split.val)
    fitters: dict[str, Callable[[], RegressorMixin]] = {
        "Naive": lambda: NaivePaceBaseline().fit(X_train, y_train),
        "Ridge": lambda: make_ridge_pipeline().fit(X_train, y_train),
        "CatBoost": lambda: fit_catboost(
            make_catboost(**(catboost_params or {})), X_train, y_train, X_val, y_val
        ),
        "MLP": lambda: TorchMLPRegressor(**(mlp_params or {})).fit(
            X_train, y_train, X_val=X_val, y_val=y_val
        ),
    }
    return {name: fit() for name, fit in fitters.items()}


def run_benchmark(
    table: pd.DataFrame,
    *,
    confidence_level: float = 0.95,
    n_val: int = 3,
    n_calib: int = 3,
    n_test: int = 4,
    catboost_params: dict[str, Any] | None = None,
    mlp_params: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """
    Fit all models, conformalize on the calibration races, evaluate on test races.

    Args:
        table: Modelling table from ``load_model_table``.
        confidence_level: Target coverage of the prediction intervals.
        n_val, n_calib, n_test: Races per split (see ``chronological_race_split``).
        catboost_params: Overrides for ``make_catboost``.
        mlp_params: Keyword arguments for ``TorchMLPRegressor``.

    Returns:
        ``BenchmarkResult``. ``metrics`` columns: model, MAE, RMSE, MAE_vs_naive
        (relative improvement, > 0 is better), q, coverage, mean_width.
        ``predictions`` columns: ID_COLUMNS, model, y_true, y_pred, lower, upper.
    """
    split = chronological_race_split(table, n_val=n_val, n_calib=n_calib, n_test=n_test)
    models = _fit_models(split, catboost_params, mlp_params)
    X_calib, y_calib = feature_matrix(split.calib)
    X_test, y_test = feature_matrix(split.test)

    rows, pred_frames = [], []
    for name, model in models.items():
        q = split_conformal_quantile(y_calib - model.predict(X_calib), confidence_level)
        y_pred = model.predict(X_test)
        lower, upper = conformal_interval(y_pred, q)
        rows.append(
            {
                "model": name,
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": root_mean_squared_error(y_test, y_pred),
                "q": q,
                **interval_metrics(y_test, lower, upper),
            }
        )
        pred_frames.append(
            split.test[ID_COLUMNS].assign(
                model=name, y_true=y_test.to_numpy(), y_pred=y_pred, lower=lower, upper=upper
            )
        )

    metrics = pd.DataFrame(rows)
    naive_mae = metrics.loc[metrics["model"] == "Naive", "MAE"].iloc[0]
    metrics.insert(3, "MAE_vs_naive", 1 - metrics["MAE"] / naive_mae)
    return BenchmarkResult(
        split=split,
        metrics=metrics,
        predictions=pd.concat(pred_frames, ignore_index=True),
        models=models,
    )
