"""
Gradient-boosted decision trees (CatBoost) for PaceDelta_s.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from catboost import CatBoostRegressor

from src.models.dataset import CATEGORICAL_FEATURES

DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "loss_function": "RMSE",
    "iterations": 2000,  # upper bound; early stopping picks the real number
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,  # no catboost_info/ folder in the repo
}


def make_catboost(**overrides: Any) -> CatBoostRegressor:
    """
    Unfitted CatBoost regressor with ``DEFAULT_CATBOOST_PARAMS`` (+ overrides).

    ``CATEGORICAL_FEATURES`` are declared by column name, so X must be a
    DataFrame (as returned by ``feature_matrix``). CatBoost handles NaN in
    numeric columns natively -- no imputation, no scaling needed.
    """
    params = {**DEFAULT_CATBOOST_PARAMS, **overrides}
    return CatBoostRegressor(cat_features=CATEGORICAL_FEATURES, **params)


def fit_catboost(
    model: CatBoostRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    early_stopping_rounds: int = 100,
) -> CatBoostRegressor:
    """
    Fit with early stopping on the validation races and keep the best iteration.

    Training stops once validation loss has not improved for
    ``early_stopping_rounds`` consecutive trees, and the returned model is
    truncated to the tree count with the lowest validation loss.

    Args:
        model: Unfitted model from :func:`make_catboost`.
        X_train, y_train: Training races.
        X_val, y_val: Validation races (never the calibration or test races).
        early_stopping_rounds: Patience, in trees.

    Returns:
        The same model object, fitted.
    """
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=early_stopping_rounds,
        use_best_model=True,
    )
    return model
