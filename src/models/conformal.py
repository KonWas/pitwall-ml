"""
Split conformal prediction intervals: hand-written version + MAPIE cross-check.

Workflow (any fitted regressor):
    1. residuals on the CALIBRATION races  -> split_conformal_quantile -> q
    2. interval on new laps               =  prediction +/- q
    3. on the test races, check coverage  ~= confidence_level (interval_metrics)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin


def split_conformal_quantile(residuals: ArrayLike, confidence_level: float = 0.95) -> float:
    """
    Half-width ``q`` of a split-conformal interval from calibration residuals.

    With ``n`` calibration residuals ``r_i = y_i - y_hat_i`` and
    ``k = ceil((n + 1) * confidence_level)``, ``q`` is the k-th SMALLEST value
    of ``|r_i|`` (1-based). If ``k > n`` there are too few calibration points
    for the requested confidence and ``q`` is ``inf``.

    Args:
        residuals: Calibration residuals (signs don't matter, NaN not allowed).
        confidence_level: Target coverage, in (0, 1).

    Returns:
        ``q`` as a Python float (possibly ``inf``).
    """
    residuals = np.abs(residuals)
    n = len(residuals)
    k = int(np.ceil((n + 1) * confidence_level))
    if k > n:
        return float("inf")
    return float(np.sort(residuals)[k - 1])

def conformal_interval(
    predictions: ArrayLike, q: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Symmetric interval ``(predictions - q, predictions + q)``."""
    pred = np.asarray(predictions, dtype=np.float64)
    return pred - q, pred + q


def interval_metrics(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> dict[str, float]:
    """
    Empirical coverage and mean width of prediction intervals.

    Args:
        y_true: Observed values.
        lower, upper: Interval bounds, same length (bounds are inclusive).

    Returns:
        ``{"coverage": fraction of y_true inside [lower, upper],
           "mean_width": mean of upper - lower}`` as Python floats.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    mean_width = float(np.mean(upper - lower))

    return {"coverage": coverage, "mean_width": mean_width}


class _PrefitAdapter(RegressorMixin, BaseEstimator):
    """
    Pass-through wrapper for an already-fitted model, for MAPIE only.

    MAPIE probes models that expose ``n_features_in_`` with a float numpy array
    of zeros; CatBoost (string ``Compound`` column) rejects that. This adapter
    exposes ``fitted_`` instead, so MAPIE skips the probe and only ever calls
    ``predict`` on our real DataFrames.
    """

    def __init__(self, model: RegressorMixin):
        self.model = model
        self.fitted_ = True

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> _PrefitAdapter:
        raise NotImplementedError("prefit only")

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]:
        return np.asarray(self.model.predict(X), dtype=np.float64)


def mapie_split_conformal(
    fitted_model: RegressorMixin,
    X_calib: pd.DataFrame,
    y_calib: ArrayLike,
    X_test: pd.DataFrame,
    confidence_level: float = 0.95,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Same split-conformal interval computed by MAPIE (reference implementation).

    Args:
        fitted_model: Any already-fitted sklearn-compatible regressor.
        X_calib, y_calib: Calibration races.
        X_test: Rows to build intervals for.
        confidence_level: Target coverage.

    Returns:
        ``(predictions, lower, upper)`` for X_test.
    """
    mapie = SplitConformalRegressor(
        _PrefitAdapter(fitted_model),
        confidence_level=confidence_level,
        conformity_score="absolute",
        prefit=True,
    )
    mapie.conformalize(X_calib, y_calib)
    pred, bounds = mapie.predict_interval(X_test)  # bounds: (n, 2, n_confidence_levels)
    return pred.astype(np.float64), bounds[:, 0, 0], bounds[:, 1, 0]
