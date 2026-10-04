"""
Naive persistence baseline: "this lap will be exactly as fast as recent form".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin


class NaivePaceBaseline(RegressorMixin, BaseEstimator):
    """
    Always predicts a PaceDelta_s of 0, i.e. lap time = RollingPace_s (+ fuel).

    Every model must beat this to justify itself. In forecasting, the
    "tomorrow = today" forecast is often surprisingly hard to beat, and a model
    that doesn't is adding complexity, not information.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> NaivePaceBaseline:
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]:
        return np.zeros(len(X), dtype=np.float64)
