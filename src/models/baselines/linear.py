"""
Linear baseline: standardised features -> Ridge regression.
"""

from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.features.preprocessing import make_preprocessor
from src.models.dataset import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """
    Unfitted ``preprocess -> Ridge`` pipeline on ``MODEL_FEATURES``.

    Ridge = least squares + ``alpha * ||w||^2`` penalty, which shrinks
    coefficients of correlated features (AirTemp vs TrackTemp vs AirDensity)
    instead of letting them blow up in opposite directions. Because the
    preprocessor sits INSIDE the pipeline, ``pipeline.fit(X_train, y_train)``
    learns imputation/scaling statistics from training rows only.
    """
    return Pipeline(
        [
            ("preprocess", make_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
