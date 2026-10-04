"""
scikit-learn preprocessing shared by every model that needs a purely numeric,
scaled input matrix (Ridge baseline, PyTorch MLP). CatBoost does NOT use this:
it handles NaN and categorical columns natively.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """
    Build an UNFITTED transformer: DataFrame -> dense float matrix.

    Numeric columns:     median imputation, then standardisation (mean 0, std 1).
    Categorical columns: one-hot encoding; categories unseen during fit encode
                         as all zeros instead of raising.
    Output columns: all numeric features (in the given order), then the one-hot
    columns. Any other input column is dropped. Output is a dense numpy array.

    Args:
        numeric_features: Names of numeric columns (may contain NaN).
        categorical_features: Names of string/categorical columns.

    Returns:
        An sklearn ``ColumnTransformer``. Call ``.fit`` on TRAINING rows only.
    """
    transformers = [
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            list(numeric_features),
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            list(categorical_features),
        ),
    ]
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)
