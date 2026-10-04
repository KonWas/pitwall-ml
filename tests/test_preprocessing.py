"""Tests for src.features.preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.features.preprocessing import make_preprocessor

NUM = ["TyreLife", "TrackTemp"]
CAT = ["Compound"]


@pytest.fixture
def train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TyreLife": [1.0, 2.0, 3.0, np.nan, 5.0],
            "TrackTemp": [30.0, 32.0, 34.0, 36.0, 38.0],
            "Compound": ["SOFT", "HARD", "SOFT", "MEDIUM", "HARD"],
            "Unused": ["x"] * 5,
        }
    )


def test_returns_unfitted_column_transformer() -> None:
    pre = make_preprocessor(NUM, CAT)
    assert isinstance(pre, ColumnTransformer)
    assert not hasattr(pre, "transformers_")  # fitted attribute absent


def test_output_shape_dense_no_nan(train: pd.DataFrame) -> None:
    out = make_preprocessor(NUM, CAT).fit_transform(train)
    assert isinstance(out, np.ndarray)
    assert out.shape == (5, len(NUM) + 3)  # 2 numeric + 3 one-hot, "Unused" dropped
    assert not np.isnan(out).any()


def test_numeric_columns_standardised_on_train(train: pd.DataFrame) -> None:
    out = make_preprocessor(NUM, CAT).fit_transform(train)
    np.testing.assert_allclose(out[:, :2].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(out[:, :2].std(axis=0), 1.0, atol=1e-12)


def test_test_rows_use_train_statistics(train: pd.DataFrame) -> None:
    pre = make_preprocessor(NUM, CAT).fit(train)
    test = pd.DataFrame({"TyreLife": [np.nan], "TrackTemp": [100.0], "Compound": ["WET"]})
    out = pre.transform(test)
    imputed_train = np.array([1.0, 2.0, 3.0, 2.5, 5.0])  # NaN -> train median 2.5
    expected_tyre = (2.5 - imputed_train.mean()) / imputed_train.std()
    assert out[0, 0] == pytest.approx(expected_tyre)
    assert out[0, 1] == pytest.approx((100.0 - 34.0) / np.std([30, 32, 34, 36, 38]))
    np.testing.assert_array_equal(out[0, 2:], [0.0, 0.0, 0.0])  # unseen category -> zeros
