"""Tests for src.models.conformal."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.models.conformal import (
    conformal_interval,
    interval_metrics,
    mapie_split_conformal,
    split_conformal_quantile,
)


# --- split_conformal_quantile (student) -------------------------------------------

def test_quantile_is_kth_smallest_abs_residual() -> None:
    residuals = np.arange(1.0, 20.0)  # n = 19
    # k = ceil(20 * 0.9) = 18 -> 18th smallest = 18.0
    assert split_conformal_quantile(residuals, 0.9) == 18.0


def test_quantile_ignores_sign_and_order() -> None:
    rng = np.random.default_rng(0)
    residuals = np.arange(1.0, 20.0) * rng.choice([-1, 1], 19)
    rng.shuffle(residuals)
    assert split_conformal_quantile(residuals, 0.9) == 18.0


def test_quantile_infinite_when_too_few_points() -> None:
    # n = 10, k = ceil(11 * 0.95) = 11 > 10
    assert split_conformal_quantile(np.ones(10), 0.95) == math.inf


def test_quantile_returns_python_float() -> None:
    assert type(split_conformal_quantile(np.arange(100.0), 0.95)) is float


@pytest.fixture
def linear_problem():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(30_500, 2)), columns=["a", "b"])
    y = X["a"] * 1.5 - X["b"] + rng.standard_t(df=4, size=len(X))  # heavy tails
    model = LinearRegression().fit(X.iloc[:500], y.iloc[:500])
    # 10k calibration points: coverage varies with the calibration draw by
    # ~sqrt(0.05 * 0.95 / n), i.e. +/-0.2 % here (+/-1 % with only n = 500).
    return model, (X.iloc[500:10_500], y.iloc[500:10_500]), (X.iloc[10_500:], y.iloc[10_500:])


def test_matches_mapie(linear_problem) -> None:
    model, (X_cal, y_cal), (X_test, _) = linear_problem
    q = split_conformal_quantile(y_cal - model.predict(X_cal), 0.95)
    lower, upper = conformal_interval(model.predict(X_test), q)
    _, m_lower, m_upper = mapie_split_conformal(model, X_cal, y_cal, X_test, 0.95)
    np.testing.assert_allclose(lower, m_lower)
    np.testing.assert_allclose(upper, m_upper)


def test_coverage_guarantee_holds_on_exchangeable_data(linear_problem) -> None:
    model, (X_cal, y_cal), (X_test, y_test) = linear_problem
    q = split_conformal_quantile(y_cal - model.predict(X_cal), 0.95)
    lower, upper = conformal_interval(model.predict(X_test), q)
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    assert 0.94 <= coverage <= 0.96


# --- interval_metrics (student) ---------------------------------------------------

def test_interval_metrics_values() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.0, 2.0, 3.5, 0.0])
    upper = np.array([2.0, 2.0, 4.0, 3.0])  # inside, on-boundary (counts), below, above
    out = interval_metrics(y, lower, upper)
    assert out == {"coverage": 0.5, "mean_width": 1.375}
    assert all(type(v) is float for v in out.values())
