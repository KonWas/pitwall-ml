"""End-to-end test for src.models.benchmark (synthetic data, small models)."""

from __future__ import annotations

import pandas as pd

from src.models.benchmark import ID_COLUMNS, run_benchmark


def test_run_benchmark_end_to_end(model_table: pd.DataFrame) -> None:
    result = run_benchmark(
        model_table,
        n_val=1,
        n_calib=2,
        n_test=2,
        catboost_params={"iterations": 300},
        mlp_params={"epochs": 150, "lr": 3e-3, "batch_size": 64},
    )
    m = result.metrics.set_index("model")
    assert list(m.index) == ["Naive", "Ridge", "CatBoost", "MLP"]
    assert list(result.metrics.columns) == [
        "model", "MAE", "RMSE", "MAE_vs_naive", "q", "coverage", "mean_width"
    ]
    assert m.loc["Naive", "MAE_vs_naive"] == 0.0
    assert (m.loc[["Ridge", "CatBoost", "MLP"], "MAE_vs_naive"] > 0.3).all()
    assert m["coverage"].between(0.8, 1.0).all()
    # Better point predictions -> narrower conformal intervals.
    assert m.loc["Ridge", "mean_width"] < m.loc["Naive", "mean_width"]

    preds = result.predictions
    assert len(preds) == 4 * len(result.split.test)
    assert set(ID_COLUMNS + ["model", "y_true", "y_pred", "lower", "upper"]) <= set(preds.columns)
    assert (preds["lower"] <= preds["upper"]).all()
