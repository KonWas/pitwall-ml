"""Tests for src.models.baselines and src.models.advanced."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import mean_absolute_error

from src.models.advanced.gbdt import fit_catboost, make_catboost
from src.models.advanced.mlp import LapTimeMLP, TorchMLPRegressor, train_mlp
from src.models.baselines.linear import make_ridge_pipeline
from src.models.baselines.naive import NaivePaceBaseline
from src.models.dataset import feature_matrix


@pytest.fixture
def xy_split(model_table: pd.DataFrame):
    """Rounds 1-5 train, 6 val, 7-8 test (done by hand: independent of student split)."""
    parts = {
        "train": model_table[model_table["RoundNumber"] <= 5],
        "val": model_table[model_table["RoundNumber"] == 6],
        "test": model_table[model_table["RoundNumber"] >= 7],
    }
    return {k: feature_matrix(v.reset_index(drop=True)) for k, v in parts.items()}


def test_naive_predicts_zero(xy_split) -> None:
    X, y = xy_split["train"]
    pred = NaivePaceBaseline().fit(X, y).predict(xy_split["test"][0])
    assert pred.shape == (len(xy_split["test"][0]),)
    assert (pred == 0).all()


def test_ridge_beats_naive(xy_split) -> None:
    (X, y), (X_test, y_test) = xy_split["train"], xy_split["test"]
    pred = make_ridge_pipeline().fit(X, y).predict(X_test)
    assert mean_absolute_error(y_test, pred) < 0.6 * mean_absolute_error(y_test, np.zeros_like(pred))


# --- fit_catboost (student) ---------------------------------------------------------

def test_catboost_early_stopping_keeps_best_iteration(xy_split) -> None:
    (X, _), (X_val, _) = xy_split["train"], xy_split["val"]
    rng = np.random.default_rng(0)
    y_noise, y_val_noise = rng.normal(size=len(X)), rng.normal(size=len(X_val))  # nothing to learn
    model = make_catboost(iterations=3000, learning_rate=0.3)
    fitted = fit_catboost(model, X, y_noise, X_val, y_val_noise, early_stopping_rounds=50)
    assert fitted is model
    assert model.tree_count_ < 3000  # stopped early
    assert model.tree_count_ == model.get_best_iteration() + 1  # truncated to best


def test_catboost_learns_signal(xy_split) -> None:
    (X, y), (X_val, y_val), (X_test, y_test) = xy_split["train"], xy_split["val"], xy_split["test"]
    model = fit_catboost(make_catboost(iterations=500), X, y, X_val, y_val)
    pred = model.predict(X_test)
    assert mean_absolute_error(y_test, pred) < 0.6 * mean_absolute_error(y_test, np.zeros_like(pred))


# --- train_mlp (student) -------------------------------------------------------------

def _tensors(n: int, seed: int, noise: float):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 3, generator=g)
    y = X @ torch.tensor([1.0, -2.0, 0.5]) + noise * torch.randn(n, generator=g)
    return X, y


def test_train_mlp_learns_and_restores_best_weights() -> None:
    torch.manual_seed(0)
    X, y = _tensors(800, seed=1, noise=0.1)
    X_val, y_val = _tensors(200, seed=2, noise=0.1)
    model = LapTimeMLP(3, hidden=(32,), dropout=0.0)
    hist = train_mlp(model, X, y, X_val, y_val, epochs=60, batch_size=64, lr=1e-2, patience=60)

    assert len(hist.train_loss) == len(hist.val_loss) == 60
    assert hist.best_epoch == int(np.argmin(hist.val_loss))
    assert min(hist.val_loss) < 0.1 * hist.val_loss[0]  # actually learned
    model.eval()
    with torch.no_grad():
        final_val = torch.nn.functional.mse_loss(model(X_val), y_val).item()
    assert final_val == pytest.approx(hist.val_loss[hist.best_epoch], rel=1e-5)


def test_train_mlp_stops_after_patience() -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(3)
    X, y = torch.randn(400, 3, generator=g), torch.randn(400, generator=g)  # pure noise
    X_val, y_val = torch.randn(200, 3, generator=g), torch.randn(200, generator=g)
    model = LapTimeMLP(3, hidden=(64,), dropout=0.0)
    hist = train_mlp(model, X, y, X_val, y_val, epochs=500, lr=1e-2, patience=5)
    assert len(hist.val_loss) < 500
    assert len(hist.val_loss) == hist.best_epoch + 5 + 1


def test_train_mlp_is_reproducible() -> None:
    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        X, y = _tensors(300, seed=1, noise=0.5)
        X_val, y_val = _tensors(100, seed=2, noise=0.5)
        model = LapTimeMLP(3, hidden=(16,), dropout=0.1)
        runs.append(train_mlp(model, X, y, X_val, y_val, epochs=5, seed=7).train_loss)
    assert runs[0] == runs[1]


def test_torch_regressor_end_to_end(xy_split) -> None:
    (X, y), (X_val, y_val), (X_test, y_test) = xy_split["train"], xy_split["val"], xy_split["test"]
    reg = TorchMLPRegressor(epochs=150, lr=3e-3).fit(X, y, X_val=X_val, y_val=y_val)
    pred = reg.predict(X_test)
    assert pred.dtype == np.float64 and pred.shape == (len(X_test),)
    assert mean_absolute_error(y_test, pred) < 0.6 * mean_absolute_error(y_test, np.zeros_like(pred))
