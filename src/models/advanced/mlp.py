"""
PyTorch multilayer perceptron for PaceDelta_s, wrapped as an sklearn regressor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from torch import Tensor, nn

from src.features.preprocessing import make_preprocessor
from src.models.dataset import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class LapTimeMLP(nn.Module):
    """
    Fully connected net: ``n_features -> hidden[0] -> ... -> 1``.

    Each hidden block is ``Linear -> ReLU -> Dropout``. Output is one scalar per
    row (shape ``(batch,)``), the predicted PaceDelta_s.
    """

    def __init__(self, n_features: int, hidden: Sequence[int] = (64, 32), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainHistory:
    """Per-epoch mean losses (MSE) and the epoch whose weights were kept."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1  # 0-based index into val_loss


def train_mlp(
    model: nn.Module,
    X_train: Tensor,
    y_train: Tensor,
    X_val: Tensor,
    y_val: Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    seed: int = 0,
) -> TrainHistory:
    """
    Mini-batch training with Adam + MSE loss and early stopping on validation loss.

    Each epoch: shuffle the training rows, update weights batch by batch, then
    record the mean training loss and the full validation loss. Stop when the
    validation loss has not improved for ``patience`` consecutive epochs, then
    load the weights of the best epoch back into ``model`` (in place).

    Args:
        model: Network to train (modified in place).
        X_train, y_train: float32 tensors, shapes ``(n, f)`` and ``(n,)``.
        X_val, y_val: Validation tensors, same layout.
        epochs: Maximum number of epochs.
        batch_size: Rows per gradient step.
        lr: Adam learning rate.
        weight_decay: L2 penalty (Adam's ``weight_decay``).
        patience: Epochs without val improvement before stopping.
        seed: Seed for the batch shuffling (``torch.Generator``).

    Returns:
        ``TrainHistory`` with one entry per epoch actually run, and
        ``best_epoch == argmin(val_loss)``.
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history = TrainHistory()
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
        mean_train_loss = train_loss_sum / len(X_train)
        history.train_loss.append(mean_train_loss)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = loss_fn(val_preds, y_val).item()
            history.val_loss.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        return history
    else:
        raise RuntimeError("No best state found during training.")


class TorchMLPRegressor(RegressorMixin, BaseEstimator):
    """
    sklearn-style wrapper: ``fit(X_df, y)`` / ``predict(X_df)`` around preprocessing + LapTimeMLP.

    Being a proper sklearn estimator lets the MLP be benchmarked, conformalized
    and (later) cross-checked with MAPIE exactly like Ridge and CatBoost.
    """

    def __init__(
        self,
        hidden: Sequence[int] = (64, 32),
        dropout: float = 0.1,
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 20,
        seed: int = 0,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed

    def _to_tensor(self, X: pd.DataFrame) -> Tensor:
        return torch.as_tensor(self.preprocessor_.transform(X), dtype=torch.float32)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> TorchMLPRegressor:
        """
        Fit the preprocessor on X, then train the net with early stopping on
        (X_val, y_val). Without validation data, the training rows double as
        validation (early stopping then only guards against divergence).
        """
        torch.manual_seed(self.seed)
        self.preprocessor_ = make_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(X)
        if X_val is None or y_val is None:
            X_val, y_val = X, y
        X_t, X_v = self._to_tensor(X), self._to_tensor(X_val)
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float32))
        y_v = torch.as_tensor(np.asarray(y_val, dtype=np.float32))
        self.model_ = LapTimeMLP(X_t.shape[1], self.hidden, self.dropout)
        self.history_ = train_mlp(
            self.model_, X_t, y_t, X_v, y_v,
            epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
            weight_decay=self.weight_decay, patience=self.patience, seed=self.seed,
        )
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]:
        self.model_.eval()
        with torch.no_grad():
            return self.model_(self._to_tensor(X)).numpy().astype(np.float64)
