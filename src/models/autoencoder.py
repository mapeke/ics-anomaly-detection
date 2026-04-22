"""Dense autoencoder in PyTorch.

Reconstruction MSE per sample is the anomaly score. `attribute(X)` returns
the per-feature squared error — trivial attribution for free.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import AnomalyDetector


class _DenseAE(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (64, 32, 16, 32, 64)):
        super().__init__()
        dims = [input_dim, *hidden]
        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU())
        # Reconstruct back to input_dim with sigmoid (features are [0, 1]).
        layers.append(nn.Linear(dims[-1], input_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseAutoencoderAD(AnomalyDetector):
    name = "dense_ae"

    def __init__(
        self,
        input_dim: int,
        hidden: tuple[int, ...] = (64, 32, 16, 32, 64),
        learning_rate: float = 1e-3,
        epochs: int = 25,
        batch_size: int = 512,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self.input_dim = input_dim
        self.hidden = tuple(hidden)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.verbose = verbose
        self.model = _DenseAE(input_dim=input_dim, hidden=hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.history: dict[str, list[float]] = {"loss": [], "val_loss": []}

    def fit(
        self, X_train: np.ndarray, X_val: np.ndarray | None = None
    ) -> "DenseAutoencoderAD":
        X_train_t = torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32))
        train_loader = DataLoader(
            TensorDataset(X_train_t),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        X_val_t: torch.Tensor | None = None
        if X_val is not None and len(X_val):
            X_val_t = torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32)).to(
                self.device
            )

        for epoch in range(self.epochs):
            self.model.train()
            running = 0.0
            n = 0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                recon = self.model(batch)
                loss = self.loss_fn(recon, batch)
                loss.backward()
                self.optimizer.step()
                running += loss.item() * len(batch)
                n += len(batch)
            train_loss = running / max(n, 1)
            self.history["loss"].append(train_loss)

            val_loss = float("nan")
            if X_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    recon = self.model(X_val_t)
                    val_loss = self.loss_fn(recon, X_val_t).item()
                self.history["val_loss"].append(val_loss)

            if self.verbose:
                print(f"  epoch {epoch + 1:3d}/{self.epochs}  loss={train_loss:.5f}  val={val_loss:.5f}")

        return self

    def _per_feature_se(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            recon = self.model(X_t)
            se = (X_t - recon) ** 2
        return se.cpu().numpy()

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._per_feature_se(X).mean(axis=1)

    def attribute(self, X: np.ndarray) -> np.ndarray:
        return self._per_feature_se(X)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        (path / "hyperparams.json").write_text(
            json.dumps(
                {
                    "input_dim": self.input_dim,
                    "hidden": list(self.hidden),
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                }
            )
        )
        (path / "meta.json").write_text(json.dumps({"name": self.name}))

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "DenseAutoencoderAD":
        path = Path(path)
        kwargs = json.loads((path / "hyperparams.json").read_text())
        kwargs["hidden"] = tuple(kwargs["hidden"])
        kwargs["device"] = device
        obj = cls(**kwargs)
        state = torch.load(path / "model.pt", map_location=device)
        obj.model.load_state_dict(state)
        obj.model.eval()
        return obj
