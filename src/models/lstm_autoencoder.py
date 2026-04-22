"""LSTM sequence-to-sequence autoencoder in PyTorch.

Input is a (N, window, F) tensor. The encoder consumes the sequence into a
latent vector; a decoder LSTM unrolls for `window` steps and a per-step
linear head projects back to F features. Score is per-window mean MSE;
attribution is per-feature MSE averaged across the window.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import AnomalyDetector


class _LSTMAE(nn.Module):
    def __init__(self, window: int, n_features: int, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.window = window
        self.n_features = n_features
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.enc_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, F)
        _, (h, _) = self.encoder(x)
        latent = torch.relu(self.enc_to_latent(h.squeeze(0)))            # (B, L)
        hidden = torch.relu(self.latent_to_hidden(latent))               # (B, H)
        repeated = hidden.unsqueeze(1).repeat(1, self.window, 1)         # (B, W, H)
        decoded, _ = self.decoder(repeated)
        return self.head(decoded)                                        # (B, W, F)


class LSTMAutoencoderAD(AnomalyDetector):
    name = "lstm_ae"

    def __init__(
        self,
        window: int,
        n_features: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 256,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self.window = window
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.verbose = verbose
        self.model = _LSTMAE(window, n_features, hidden_dim, latent_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.history: dict[str, list[float]] = {"loss": [], "val_loss": []}

    def fit(
        self, X_train: np.ndarray, X_val: np.ndarray | None = None
    ) -> "LSTMAutoencoderAD":
        if X_train.ndim != 3:
            raise ValueError(f"LSTM-AE needs (N, W, F) input; got {X_train.shape}")
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

    def _per_window_se(self, X: np.ndarray) -> np.ndarray:
        """Return per-(window, timestep, feature) squared errors."""
        self.model.eval()
        X_t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            recon = self.model(X_t)
            se = (X_t - recon) ** 2
        return se.cpu().numpy()

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._per_window_se(X).mean(axis=(1, 2))

    def attribute(self, X: np.ndarray) -> np.ndarray:
        # Per-feature MSE averaged over the window axis — shape (N, F).
        return self._per_window_se(X).mean(axis=1)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        (path / "hyperparams.json").write_text(
            json.dumps(
                {
                    "window": self.window,
                    "n_features": self.n_features,
                    "hidden_dim": self.hidden_dim,
                    "latent_dim": self.latent_dim,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                }
            )
        )
        (path / "meta.json").write_text(json.dumps({"name": self.name}))

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "LSTMAutoencoderAD":
        path = Path(path)
        kwargs = json.loads((path / "hyperparams.json").read_text())
        kwargs["device"] = device
        obj = cls(**kwargs)
        state = torch.load(path / "model.pt", map_location=device)
        obj.model.load_state_dict(state)
        obj.model.eval()
        return obj
