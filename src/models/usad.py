"""USAD: UnSupervised Anomaly Detection on multivariate time series.

Audibert, Marti, Guyard, Zuluaga, Granger — *USAD: UnSupervised Anomaly
Detection on Multivariate Time Series*, KDD 2020.

Architecture
------------
A shared encoder ``E`` and two decoders ``D1`` and ``D2`` form two
autoencoders that share the bottleneck:

    AE1(W) = D1(E(W))
    AE2(W) = D2(E(W))

Both encoder and decoders are dense MLPs operating on flattened windows.

Training is two-phase, alternated each epoch (we follow the joint-loss
formulation from §3.2 of the paper, swept by an ``epoch+1`` factor that
shifts emphasis from reconstruction to adversarial as training proceeds):

    n = epoch + 1
    L1 = (1/n) * ||W - AE1(W)||_2^2 + (1 - 1/n) * ||W - AE2(AE1(W))||_2^2
    L2 = (1/n) * ||W - AE2(W)||_2^2 - (1 - 1/n) * ||W - AE2(AE1(W))||_2^2

The discriminator term in ``L2`` is *adversarial*: AE2 maximises the
reconstruction error of AE2(AE1(W)), pushing the system to detect
slight reconstructions as anomalies in inference.

Inference score (per window):
    s = alpha * ||W - AE1(W)||_2 + beta * ||W - AE2(AE1(W))||_2
with alpha + beta = 1; defaults alpha=0.5.

Divergences from the reference repo
-----------------------------------
* We flatten the windowed input through dense layers (no convolutional
  variant). The paper's primary results use this MLP-USAD form.
* L2's optimisation step uses the same Adam optimizer as L1 with a single
  ``backward()`` per batch, summing the two losses (each gradient targets
  only its decoder's parameters). This matches the joint-update form
  rather than the alternating-step form.

Attribution: per-feature squared error from AE2(AE1(W)) averaged over
the window axis — same shape as ``LSTMAutoencoderAD.attribute``.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import AnomalyDetector


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, latent),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, output_dim: int, hidden: int, latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _USAD(nn.Module):
    def __init__(self, window: int, n_features: int, hidden: int = 100, latent: int = 40):
        super().__init__()
        flat_dim = window * n_features
        self.window = window
        self.n_features = n_features
        self.encoder = _Encoder(flat_dim, hidden, latent)
        self.decoder1 = _Decoder(flat_dim, hidden, latent)
        self.decoder2 = _Decoder(flat_dim, hidden, latent)

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_flat = self._flatten(x)
        z = self.encoder(x_flat)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        # Cycle-through term: AE2 of AE1(x).
        z_ae1 = self.encoder(w1)
        w12 = self.decoder2(z_ae1)
        return w1, w2, w12


class USADModel(AnomalyDetector):
    name = "usad"

    def __init__(
        self,
        window: int,
        n_features: int,
        hidden: int = 100,
        latent: int = 40,
        learning_rate: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 128,
        alpha: float = 0.5,
        beta: float = 0.5,
        device: str = "cpu",
        verbose: bool = False,
    ):
        if abs(alpha + beta - 1.0) > 1e-6:
            raise ValueError(f"USAD scoring weights must sum to 1; got alpha={alpha}, beta={beta}")
        self.window = window
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.device = torch.device(device)
        self.verbose = verbose
        self.model = _USAD(window, n_features, hidden, latent).to(self.device)
        self.opt1 = torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder1.parameters()),
            lr=learning_rate,
        )
        self.opt2 = torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder2.parameters()),
            lr=learning_rate,
        )
        self.history: dict[str, list[float]] = {"loss1": [], "loss2": [], "val": []}

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> "USADModel":
        if X_train.ndim != 3:
            raise ValueError(f"USAD needs (N, W, F) input; got {X_train.shape}")
        X_train_t = torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32))
        loader = DataLoader(
            TensorDataset(X_train_t), batch_size=self.batch_size, shuffle=True, drop_last=False
        )
        X_val_t: torch.Tensor | None = None
        if X_val is not None and len(X_val):
            X_val_t = torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32)).to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            n_factor = epoch + 1  # paper's "n" annealing
            running1, running2, n = 0.0, 0.0, 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                batch_flat = batch.reshape(batch.shape[0], -1)

                w1, w2, w12 = self.model(batch)

                rec1 = ((batch_flat - w1) ** 2).mean()
                rec2 = ((batch_flat - w2) ** 2).mean()
                cycle = ((batch_flat - w12) ** 2).mean()

                loss1 = (1.0 / n_factor) * rec1 + (1.0 - 1.0 / n_factor) * cycle
                loss2 = (1.0 / n_factor) * rec2 - (1.0 - 1.0 / n_factor) * cycle

                # Both backwards before either step — otherwise opt1.step()
                # mutates the encoder weights that loss2's autograd graph
                # still depends on.
                self.opt1.zero_grad()
                self.opt2.zero_grad()
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.opt1.step()
                self.opt2.step()

                running1 += loss1.item() * len(batch)
                running2 += loss2.item() * len(batch)
                n += len(batch)

            self.history["loss1"].append(running1 / max(n, 1))
            self.history["loss2"].append(running2 / max(n, 1))

            if X_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    w1, _, w12 = self.model(X_val_t)
                    val_flat = X_val_t.reshape(X_val_t.shape[0], -1)
                    val_loss = (
                        self.alpha * ((val_flat - w1) ** 2).mean().item()
                        + self.beta * ((val_flat - w12) ** 2).mean().item()
                    )
                self.history["val"].append(val_loss)
                if self.verbose:
                    print(f"  epoch {epoch + 1:3d}/{self.epochs}  L1={self.history['loss1'][-1]:.5f}  "
                          f"L2={self.history['loss2'][-1]:.5f}  val={val_loss:.5f}")

        return self

    def _per_window_se(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (per-window-feature SE from AE1, per-window-feature SE from AE2(AE1))."""
        self.model.eval()
        X_t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            w1, _, w12 = self.model(X_t)
            x_flat = X_t.reshape(X_t.shape[0], -1)
            se1 = ((x_flat - w1) ** 2).reshape(X_t.shape)
            se12 = ((x_flat - w12) ** 2).reshape(X_t.shape)
        return se1.cpu().numpy(), se12.cpu().numpy()

    def score(self, X: np.ndarray) -> np.ndarray:
        se1, se12 = self._per_window_se(X)
        # Mean over (window, feature) -> per-window score.
        return self.alpha * se1.mean(axis=(1, 2)) + self.beta * se12.mean(axis=(1, 2))

    def attribute(self, X: np.ndarray) -> np.ndarray:
        # Per-feature MSE from the cycle path averaged over the window axis.
        _, se12 = self._per_window_se(X)
        return se12.mean(axis=1)
