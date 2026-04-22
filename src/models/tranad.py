"""TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series.

Tuli, Casale, Jennings — *TranAD: Deep Transformer Networks for Anomaly
Detection in Multivariate Time Series Data*, VLDB 2022.

Architecture
------------
TranAD uses a Transformer encoder shared between two decoders, plus a
two-phase adversarial training scheme guided by a per-window *focus
score*:

    Phase 1: O1 = D1(Encoder(W, focus=zeros))
             reconstruct W from W with no anomaly emphasis.
    Phase 2: focus = ||W - O1||_2^2  (the per-element residual)
             O2 = D2(Encoder(W, focus=focus))
             reconstruct again, this time conditioning on the residual.

Training loss alternates the two heads with an adversarial weighting:

    n = epoch + 1
    L1 = (1/n) * ||W - O1||_2^2 + (1 - 1/n) * ||W - O2||_2^2
    L2 = (1/n) * ||W - O1||_2^2 - (1 - 1/n) * ||W - O2||_2^2

(D2 maximises the second-phase residual; D1 minimises it. The encoder
participates in both updates.)

Inference score (per window):
    s = 0.5 * ||W - O1||_2^2 + 0.5 * ||W - O2||_2^2

Divergences from the reference repo
-----------------------------------
* We use a single TransformerEncoderLayer (PyTorch built-in) with
  learnable positional encoding, instead of the paper's hand-rolled
  *Multi-Head Attention + position-wise FF* block. Functionally
  equivalent but a few hundred LOC lighter.
* Focus injection: we **concatenate** focus to W along the feature axis
  before the input projection (paper does the same in §3.2).
* Adversarial sign on L2 follows the published gradient form; we apply
  it via two optimisers (one per decoder) that share the encoder, so the
  encoder receives the sum of both gradients each step.
* Attribution: per-feature MSE from O2 averaged over the window. The
  paper proposes attention rollout for attribution; we provide the
  reconstruction-error attribution here and leave attention rollout for
  the dedicated Phase-4 attribution module.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import AnomalyDetector


class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _TranAD(nn.Module):
    def __init__(
        self,
        window: int,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window = window
        self.n_features = n_features
        # Input projection: features doubled because we concatenate focus.
        self.input_proj = nn.Linear(n_features * 2, d_model)
        self.pos = _PositionalEncoding(d_model, max_len=max(window, 256))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.decoder1 = nn.Linear(d_model, n_features)
        self.decoder2 = nn.Linear(d_model, n_features)

    def encode(self, x: torch.Tensor, focus: torch.Tensor) -> torch.Tensor:
        # x, focus: (B, W, F) -> concatenate along feature axis -> (B, W, 2F)
        h = torch.cat([x, focus], dim=-1)
        h = self.input_proj(h)
        h = self.pos(h)
        return self.encoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(x)
        h1 = self.encode(x, zeros)
        o1 = torch.sigmoid(self.decoder1(h1))           # (B, W, F)
        residual = (x - o1) ** 2
        h2 = self.encode(x, residual.detach())
        o2 = torch.sigmoid(self.decoder2(h2))
        return o1, o2

    def encoder_attention(self, x: torch.Tensor, focus: torch.Tensor) -> torch.Tensor:
        """Return encoder self-attention weights averaged over heads,
        shape ``(B, W, W)``. Bypasses the encoder's .forward to pick up
        ``need_weights=True`` from ``MultiheadAttention``.
        """
        # Input projection + positional encoding identical to .encode().
        h = torch.cat([x, focus], dim=-1)
        h = self.input_proj(h)
        h = self.pos(h)

        layer = self.encoder.layers[0]
        # PyTorch TransformerEncoderLayer has norm_first False by default;
        # self-attn is sub-block 1. We only need the attn weights.
        _, attn = layer.self_attn(
            h, h, h,
            need_weights=True,
            average_attn_weights=True,   # average over heads for rollout
        )
        return attn                      # (B, W, W)


class TranADModel(AnomalyDetector):
    name = "tranad"

    def __init__(
        self,
        window: int,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        epochs: int = 30,
        batch_size: int = 128,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self.window = window
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.verbose = verbose
        self.model = _TranAD(window, n_features, d_model, n_heads, ff_dim, dropout).to(self.device)
        self.opt1 = torch.optim.AdamW(
            [p for n, p in self.model.named_parameters()
             if not n.startswith("decoder2")],
            lr=learning_rate,
        )
        self.opt2 = torch.optim.AdamW(
            [p for n, p in self.model.named_parameters()
             if not n.startswith("decoder1")],
            lr=learning_rate,
        )
        self.history: dict[str, list[float]] = {"loss1": [], "loss2": [], "val": []}

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> "TranADModel":
        if X_train.ndim != 3:
            raise ValueError(f"TranAD needs (N, W, F) input; got {X_train.shape}")
        X_train_t = torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32))
        loader = DataLoader(
            TensorDataset(X_train_t), batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        X_val_t: torch.Tensor | None = None
        if X_val is not None and len(X_val):
            X_val_t = torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32)).to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            n_factor = epoch + 1
            running1, running2, n = 0.0, 0.0, 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                o1, o2 = self.model(batch)
                rec1 = ((batch - o1) ** 2).mean()
                rec2 = ((batch - o2) ** 2).mean()

                loss1 = (1.0 / n_factor) * rec1 + (1.0 - 1.0 / n_factor) * rec2
                loss2 = (1.0 / n_factor) * rec1 - (1.0 - 1.0 / n_factor) * rec2

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
                    o1, o2 = self.model(X_val_t)
                    val_loss = 0.5 * ((X_val_t - o1) ** 2).mean().item() + 0.5 * (
                        (X_val_t - o2) ** 2
                    ).mean().item()
                self.history["val"].append(val_loss)
                if self.verbose:
                    print(f"  epoch {epoch + 1:3d}/{self.epochs}  L1={self.history['loss1'][-1]:.5f}  "
                          f"L2={self.history['loss2'][-1]:.5f}  val={val_loss:.5f}")

        return self

    def _per_window_se(self, X: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """Return (N, W, F) squared error from the second-phase output."""
        self.model.eval()
        bs = batch_size or self.batch_size
        out = np.empty(X.shape, dtype=np.float32)
        for i in range(0, len(X), bs):
            chunk = torch.from_numpy(np.ascontiguousarray(X[i : i + bs], dtype=np.float32)).to(
                self.device
            )
            with torch.no_grad():
                o1, o2 = self.model(chunk)
                se = 0.5 * (chunk - o1) ** 2 + 0.5 * (chunk - o2) ** 2
            out[i : i + bs] = se.cpu().numpy()
        return out

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._per_window_se(X).mean(axis=(1, 2))

    def attribute(self, X: np.ndarray) -> np.ndarray:
        return self._per_window_se(X).mean(axis=1)

    def attribute_attention(self, X: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """Attention-weighted per-feature attribution.

        For each window, weight the per-(timestep, feature) reconstruction
        error by the encoder's attention mass: if output position *t* attends
        strongly to input position *i*, position *i*'s features contribute
        more to the window's attribution. Single-layer rollout degenerates
        to the attention matrix itself; we average it over output positions
        to get per-input-timestep weights, then sum the weighted per-feature
        squared error over time.

        Returns ``(N, F)`` — same schema as :py:meth:`attribute`, so the
        Phase-4 evaluator can be reused.
        """
        self.model.eval()
        bs = batch_size or self.batch_size
        N, W, F = X.shape
        out = np.empty((N, F), dtype=np.float32)
        for i in range(0, N, bs):
            chunk = torch.from_numpy(
                np.ascontiguousarray(X[i : i + bs], dtype=np.float32)
            ).to(self.device)
            with torch.no_grad():
                o1, o2 = self.model(chunk)                        # (B, W, F)
                se = 0.5 * (chunk - o1) ** 2 + 0.5 * (chunk - o2) ** 2
                # Use the *second-phase* encoder attention, since that path
                # is the one conditioned on the residual.
                residual = ((chunk - o1) ** 2).detach()
                attn = self.model.encoder_attention(chunk, residual)  # (B, W, W)
                # Aggregate over output positions → per-input-timestep weight.
                w_time = attn.mean(dim=1)                         # (B, W)
                w_time = w_time / (w_time.sum(dim=1, keepdim=True) + 1e-12)
                # Weighted sum of per-(t, f) SE over time.
                attr = (w_time.unsqueeze(-1) * se).sum(dim=1)     # (B, F)
            out[i : i + bs] = attr.cpu().numpy()
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        (path / "hyperparams.json").write_text(
            json.dumps(
                {
                    "window": self.window,
                    "n_features": self.n_features,
                    "d_model": self.d_model,
                    "n_heads": self.n_heads,
                    "ff_dim": self.ff_dim,
                    "dropout": self.dropout,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                }
            )
        )
        (path / "meta.json").write_text(json.dumps({"name": self.name}))

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "TranADModel":
        path = Path(path)
        kwargs = json.loads((path / "hyperparams.json").read_text())
        kwargs["device"] = device
        obj = cls(**kwargs)
        state = torch.load(path / "model.pt", map_location=device)
        obj.model.load_state_dict(state)
        obj.model.eval()
        return obj
