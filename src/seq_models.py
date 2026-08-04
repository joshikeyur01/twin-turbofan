"""Sequence architectures for RUL regression.

All of them take a batch of windows shaped ``(batch, seq_len, n_features)`` and return
one RUL estimate per window, shaped ``(batch,)``. Sharing that contract is what lets
``train_seq.py`` drive any of them through one training and evaluation path.

Why these four:

- **LSTM** — the standard baseline for C-MAPSS. Gated recurrence handles the long,
  slow degradation ramp without the vanishing-gradient problem of a plain RNN.
- **GRU** — same idea, one fewer gate and no cell state. Fewer parameters, so it is
  the useful control for "is the LSTM's extra capacity actually earning anything?"
- **1D-CNN** — no recurrence at all. Convolutions over the time axis detect local
  degradation *shapes* in parallel, which is far faster to train and often
  competitive. It is the control for whether sequential modelling is needed at all.
- **Attention** — a GRU encoder whose final-timestep readout is replaced by additive
  attention pooling. Added for interpretability rather than accuracy: it yields one
  weight per cycle and the prediction *is* that weighted sum, so ``src/interpret.py``
  can say which cycles an estimate rests on. See the class docstring.

Torch is imported lazily by the caller (``train_seq``) so the core pipeline stays
importable with only numpy/pandas/matplotlib.
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """Stacked LSTM; prediction is read from the final timestep's hidden state."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            layers,
            batch_first=True,
            # torch ignores dropout on a 1-layer RNN and warns; don't ask for it.
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class GRURegressor(nn.Module):
    """Same shape as the LSTM with a simpler gating scheme — the capacity control."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            n_features,
            hidden,
            layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class CNN1DRegressor(nn.Module):
    """Temporal CNN over the window.

    Conv1d expects ``(batch, channels, time)``, so the input is transposed on entry —
    sensors become channels and the convolution slides along time. Global average
    pooling then collapses the time axis, which keeps the head independent of
    ``seq_len`` (so the same architecture works across the sequence-length sweep).
    """

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        blocks = []
        in_ch = n_features
        for _ in range(max(1, layers)):
            blocks += [
                nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = hidden
        self.conv = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2))  # (B, T, F) -> (B, F, T)
        h = h.mean(dim=2)  # global average pool over time
        return self.head(h).squeeze(-1)


class AttentionRegressor(nn.Module):
    """Recurrent encoder with **additive attention pooling** over the window.

    Why this rather than a plain transformer encoder: the point of adding attention here is
    interpretability, and multi-head self-attention maps are notoriously hard to read as
    explanations (many heads, and attention-to-token does not equal token-importance). A
    single additive-attention pooling layer produces exactly one weight per cycle, summing
    to 1, that *is* the model's readout — the prediction is literally the weighted sum. That
    makes "which cycles drove this estimate?" answerable rather than suggestive.

    ``attention_weights`` exposes those weights for ``src/interpret.py``.

    The GRU encoder is kept underneath because the sweep showed recurrence is what works on
    this data; attention replaces only the final-timestep readout, so any gain is
    attributable to the pooling rather than to a different sequence model.
    """

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.GRU(
            n_features,
            hidden,
            layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        # score_t = v . tanh(W h_t)  -> softmax over t
        self.attn = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.encoder(x)  # (B, T, H)
        weights = torch.softmax(self.attn(h).squeeze(-1), dim=1)  # (B, T)
        return h, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, weights = self._encode(x)
        pooled = torch.einsum("bth,bt->bh", h, weights)
        return self.head(pooled).squeeze(-1)

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Per-cycle attention weights, shape (batch, seq_len), each row summing to 1."""
        return self._encode(x)[1]


ARCHITECTURES = {
    "lstm": LSTMRegressor,
    "gru": GRURegressor,
    "cnn": CNN1DRegressor,
    "attention": AttentionRegressor,
}


# Single source of truth for CLI `choices=`. Registering an architecture above must make
# it reachable from every entry point; four scripts previously hardcoded
# ["lstm", "gru", "cnn"], so adding "attention" silently failed at the argparse layer
# while the model itself worked fine.
ARCH_NAMES = sorted(ARCHITECTURES)


def make_seq_model(arch: str, n_features: int, **kwargs) -> nn.Module:
    """Build one of ``ARCHITECTURES`` by name."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch](n_features, **kwargs)
