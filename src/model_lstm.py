"""LSTM RUL twin — trainable and evaluable.

This module was the Week-2 scaffold: it trained an LSTM but never evaluated it, and
carried its own copies of the window dataset and the model class. Both now live in
shared modules so the LSTM, GRU and CNN are driven by one code path:

- architectures → ``src/seq_models.py``
- training, by-engine splitting, last-cycle evaluation → ``src/train_seq.py``

What is kept here is the LSTM-specific entry point, so the documented command still
works — and now reports test metrics instead of stopping after the final epoch:

    python -m src.model_lstm

For anything else — other architectures, sweeps, machine-readable output — use the
harness directly:

    python -m src.train_seq --arch gru --seq-len 50 --json

Sequence models capture the *trajectory* of degradation rather than a single snapshot,
and typically beat the RandomForest baseline on real C-MAPSS.
"""

from __future__ import annotations

import torch

from .paths import OUTPUTS_DIR
from .seq_models import LSTMRegressor  # noqa: F401  (re-export, backwards compatible)
from .train_seq import WindowDataset, train  # noqa: F401  (re-export)

__all__ = ["LSTMRegressor", "WindowDataset", "main"]


def main(
    subset: str = "FD001",
    seq_len: int = 30,
    epochs: int = 20,
    batch: int = 256,
    lr: float = 1e-3,
):
    """Train the LSTM twin, score it on the last-cycle protocol, save the weights."""
    model, result, _ = train(
        arch="lstm",
        subset=subset,
        seq_len=seq_len,
        epochs=epochs,
        batch=batch,
        lr=lr,
    )
    path = OUTPUTS_DIR / "lstm_rul.pt"
    torch.save(model.state_dict(), path)
    print(f"saved {path}")
    return result


if __name__ == "__main__":
    main()
