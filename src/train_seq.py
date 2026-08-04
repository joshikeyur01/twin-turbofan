"""One training + evaluation harness for every sequence model (LSTM / GRU / 1D-CNN).

Run:
    python -m src.train_seq --arch lstm
    python -m src.train_seq --arch cnn --seq-len 50 --hidden 128
    python -m src.train_seq --arch gru --json    # machine-readable, for the sweep

Protocol notes — the parts that are easy to get wrong:

**Split by engine, never by cycle.** Cycles from one engine are near-duplicates of
their neighbours, so splitting mid-engine leaks the answer into validation and every
val metric becomes optimistic fiction. ``split_engines`` partitions unit ids.

**Validation is scored over all windows, not last cycles.** Training engines run to
failure, so a held-out engine's final cycle always has RUL = 0. Scoring only there
would measure one degenerate point and mirror nothing about the test set, whose
engines are truncated mid-life. Early stopping therefore uses val RMSE across every
window of the held-out engines. (A closer mimic would randomly truncate each val
engine to imitate the test distribution — noted as a refinement, not done here.)

**Test scoring matches the RF baseline exactly:** one window per engine ending at its
last recorded cycle, then RMSE and the asymmetric PHM score. That is what makes the
sequence numbers comparable to `outputs/metrics.json`.

**Standardisation** reuses ``features.build_xy``, whose statistics come from the whole
training file — including the engines later held out for validation. That is a mild
optimism in the val number, kept deliberately so sequence models and the RF baseline
share identical inputs; test metrics are unaffected.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data_loader import load_cmapss
from .evaluate import phm_score, rmse
from .features import build_xy
from .paths import OUTPUTS_DIR
from .seq_models import ARCH_NAMES, make_seq_model


def pick_device(prefer: str = "auto") -> torch.device:
    """MPS on Apple silicon, CUDA if present, else CPU."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed every RNG the training path touches.

    Note: full bitwise determinism is not guaranteed on MPS, so repeated runs can
    differ slightly even with the seed fixed. Seeding still removes the large
    run-to-run swings that would otherwise swamp a hyperparameter comparison.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WindowDataset(Dataset):
    """Sliding windows of ``seq_len`` cycles, built lazily per item.

    Windows are cut *within* one engine and left-padded with zeros when the engine has
    not yet lived ``seq_len`` cycles. Because features are already standardised to zero
    mean, that padding is equivalent to mean-imputation rather than an injected outlier.

    ``last_only=True`` yields exactly one window per engine, ending at its final
    recorded cycle — the test-time scoring protocol.
    """

    def __init__(self, df, feat_cols, seq_len: int = 30, last_only: bool = False):
        self.seq_len = seq_len
        self.engines: list[tuple[np.ndarray, np.ndarray]] = []
        self.index: list[tuple[int, int]] = []
        self.units: list[int] = []

        for unit, g in df.sort_values(["unit", "cycle"]).groupby("unit"):
            X = g[feat_cols].to_numpy(dtype=np.float32)
            y = g["RUL"].to_numpy(dtype=np.float32)
            ei = len(self.engines)
            self.engines.append((X, y))
            if last_only:
                self.index.append((ei, len(g) - 1))
                self.units.append(int(unit))
            else:
                self.index.extend((ei, i) for i in range(len(g)))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        ei, i = self.index[k]
        X, y = self.engines[ei]
        lo = max(0, i - self.seq_len + 1)
        w = X[lo : i + 1]
        if len(w) < self.seq_len:
            pad = np.zeros((self.seq_len - len(w), X.shape[1]), dtype=np.float32)
            w = np.vstack([pad, w])
        return torch.from_numpy(np.ascontiguousarray(w)), torch.tensor(y[i])


def split_engines(df, val_frac: float = 0.2, seed: int = 42):
    """Partition by unit id so no engine's cycles straddle the split."""
    units = np.sort(df["unit"].unique())
    perm = np.random.default_rng(seed).permutation(units)
    n_val = max(1, int(round(len(units) * val_frac)))
    val_units = set(perm[:n_val].tolist())
    is_val = df["unit"].isin(val_units)
    return df[~is_val].copy(), df[is_val].copy(), sorted(val_units)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) with predictions clipped at 0 like the RF baseline."""
    model.eval()
    trues, preds = [], []
    for xb, yb in loader:
        out = model(xb.to(device))
        preds.append(out.float().cpu().numpy())
        trues.append(yb.numpy())
    y_true = np.concatenate(trues)
    y_pred = np.clip(np.concatenate(preds), 0, None)
    return y_true, y_pred


def train(
    arch: str = "lstm",
    subset: str = "FD001",
    seq_len: int = 30,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 20,
    batch: int = 256,
    val_frac: float = 0.2,
    patience: int = 5,
    seed: int = 42,
    device: str = "auto",
    quiet: bool = False,
):
    """Train one architecture and score it on the last-cycle test protocol."""
    set_seed(seed)
    dev = pick_device(device)

    train_raw, test_raw = load_cmapss(subset)
    train_f, test_f, feat_cols, _ = build_xy(train_raw, test_raw)
    tr_df, val_df, val_units = split_engines(train_f, val_frac, seed)

    tr_ds = WindowDataset(tr_df, feat_cols, seq_len)
    val_ds = WindowDataset(val_df, feat_cols, seq_len)
    test_ds = WindowDataset(test_f, feat_cols, seq_len, last_only=True)

    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=512)
    test_dl = DataLoader(test_ds, batch_size=512)

    model = make_seq_model(arch, len(feat_cols), hidden=hidden, layers=layers, dropout=dropout).to(
        dev
    )
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    if not quiet:
        print(
            f"arch={arch} device={dev.type} params={n_params:,} feats={len(feat_cols)} "
            f"seq_len={seq_len} hidden={hidden} layers={layers} lr={lr}"
        )
        print(
            f"engines: train={tr_df['unit'].nunique()} val={len(val_units)} "
            f"| windows: train={len(tr_ds):,} val={len(val_ds):,} test={len(test_ds)}"
        )

    # Tracked as separate locals rather than one dict: the three values have three
    # different types, and bundling them defeats type checking on every access.
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        train_mse = running / len(tr_ds)

        yv, pv = predict(model, val_dl, dev)
        val_rmse = rmse(yv, pv)
        history.append({"epoch": ep, "train_mse": train_mse, "val_rmse": val_rmse})

        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"
        else:
            marker = ""

        if not quiet:
            print(f"  epoch {ep:3d}  train_mse={train_mse:8.2f}  val_rmse={val_rmse:7.3f}{marker}")

        if ep - best_epoch >= patience:
            if not quiet:
                print(f"  early stop: no val improvement in {patience} epochs")
            break

    # Restore the best checkpoint before scoring — the final epoch is often not the best.
    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred = predict(model, test_dl, dev)
    resid = y_pred - y_true
    result = {
        "arch": arch,
        "subset": subset,
        "seq_len": seq_len,
        "hidden": hidden,
        "layers": layers,
        "lr": lr,
        "n_params": int(n_params),
        # Recorded so a result traces back to the run that produced it — the variance
        # study needs to tell same-seed repeats apart from different-seed runs.
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "val_rmse": round(best_val, 3),
        "rmse": round(rmse(y_true, y_pred), 3),
        "phm": round(phm_score(y_true, y_pred), 1),
        "mean_resid": round(float(resid.mean()), 2),
        "pct_late": round(100.0 * float((resid > 0).mean()), 1),
        "train_s": round(time.perf_counter() - t0, 1),
        "device": dev.type,
    }

    if not quiet:
        print(
            f"TEST  rmse={result['rmse']}  phm={result['phm']}  "
            f"late={result['pct_late']}%  (best epoch {result['best_epoch']}, "
            f"{result['train_s']}s)"
        )
    return model, result, history


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", default="lstm", choices=ARCH_NAMES)
    p.add_argument("--subset", default="FD001")
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--save", action="store_true", help="write weights to outputs/")
    p.add_argument("--json", action="store_true", help="print the result dict as JSON")
    a = p.parse_args()

    model, result, _ = train(
        arch=a.arch,
        subset=a.subset,
        seq_len=a.seq_len,
        hidden=a.hidden,
        layers=a.layers,
        dropout=a.dropout,
        lr=a.lr,
        epochs=a.epochs,
        batch=a.batch,
        val_frac=a.val_frac,
        patience=a.patience,
        seed=a.seed,
        device=a.device,
        quiet=a.json,
    )

    if a.save:
        path = OUTPUTS_DIR / f"{a.arch}_{a.subset}.pt"
        torch.save(model.state_dict(), path)
        print(f"saved {path}")
    if a.json:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
