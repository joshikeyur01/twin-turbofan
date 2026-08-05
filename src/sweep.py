"""Hyperparameter sweep over sequence length, hidden size and learning rate.

Run:
    python -m src.sweep --arch lstm
    python -m src.sweep --arch cnn --seq-lens 20 50 --hiddens 64 --lrs 1e-3

Writes outputs/sweep_<arch>.md, outputs/sweep_<arch>.json

**Configurations are ranked by validation RMSE, never by test.** The test column is
printed alongside so the selection can be sanity-checked, but choosing the row with
the best *test* number would be selecting on the evaluation set and would quietly
inflate every headline metric. The "selected" configuration reported at the bottom is
always the best-validation row.
"""

from __future__ import annotations

import argparse
import itertools
import json

import pandas as pd

from .data_loader import using_real_data
from .paths import OUTPUTS_DIR

DEFAULT_SEQ_LENS = [20, 30, 50]
DEFAULT_HIDDENS = [32, 64, 128]
DEFAULT_LRS = [3e-4, 1e-3, 3e-3]


def main():
    from .seq_models import ARCH_NAMES

    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="lstm", choices=ARCH_NAMES)
    p.add_argument("--subset", default="FD001")
    p.add_argument("--seq-lens", type=int, nargs="*", default=DEFAULT_SEQ_LENS)
    p.add_argument("--hiddens", type=int, nargs="*", default=DEFAULT_HIDDENS)
    p.add_argument("--lrs", type=float, nargs="*", default=DEFAULT_LRS)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--layers", type=int, default=2)
    a = p.parse_args()

    from .train_seq import train  # lazy: keeps torch out of the import path until needed

    grid = list(itertools.product(a.seq_lens, a.hiddens, a.lrs))
    print(
        f"sweeping {a.arch} on {a.subset}: {len(grid)} configs "
        f"({'REAL' if using_real_data() else 'SYNTHETIC'} data)"
    )
    print(
        f"{'seq_len':>7} {'hidden':>6} {'lr':>7} {'val_rmse':>9} {'test_rmse':>9} "
        f"{'test_phm':>9} {'ep':>3} {'s':>5}"
    )

    rows = []
    for seq_len, hidden, lr in grid:
        _, res, _ = train(
            arch=a.arch,
            subset=a.subset,
            seq_len=seq_len,
            hidden=hidden,
            lr=lr,
            layers=a.layers,
            epochs=a.epochs,
            patience=a.patience,
            quiet=True,
        )
        rows.append(res)
        print(
            f"{seq_len:>7} {hidden:>6} {lr:>7.4f} {res['val_rmse']:>9.3f} "
            f"{res['rmse']:>9.3f} {res['phm']:>9.1f} {res['best_epoch']:>3} "
            f"{res['train_s']:>5.1f}",
            flush=True,
        )

    df = pd.DataFrame(rows)[
        [
            "seq_len",
            "hidden",
            "lr",
            "n_params",
            "val_rmse",
            "rmse",
            "phm",
            "pct_late",
            "best_epoch",
            "train_s",
        ]
    ].sort_values("val_rmse")

    best = df.iloc[0]
    lines = [
        f"# Hyperparameter sweep — {a.arch.upper()} on {a.subset}",
        "",
        f"**Data: {'real NASA C-MAPSS' if using_real_data() else 'SYNTHETIC fallback'}**",
        "",
        f"{len(grid)} configurations, up to {a.epochs} epochs each with early-stopping "
        f"patience {a.patience}. Ranked by **validation** RMSE.",
        "",
        df.to_markdown(index=False),
        "",
        "## Selected configuration",
        "",
        f"Chosen on validation RMSE ({best['val_rmse']}):",
        "",
        f"- `seq_len={int(best['seq_len'])}`, `hidden={int(best['hidden'])}`, "
        f"`lr={best['lr']:g}` — {int(best['n_params']):,} parameters",
        f"- Test RMSE **{best['rmse']}**, test PHM **{best['phm']}**, "
        f"{best['pct_late']}% late, best epoch {int(best['best_epoch'])}",
        "",
        f"Best *test* RMSE in the grid was {df['rmse'].min()} — reported only as a check. "
        "Selecting on it would be selecting on the evaluation set.",
        "",
    ]
    (OUTPUTS_DIR / f"sweep_{a.arch}.md").write_text("\n".join(lines))
    (OUTPUTS_DIR / f"sweep_{a.arch}.json").write_text(json.dumps(rows, indent=2))

    print(
        f"\nselected (best val): seq_len={int(best['seq_len'])} "
        f"hidden={int(best['hidden'])} lr={best['lr']:g} -> "
        f"test rmse={best['rmse']} phm={best['phm']}"
    )
    print(f"Saved -> outputs/sweep_{a.arch}.md")


if __name__ == "__main__":
    main()
