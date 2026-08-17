"""Does blending the RandomForest with the best sequence model help?

    python -m src.ensemble --arch gru

Writes outputs/ensemble.md, outputs/ensemble.json, outputs/ensemble.png

**The honest part is the weight.** Blending exposes a free parameter, and picking it by
looking at the test score would manufacture an improvement out of nothing — with 50 test
engines and a whole curve to choose from, *something* will beat both models. So the weight
is chosen on held-out **validation** engines and the test curve is printed only as a check.

To make that work, the RandomForest is fit on exactly the engines the sequence model
trained on. ``train_seq.train`` holds out 20% of training engines for early stopping using
``split_engines(..., seed)``; calling the same function with the same seed here reproduces
that split, so the validation engines are unseen by *both* models and are a fair place to
choose a blend weight.

Two models are combined as ``w * sequence + (1 - w) * forest``, so ``w=0`` is the forest
alone and ``w=1`` the sequence model alone — both endpoints appear in the reported curve,
which is what makes "did it help?" answerable rather than assumed.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data_loader import last_cycle_rows, load_cmapss, using_real_data  # noqa: E402
from .evaluate import phm_score, rmse  # noqa: E402
from .features import build_xy  # noqa: E402
from .models import make_model  # noqa: E402
from .paths import OUTPUTS_DIR  # noqa: E402

WEIGHTS = np.round(np.arange(0.0, 1.01, 0.1), 2)


def _seq_predict(model, df, feat_cols, seq_len, device, last_only=False):
    """Sequence-model predictions for ``df``, aligned to (unit, cycle) sort order."""
    import torch
    from torch.utils.data import DataLoader

    from .train_seq import WindowDataset, predict

    ds = WindowDataset(df, feat_cols, seq_len, last_only=last_only)
    dl = DataLoader(ds, batch_size=512)
    with torch.no_grad():
        y_true, y_pred = predict(model, dl, device)
    return y_true, y_pred


def main():
    from .seq_models import ARCH_NAMES

    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="gru", choices=ARCH_NAMES)
    p.add_argument("--subset", default="FD001")
    p.add_argument("--seq-len", type=int, default=50)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    from .train_seq import split_engines, train

    train_raw, test_raw = load_cmapss(a.subset)
    train_f, test_f, feat_cols, _ = build_xy(train_raw, test_raw)

    # Same split, same seed as train_seq -> the forest never sees the seq model's val set.
    tr_df, val_df, val_units = split_engines(train_f, a.val_frac, a.seed)
    print(f"fit engines={tr_df['unit'].nunique()}  blend-val engines={len(val_units)}")

    forest, forest_name = make_model()
    forest.fit(tr_df[feat_cols].to_numpy(), tr_df["RUL"].to_numpy())
    print(f"forest: {forest_name}")

    seq_model, seq_result, _ = train(
        arch=a.arch,
        subset=a.subset,
        seq_len=a.seq_len,
        hidden=a.hidden,
        lr=a.lr,
        epochs=a.epochs,
        patience=a.patience,
        val_frac=a.val_frac,
        seed=a.seed,
        quiet=True,
    )
    device = seq_model.parameters().__next__().device
    print(f"{a.arch}: test rmse={seq_result['rmse']} phm={seq_result['phm']}")

    # --- Choose the weight on validation engines (all cycles). ---
    # Last-cycle scoring is meaningless here: validation engines are run-to-failure, so
    # their final cycle always has RUL 0. Every cycle is used instead.
    val_sorted = val_df.sort_values(["unit", "cycle"])
    v_true, v_seq = _seq_predict(seq_model, val_df, feat_cols, a.seq_len, device)
    v_forest = np.clip(forest.predict(val_sorted[feat_cols].to_numpy()), 0, None)
    assert len(v_seq) == len(v_forest) == len(val_sorted), "val predictions misaligned"
    np.testing.assert_allclose(v_true, val_sorted["RUL"].to_numpy(), rtol=1e-5)

    # --- Test predictions, last-cycle protocol, aligned by unit. ---
    test_last = last_cycle_rows(test_f).sort_values("unit")
    t_true_seq, t_seq = _seq_predict(
        seq_model, test_f, feat_cols, a.seq_len, device, last_only=True
    )
    t_true = test_last["RUL"].to_numpy()
    t_forest = np.clip(forest.predict(test_last[feat_cols].to_numpy()), 0, None)
    np.testing.assert_allclose(
        t_true_seq, t_true, rtol=1e-5, err_msg="test rows misaligned between models"
    )

    rows = []
    for w in WEIGHTS:
        val_blend = w * v_seq + (1 - w) * v_forest
        test_blend = w * t_seq + (1 - w) * t_forest
        resid = test_blend - t_true
        rows.append(
            {
                "w_seq": float(w),
                "val_rmse": round(rmse(v_true, val_blend), 3),
                "rmse": round(rmse(t_true, test_blend), 3),
                "phm": round(phm_score(t_true, test_blend), 1),
                "pct_late": round(100.0 * float((resid > 0).mean()), 1),
            }
        )
        print(
            f"  w={w:.1f}  val_rmse={rows[-1]['val_rmse']:7.3f}  "
            f"test_rmse={rows[-1]['rmse']:7.3f}  test_phm={rows[-1]['phm']:7.1f}"
        )

    df = pd.DataFrame(rows)
    chosen = df.loc[df["val_rmse"].idxmin()]
    forest_only = df[df["w_seq"] == 0.0].iloc[0]
    seq_only = df[df["w_seq"] == 1.0].iloc[0]
    best_single_rmse = min(forest_only["rmse"], seq_only["rmse"])
    best_single_phm = min(forest_only["phm"], seq_only["phm"])

    helps_rmse = chosen["rmse"] < best_single_rmse
    helps_phm = chosen["phm"] < best_single_phm

    # Raw results persisted before plotting: a rendering fault must not discard
    # expensive computation. See src/interpret.py for the incident that motivated this.
    (OUTPUTS_DIR / "ensemble.json").write_text(
        json.dumps({"chosen_w": float(chosen["w_seq"]), "rows": rows}, indent=2)
    )

    fig_path = _plot(df, chosen, a.arch, a.subset)

    verdict = _verdict(chosen, forest_only, seq_only, helps_rmse, helps_phm)
    lines = [
        f"# Ensemble — RandomForest + {a.arch.upper()} ({a.subset})",
        "",
        f"**Data: {'real NASA C-MAPSS' if using_real_data() else 'SYNTHETIC (plumbing only)'}**",
        "",
        f"Blend: `w * {a.arch} + (1 - w) * forest`. Both models fit on the same "
        f"{tr_df['unit'].nunique()} engines; weight chosen on {len(val_units)} held-out "
        "validation engines by RMSE, **never on test**.",
        "",
        df.to_markdown(index=False),
        "",
        "## Read-out",
        "",
        f"- Forest alone (w=0.0): RMSE **{forest_only['rmse']}**, PHM **{forest_only['phm']}**",
        f"- {a.arch.upper()} alone (w=1.0): RMSE **{seq_only['rmse']}**, "
        f"PHM **{seq_only['phm']}**",
        f"- Validation-selected blend (w={chosen['w_seq']:.1f}): RMSE **{chosen['rmse']}**, "
        f"PHM **{chosen['phm']}**",
        "",
        verdict,
        "",
        f"Figure: `{fig_path.name}`",
        "",
    ]
    (OUTPUTS_DIR / "ensemble.md").write_text("\n".join(lines))
    print(f"\n{verdict}")
    print(f"Saved -> outputs/ensemble.md, {fig_path.name}")


def _verdict(chosen, forest_only, seq_only, helps_rmse, helps_phm) -> str:
    """Derived from the measured numbers so the report cannot overclaim."""
    w = chosen["w_seq"]
    if w in (0.0, 1.0):
        winner = "the forest" if w == 0.0 else "the sequence model"
        return (
            f"**No, blending does not help.** Validation picked w={w:.1f} — i.e. {winner} "
            "alone. The blend curve offered nothing the better single model did not already "
            "provide, which is the honest outcome when one model dominates the other on "
            "every engine rather than making complementary errors."
        )
    if helps_rmse and helps_phm:
        return (
            f"**Yes, blending helps on both metrics.** w={w:.1f} beats the better single "
            f"model on RMSE ({chosen['rmse']} vs {min(forest_only['rmse'], seq_only['rmse'])}) "
            f"and on PHM ({chosen['phm']} vs {min(forest_only['phm'], seq_only['phm'])}), so "
            "the two models are making partly independent errors."
        )
    if helps_rmse or helps_phm:
        metric = "RMSE" if helps_rmse else "PHM"
        other = "PHM" if helps_rmse else "RMSE"
        return (
            f"**Mixed.** The validation-selected blend (w={w:.1f}) improves {metric} over "
            f"the better single model but not {other}. Worth reporting as a partial result "
            "rather than a win — the metrics disagree here just as they did for the "
            "ridge-vs-forest comparison."
        )
    return (
        f"**No.** Validation chose an interior weight (w={w:.1f}), but on test it beats "
        "neither single model on either metric. That gap between validation and test is "
        "itself the finding: with 50 test engines, blend weights are chosen on noise."
    )


def _plot(df, chosen, arch, subset):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, col, title in [
        (ax1, "rmse", "Test RMSE (lower is better)"),
        (ax2, "phm", "Test PHM (lower is better)"),
    ]:
        ax.plot(df["w_seq"], df[col], "o-", color="tab:blue")
        ax.axvline(
            chosen["w_seq"],
            color="tab:green",
            ls="--",
            lw=1.3,
            label=f"val-selected w={chosen['w_seq']:.1f}",
        )
        ax.set_xlabel(f"weight on {arch.upper()}  (0 = forest only, 1 = {arch} only)")
        ax.set_ylabel(col.upper())
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    ax1b = ax1.twinx()
    ax1b.plot(df["w_seq"], df["val_rmse"], "s--", color="tab:orange", alpha=0.7, ms=4)
    ax1b.set_ylabel("validation RMSE (selection criterion)", color="tab:orange", fontsize=8)
    ax1b.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle(f"twin-turbofan — RF + {arch.upper()} ensemble ({subset})")
    fig.tight_layout()
    path = OUTPUTS_DIR / "ensemble.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
