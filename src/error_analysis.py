"""Error analysis for the RUL baseline.

Answers two questions the aggregate RMSE / PHM numbers cannot:

1. **Does the twin track a single engine's degradation, or only get the endpoint
   right?** ``trajectories.png`` overlays predicted vs true RUL across every cycle
   of a handful of test engines.
2. **Is the error worse early in life or near failure?** ``residuals.png`` plots the
   signed residual against true RUL, with a binned mean, so bias is visible per
   life stage.

Sign convention matches ``evaluate.phm_score``: ``residual = predicted - true``, so
**positive means predicted LATE** (over-estimated remaining life). Late is the
dangerous direction — the part fails in service — and PHM penalises it harder.

Run (after ``python -m src.train_baseline``):
    python -m src.error_analysis

Writes outputs/trajectories.png, outputs/residuals.png, outputs/error_analysis.md
"""

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data_loader import last_cycle_rows, load_cmapss  # noqa: E402
from .evaluate import phm_score, rmse  # noqa: E402
from .features import build_xy  # noqa: E402
from .models import make_model  # noqa: E402
from .paths import OUTPUTS_DIR  # noqa: E402

# True-RUL bins for the life-stage breakdown. The top bin is the piecewise-linear
# cap, where the target is a plateau rather than a real countdown.
RUL_BINS = [0, 25, 50, 75, 100, 125]
N_TRAJECTORIES = 6


def _load_or_train(train_f, test_f, feat_cols):
    """Reuse outputs/baseline.pkl when present, else train a fresh model."""
    pkl = OUTPUTS_DIR / "baseline.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f), "loaded outputs/baseline.pkl"
    model, name = make_model()
    model.fit(train_f[feat_cols].to_numpy(), train_f["RUL"].to_numpy())
    return model, f"trained fresh ({name})"


def plot_trajectories(test_f, feat_cols, model, subset, n=N_TRAJECTORIES):
    """Predicted vs true RUL over the full recorded history of ``n`` test engines.

    Engines are picked spanning the range of recorded lengths, so the panel shows
    both barely-used and near-failure engines rather than n arbitrary ones.
    """
    lengths = test_f.groupby("unit")["cycle"].max().sort_values()
    picks = lengths.iloc[np.linspace(0, len(lengths) - 1, n).astype(int)].index.tolist()

    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)

    # strict=False on purpose: the grid may hold more panels than picked engines.
    for ax, unit in zip(axes.ravel(), picks, strict=False):
        g = test_f[test_f["unit"] == unit].sort_values("cycle")
        pred = np.clip(model.predict(g[feat_cols].to_numpy()), 0, None)
        ax.plot(g["cycle"], g["RUL"], "k-", lw=2, label="true RUL")
        ax.plot(g["cycle"], pred, "-", color="tab:orange", lw=1.5, label="predicted")
        ax.set_title(f"engine {unit}", fontsize=10)
        ax.set_xlabel("cycle")
        ax.set_ylabel("RUL")
        ax.grid(alpha=0.3)

    # Blank any unused panels so the grid doesn't show empty axes.
    for ax in axes.ravel()[len(picks) :]:
        ax.axis("off")

    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"twin-turbofan — predicted vs true RUL trajectories ({subset})")
    fig.tight_layout()
    path = OUTPUTS_DIR / "trajectories.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path, picks


def plot_residuals(y_true, y_pred, subset):
    """Signed residual vs true RUL, with binned means. Positive = predicted late."""
    resid = y_pred - y_true

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.axhline(0, color="k", ls="--", lw=1)
    ax1.scatter(y_true, resid, alpha=0.6, edgecolor="k")
    ax1.set_xlabel("True RUL (cycles)")
    ax1.set_ylabel("Residual (pred − true)")
    ax1.set_title("Residual vs true RUL\n(above zero = predicted LATE = dangerous)")
    ax1.grid(alpha=0.3)

    bins = pd.cut(y_true, RUL_BINS, include_lowest=True)
    by_bin = pd.DataFrame({"resid": resid, "bin": bins}).groupby("bin", observed=False)["resid"]
    centres = np.arange(len(by_bin.mean()))
    ax2.axhline(0, color="k", ls="--", lw=1)
    ax2.bar(
        centres,
        by_bin.mean().to_numpy(),
        yerr=by_bin.std().to_numpy(),
        capsize=4,
        color="tab:orange",
        edgecolor="k",
        alpha=0.85,
    )
    ax2.set_xticks(centres)
    ax2.set_xticklabels([str(i) for i in by_bin.mean().index], rotation=20, fontsize=8)
    ax2.set_xlabel("True RUL bin (cycles)")
    ax2.set_ylabel("Mean residual")
    ax2.set_title("Bias by life stage\n(left = near failure, right = early life)")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(f"twin-turbofan — error analysis ({subset})")
    fig.tight_layout()
    path = OUTPUTS_DIR / "residuals.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path, by_bin


def main(subset: str = "FD001"):
    train, test = load_cmapss(subset)
    train_f, test_f, feat_cols, _ = build_xy(train, test)

    model, how = _load_or_train(train_f, test_f, feat_cols)
    print(f"model: {how}")

    # Headline metrics use the last-cycle protocol, same as train_baseline.
    test_last = last_cycle_rows(test_f)
    y_true = test_last["RUL"].to_numpy()
    y_pred = np.clip(model.predict(test_last[feat_cols].to_numpy()), 0, None)

    traj_path, picks = plot_trajectories(test_f, feat_cols, model, subset)
    resid_path, by_bin = plot_residuals(y_true, y_pred, subset)

    resid = y_pred - y_true
    late = int((resid > 0).sum())
    summary = by_bin.agg(["count", "mean", "std"]).round(2)

    lines = [
        f"# Error analysis — {subset}",
        "",
        f"Model: {how}",
        f"Scored on each engine's last cycle ({len(y_true)} engines).",
        "",
        f"- RMSE: **{rmse(y_true, y_pred):.3f}**",
        f"- PHM score: **{phm_score(y_true, y_pred):.1f}**",
        f"- Mean residual (pred − true): **{resid.mean():+.2f}** cycles",
        f"- Predicted late (residual > 0): **{late}/{len(resid)}** engines "
        f"({100 * late / len(resid):.0f}%)",
        "",
        "## Bias by life stage",
        "",
        "Residual = predicted − true. Positive means the twin over-estimates remaining",
        "life, i.e. predicts failure later than it happens — the direction PHM punishes.",
        "",
        summary.to_markdown(),
        "",
        "## Figures",
        "",
        f"- `{traj_path.name}` — predicted vs true RUL trajectories (engines {picks})",
        f"- `{resid_path.name}` — residual vs true RUL + bias by life stage",
        "",
    ]
    report = OUTPUTS_DIR / "error_analysis.md"
    report.write_text("\n".join(lines))

    print(f"mean residual {resid.mean():+.2f}  late {late}/{len(resid)}")
    print(summary.to_string())
    print(f"Saved -> {traj_path.name}, {resid_path.name}, {report.name}")


if __name__ == "__main__":
    main()
