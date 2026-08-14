"""Feature ablation: do the rolling trend features actually earn their keep?

A single cycle's sensor reading is a snapshot; rolling mean and std over a window
are the cheapest way to give a non-sequential model some sense of *trajectory*.
This script quantifies that, and sweeps the window length, so the choice of
``window=5`` in ``features.build_xy`` is a measured decision rather than a default.

Every arm uses the same protocol as the baseline: split by engine, score on each
engine's last cycle, report both RMSE and the asymmetric PHM score.

Run:
    python -m src.ablation

Writes outputs/ablation.md, outputs/ablation.png
"""

import json
import time

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

# (label, use_rolling, window)
ARMS = [
    ("raw sensors only", False, None),
    ("+ rolling w=3", True, 3),
    ("+ rolling w=5", True, 5),
    ("+ rolling w=10", True, 10),
    ("+ rolling w=20", True, 20),
]


def run_arm(train, test, use_rolling, window, seed: int = 42):
    """Train and score one ablation arm. Returns a metrics dict.

    ``seed`` is threaded through to the forest so `compare.py` can average it over seeds
    alongside the sequence models.
    """
    t0 = time.perf_counter()
    train_f, test_f, feat_cols, _ = build_xy(
        train, test, window=window or 5, use_rolling=use_rolling
    )
    model, model_name = make_model(seed)
    model.fit(train_f[feat_cols].to_numpy(), train_f["RUL"].to_numpy())

    test_last = last_cycle_rows(test_f)
    y_true = test_last["RUL"].to_numpy()
    y_pred = np.clip(model.predict(test_last[feat_cols].to_numpy()), 0, None)
    resid = y_pred - y_true

    return {
        "n_features": len(feat_cols),
        "rmse": round(rmse(y_true, y_pred), 3),
        "phm": round(phm_score(y_true, y_pred), 1),
        "mean_resid": round(float(resid.mean()), 2),
        "pct_late": round(100.0 * float((resid > 0).mean()), 1),
        "fit_s": round(time.perf_counter() - t0, 1),
        "model": model_name,
    }


def plot(df, subset):
    """RMSE and PHM side by side; the two metrics need not agree."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(df))

    for ax, col, title in [
        (ax1, "rmse", "RMSE (lower is better)"),
        (ax2, "phm", "PHM score (lower is better)"),
    ]:
        best = df[col].min()
        colours = ["tab:green" if v == best else "tab:blue" for v in df[col]]
        ax.bar(x, df[col], color=colours, edgecolor="k", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(df["arm"], rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        for xi, v in zip(x, df[col], strict=True):
            ax.text(xi, v, f"{v:g}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"twin-turbofan — feature ablation ({subset})")
    fig.tight_layout()
    path = OUTPUTS_DIR / "ablation.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main(subset: str = "FD001"):
    train, test = load_cmapss(subset)

    rows = []
    for label, use_rolling, window in ARMS:
        res = run_arm(train, test, use_rolling, window)
        res["arm"] = label
        rows.append(res)
        print(
            f"{label:20s} feats={res['n_features']:3d}  "
            f"RMSE={res['rmse']:7.3f}  PHM={res['phm']:7.1f}  "
            f"late={res['pct_late']:4.1f}%  ({res['fit_s']}s)",
            flush=True,
        )

    df = pd.DataFrame(rows)[["arm", "n_features", "rmse", "phm", "mean_resid", "pct_late", "fit_s"]]
    # Raw results persisted before plotting: a rendering fault must not discard
    # expensive computation. See src/interpret.py for the incident that motivated this.
    (OUTPUTS_DIR / "ablation.json").write_text(json.dumps(rows, indent=2))

    path = plot(df, subset)

    baseline = df.iloc[0]
    best_rmse = df.loc[df["rmse"].idxmin()]
    best_phm = df.loc[df["phm"].idxmin()]

    lines = [
        f"# Feature ablation — {subset}",
        "",
        f"Model: {rows[0]['model']}. Scored on each engine's last cycle.",
        "",
        df.to_markdown(index=False),
        "",
        "## Read-out",
        "",
        f"- Raw-sensor baseline: RMSE **{baseline['rmse']}**, PHM **{baseline['phm']}** "
        f"({baseline['n_features']} features).",
        f"- Best RMSE: **{best_rmse['arm']}** at **{best_rmse['rmse']}** "
        f"({100 * (baseline['rmse'] - best_rmse['rmse']) / baseline['rmse']:+.1f}% vs raw).",
        f"- Best PHM: **{best_phm['arm']}** at **{best_phm['phm']}** "
        f"({100 * (baseline['phm'] - best_phm['phm']) / baseline['phm']:+.1f}% vs raw).",
        "",
        f"Rolling features triple the feature count ({baseline['n_features']} → "
        f"{df['n_features'].max()}), so a marginal gain may not justify the cost.",
        "",
        f"Figure: `{path.name}`",
        "",
    ]
    (OUTPUTS_DIR / "ablation.md").write_text("\n".join(lines))
    print(f"\nSaved -> ablation.md, ablation.json, {path.name}")


if __name__ == "__main__":
    main()
