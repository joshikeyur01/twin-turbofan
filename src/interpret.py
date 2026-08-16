"""What is the twin actually looking at?

    python -m src.interpret --epochs 80

Writes outputs/interpretability.md, outputs/interpretability.png

Two complementary views, chosen because each covers the other's blind spot:

**1. Attention over cycles (model-specific).** `AttentionRegressor` pools its encoder states
with one weight per cycle, summing to 1, and the prediction *is* that weighted sum — so the
weights are the readout rather than a proxy for it. This answers "which part of the window
does the estimate rest on?". Expectation worth stating before looking: degradation is
monotonic here, so recent cycles should dominate. If attention were flat, the model would be
averaging over the window and the sequence structure would be doing nothing.

**2. Permutation importance over sensors (model-agnostic).** Shuffle one sensor's column
across the scoring windows and measure how much RMSE degrades. This works for any
architecture, and it measures what the model *relies on* rather than what it attends to —
attention can be high on a cycle whose sensors carry no usable signal.

A caveat that applies to permutation importance generally: correlated inputs share
responsibility, so shuffling one of two near-duplicate sensors understates both. On C-MAPSS
several sensors move together during HPC degradation, so read the ranking as "which signal
groups matter", not as a clean per-sensor attribution.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import load_config  # noqa: E402
from .data_loader import using_real_data  # noqa: E402
from .evaluate import rmse  # noqa: E402
from .paths import OUTPUTS_DIR  # noqa: E402

N_TOP_SENSORS = 12


def attention_profile(model, loader, device) -> np.ndarray:
    """Mean attention weight per window position, averaged over all scoring windows."""
    rows = []
    for xb, _ in loader:
        w = model.attention_weights(xb.to(device))
        rows.append(w.float().cpu().numpy())
    return np.concatenate(rows).mean(axis=0)


def permutation_importance(model, ds, feat_cols, device, seed=0, repeats=3) -> dict[str, float]:
    """RMSE increase when each feature column is shuffled across windows.

    Shuffling is done *across windows* at every timestep, which destroys that feature's
    relationship to the target while preserving its marginal distribution.
    """
    import torch
    from torch.utils.data import DataLoader

    def score(transform=None) -> float:
        trues, preds = [], []
        for xb, yb in DataLoader(ds, batch_size=512):
            x = xb.clone()
            if transform is not None:
                x = transform(x)
            with torch.no_grad():
                out = model(x.to(device))
            preds.append(np.clip(out.float().cpu().numpy(), 0, None))
            trues.append(yb.numpy())
        return rmse(np.concatenate(trues), np.concatenate(preds))

    base = score()
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}

    def permute_column(x, j):
        """Shuffle feature ``j`` across the rows of this batch, keeping time order intact.

        Permuting *within the batch* keeps this a genuine permutation whatever the batch
        size. Indexing a batch with a permutation of ``len(ds)`` would only be valid while
        the whole dataset fits in one batch; otherwise it maps many rows onto few and the
        "shuffle" becomes biased rather than random.
        """
        idx = torch.from_numpy(rng.permutation(len(x)))
        x[:, :, j] = x[idx, :, j]
        return x

    for j, name in enumerate(feat_cols):
        deltas = [score(lambda x, j=j: permute_column(x, j)) - base for _ in range(repeats)]
        out[name] = float(np.mean(deltas))
    return out


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default=cfg.subset)
    p.add_argument("--seq-len", type=int, default=cfg.seq_len)
    p.add_argument("--hidden", type=int, default=cfg.hidden)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--patience", type=int, default=cfg.patience)
    p.add_argument("--repeats", type=int, default=3, help="permutation shuffles per feature")
    a = p.parse_args()

    from torch.utils.data import DataLoader

    from .data_loader import load_cmapss
    from .features import build_xy
    from .train_seq import WindowDataset, train

    model, result, _ = train(
        arch="attention",
        subset=a.subset,
        seq_len=a.seq_len,
        hidden=a.hidden,
        lr=a.lr,
        epochs=a.epochs,
        patience=a.patience,
        quiet=False,
    )
    device = next(model.parameters()).device
    model.eval()

    train_raw, test_raw = load_cmapss(a.subset)
    _, test_f, feat_cols, _ = build_xy(train_raw, test_raw)
    ds = WindowDataset(test_f, feat_cols, a.seq_len, last_only=True)

    profile = attention_profile(model, DataLoader(ds, batch_size=512), device)
    print(f"\nattention: first cycle {profile[0]:.4f}  last cycle {profile[-1]:.4f}")

    # Concentration: how much of the total weight sits in the most recent quarter?
    tail = int(np.ceil(len(profile) / 4))
    recent_share = float(profile[-tail:].sum())
    uniform_share = tail / len(profile)
    print(
        f"last {tail} cycles hold {recent_share:.1%} of attention "
        f"(uniform would be {uniform_share:.1%})"
    )

    print("\npermutation importance (this takes a moment)...")
    importance = permutation_importance(model, ds, feat_cols, device, repeats=a.repeats)
    ranked = sorted(importance.items(), key=lambda kv: -kv[1])
    for name, delta in ranked[:8]:
        print(f"  {name:16s} +{delta:7.3f} RMSE when shuffled")

    # Persist the expensive results FIRST, then render.
    #
    # This ordering is deliberate and was learned the hard way: an earlier version plotted
    # before writing, and an invalid matplotlib colour name raised inside `_plot` *after*
    # ~150s of training and a full permutation-importance pass — discarding all of it for a
    # purely cosmetic fault. Anything costly should be durable before anything decorative
    # can fail. The figure path is filled in afterwards.
    json_path = OUTPUTS_DIR / "interpretability.json"
    json_path.write_text(
        json.dumps(
            {
                "result": result,
                "attention_profile": [round(float(v), 5) for v in profile],
                "recent_quarter_share": round(recent_share, 4),
                "importance": {k: round(v, 4) for k, v in importance.items()},
            },
            indent=2,
        )
    )
    print(f"Saved raw results -> {json_path.name} (before plotting)")

    fig_path = _plot(profile, ranked, a.subset, recent_share, uniform_share)
    lines = _report(result, profile, ranked, recent_share, uniform_share, a, fig_path)
    (OUTPUTS_DIR / "interpretability.md").write_text("\n".join(lines))
    print(f"Saved -> outputs/interpretability.md, {fig_path.name}")


def _report(result, profile, ranked, recent_share, uniform_share, a, fig_path) -> list[str]:
    concentrated = recent_share > 1.5 * uniform_share
    top = ranked[0]
    dead = [n for n, d in ranked if d <= 0.001]

    return [
        f"# Interpretability — attention model ({a.subset})",
        "",
        f"**Data: {'real NASA C-MAPSS' if using_real_data() else 'SYNTHETIC (plumbing only)'}**",
        "",
        f"`AttentionRegressor` (GRU encoder + additive attention pooling), "
        f"seq_len={a.seq_len}, hidden={a.hidden}, lr={a.lr:g}. "
        f"Test RMSE **{result['rmse']}**, PHM **{result['phm']}** "
        f"(best epoch {result['best_epoch']} of {result['epochs_run']}).",
        "",
        "## Which cycles does the estimate rest on?",
        "",
        "The prediction is the attention-weighted sum of encoder states, so these weights are",
        "the readout itself, not a proxy for it. Each row of weights sums to 1.",
        "",
        f"- First cycle in the window: **{profile[0]:.4f}**",
        f"- Last (most recent) cycle: **{profile[-1]:.4f}**",
        f"- Most recent quarter of the window holds **{recent_share:.1%}** of total attention "
        f"(uniform attention would give {uniform_share:.1%})",
        "",
        (
            "So attention is **concentrated on recent cycles**, which is what monotonic "
            "degradation should produce: the newest readings carry the most information about "
            "current health, and older ones mostly repeat it."
            if concentrated
            else (
                "Attention is close to **uniform** across the window. That is worth flagging "
                "rather than glossing: it means the model is effectively averaging, and the "
                "sequence structure is contributing little beyond smoothing — consistent with "
                "the ablation finding that rolling means already capture most of the signal on "
                "this data."
            )
        ),
        "",
        "## Which sensors does it rely on?",
        "",
        "Permutation importance: RMSE increase when a feature column is shuffled across the",
        "scoring windows. Model-agnostic, and measures reliance rather than attention.",
        "",
        "| feature | ΔRMSE when shuffled |",
        "|---|---|",
        *[f"| `{n}` | +{d:.3f} |" for n, d in ranked[:N_TOP_SENSORS]],
        "",
        f"Top feature: **`{top[0]}`** (+{top[1]:.3f} RMSE).",
        (
            f"\n{len(dead)} of {len(ranked)} features change RMSE by ≤0.001 when destroyed — "
            "the model does not use them. On synthetic data that is unsurprising: the generator "
            "only lets a subset of sensors drift, and `select_informative_sensors` cannot drop "
            "the rest because they still carry noise (see the ablation caveat)."
            if dead
            else ""
        ),
        "",
        "**Read the ranking as signal *groups*, not clean per-sensor attribution.** Correlated",
        "inputs share responsibility under permutation, so shuffling one of two near-duplicate",
        "sensors understates both — and C-MAPSS sensors move together during HPC degradation.",
        "",
        f"Figure: `{fig_path.name}`",
        "",
    ]


def _plot(profile, ranked, subset, recent_share, uniform_share):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    positions = np.arange(-len(profile) + 1, 1)  # 0 = most recent cycle
    ax1.plot(positions, profile, "o-", color="tab:purple", ms=3)
    ax1.axhline(1 / len(profile), color="k", ls="--", lw=1, label="uniform attention")
    ax1.set_xlabel("cycle offset from the scoring point (0 = most recent)")
    ax1.set_ylabel("mean attention weight")
    ax1.set_title(
        f"Attention over the window\nrecent quarter holds {recent_share:.0%} "
        f"(uniform: {uniform_share:.0%})"
    )
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    top = ranked[:N_TOP_SENSORS][::-1]
    names = [n for n, _ in top]
    vals = [d for _, d in top]
    ax2.barh(np.arange(len(top)), vals, color="tab:cyan", edgecolor="k", alpha=0.85)
    ax2.set_yticks(np.arange(len(top)))
    ax2.set_yticklabels(names, fontsize=7)
    ax2.set_xlabel("ΔRMSE when shuffled")
    ax2.set_title("Permutation importance (top features)")
    ax2.grid(alpha=0.3, axis="x")

    fig.suptitle(f"twin-turbofan — what the twin looks at ({subset})")
    fig.tight_layout()
    path = OUTPUTS_DIR / "interpretability.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
