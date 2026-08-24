"""Render the live twin as an animated GIF for the README.

    python -m src.make_demo_gif --unit 1

Generated from the twin's real output rather than screen-recorded, so it is reproducible,
needs no capture tooling, and cannot drift out of sync with the model: the frames come
from the same ``OnlineFeatureBuilder`` path that ``telemetry`` and ``dashboard`` use.

Writes docs/demo.gif
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from .paths import ROOT  # noqa: E402
from .telemetry import DEFAULT_THRESHOLD, engine_stream, load_twin  # noqa: E402

DOCS_DIR = ROOT / "docs"


def collect(unit: int, subset: str, threshold: float):
    """Replay one engine through the online path, returning the full series."""
    model, builder = load_twin()
    cycles, preds, trues = [], [], []
    for payload in engine_stream(unit, subset):
        cycles.append(payload["cycle"])
        preds.append(builder.predict(model, payload["unit"], payload["sensors"]))
        trues.append(payload["rul_true"])
    alert_at = next((c for c, p in zip(cycles, preds, strict=True) if p < threshold), None)
    return np.array(cycles), np.array(preds), np.array(trues), alert_at


def build(unit=1, subset="FD001", threshold=DEFAULT_THRESHOLD, frames=90, fps=12):
    cycles, preds, trues, alert_at = collect(unit, subset, threshold)
    # Subsample to a fixed frame count so GIF size stays bounded regardless of how many
    # cycles the engine ran for.
    step = max(1, len(cycles) // frames)
    idx = list(range(0, len(cycles), step)) + [len(cycles) - 1]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    fig.suptitle(f"twin-turbofan — live RUL, engine {unit}", fontsize=12)

    ax.set_xlim(0, cycles.max() * 1.02)
    ax.set_ylim(0, max(trues.max(), preds.max()) * 1.08)
    ax.axhline(threshold, color="tab:red", ls="--", lw=1.1, label=f"alert at {threshold:g}")
    (line_true,) = ax.plot([], [], "k-", lw=2, label="actual RUL")
    (line_pred,) = ax.plot([], [], "-", color="tab:orange", lw=1.8, label="twin estimate")
    banner = ax.text(
        0.015, 0.06, "", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom"
    )
    ax.set_ylabel("RUL (cycles)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax2.set_xlim(ax.get_xlim())
    div = preds - trues
    pad = max(1.0, float(np.abs(div).max()) * 1.15)
    ax2.set_ylim(-pad, pad)
    ax2.axhline(0, color="k", ls="--", lw=1)
    ax2.set_xlabel("cycle")
    ax2.set_ylabel("twin − actual")
    ax2.grid(alpha=0.3)
    fill = [ax2.fill_between([], [])]

    def update(k):
        n = idx[k]
        c, p, t = cycles[: n + 1], preds[: n + 1], trues[: n + 1]
        line_true.set_data(c, t)
        line_pred.set_data(c, p)

        fill[0].remove()
        fill[0] = ax2.fill_between(c, p - t, 0, color="tab:orange", alpha=0.6)

        alerting = alert_at is not None and cycles[n] >= alert_at
        banner.set_text(
            f"cycle {cycles[n]}   twin {p[-1]:.1f}   actual {t[-1]:.0f}"
            + ("   ⚠ MAINTENANCE" if alerting else "   ✓ nominal")
        )
        banner.set_color("tab:red" if alerting else "tab:green")
        return line_true, line_pred, banner, fill[0]

    DOCS_DIR.mkdir(exist_ok=True)
    out = DOCS_DIR / "demo.gif"
    anim = FuncAnimation(fig, update, frames=len(idx), blit=False)
    # Hold the final frame so the alert state is readable before the loop restarts.
    anim.save(out, writer=PillowWriter(fps=fps), dpi=80, savefig_kwargs={"facecolor": "white"})
    plt.close(fig)

    print(
        f"engine {unit}: {len(cycles)} cycles, "
        + (f"alert at {alert_at}" if alert_at else "no alert")
    )
    print(f"Saved -> {out}  ({out.stat().st_size / 1024:.0f} KB, {len(idx)} frames)")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--unit", type=int, default=1)
    p.add_argument("--subset", default="FD001")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--fps", type=int, default=12)
    a = p.parse_args()
    build(a.unit, a.subset, a.threshold, a.frames, a.fps)
