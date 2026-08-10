"""How much of the model comparison is real, and how much is run-to-run noise?

    python -m src.variance --archs lstm gru --repeats 3

Writes outputs/variance.md, outputs/variance.json, outputs/variance.png

**Why this exists.** The GRU sweep and an earlier comparison run reported test RMSE 7.489
and 7.292 for what looked like the same configuration and the same seed. I initially wrote
that off as MPS nondeterminism. **That was wrong**, and checking it is what this module grew
out of: the two runs differed in *epoch budget* (60/patience 8 versus 80/patience 10), and
re-running each reproduces its number exactly — 7.489 and 7.292 respectively, with best
epochs 60 and 73. The GRU was simply still improving when the shorter run was truncated.

So the honest position is:

- **Same seed, repeated** — reproducible to the digit on this setup. Seeding Python, NumPy
  and torch is sufficient here, which is worth having measured rather than assumed; MPS and
  cuDNN offer no general bitwise guarantee, so this is a property of this workload, not a
  promise. Keeping the condition in the study is what would catch it changing.
- **Different seeds** — the real source of spread. The seed drives weight init, batch
  shuffling *and* the by-engine train/val split, so this measures how much a result depends
  on choices with no principled basis.

A difference between two architectures is only worth reporting if it exceeds the across-seed
spread. That is the question this answers, and it is a live one: the GRU's reported advantage
over the LSTM is ~4% on RMSE.
"""

from __future__ import annotations

import argparse
import json
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import load_config  # noqa: E402
from .data_loader import using_real_data  # noqa: E402
from .paths import OUTPUTS_DIR  # noqa: E402


def merge_results(path, results: dict[str, dict]) -> dict[str, dict]:
    """Write ``results`` into ``path``, preserving architectures this run did not cover.

    The file is keyed by architecture and accumulates across runs. It used to be overwritten
    wholesale, so `--archs lstm gru` followed by `--archs attention cnn` left only the second
    pair — silently discarding an hour of compute and stranding whichever documentation
    section cited the first. Architectures present in ``results`` replace their own entries;
    everything else survives.

    A corrupt existing file is reported and replaced rather than allowed to abort the run,
    since the alternative is losing the results being written right now.
    """
    merged: dict[str, dict] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            if isinstance(existing, dict):
                merged = existing
            else:
                print(f"warning: {path.name} is not an object, replacing", flush=True)
        except json.JSONDecodeError:
            print(f"warning: {path.name} unreadable, starting fresh", flush=True)

    merged.update(results)
    path.write_text(json.dumps(merged, indent=2))
    return merged


def _summarise(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 3),
        "std": round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "spread": round(max(values) - min(values), 3),
    }


def main():
    from .seq_models import ARCH_NAMES

    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="*", default=["lstm", "gru"], choices=ARCH_NAMES)
    p.add_argument("--repeats", type=int, default=3, help="runs per condition")
    p.add_argument("--seq-len", type=int, default=cfg.seq_len)
    p.add_argument("--hidden", type=int, default=cfg.hidden)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--patience", type=int, default=cfg.patience)
    p.add_argument("--base-seed", type=int, default=cfg.seed)
    a = p.parse_args()

    from .train_seq import train

    common = {
        "seq_len": a.seq_len,
        "hidden": a.hidden,
        "lr": a.lr,
        "epochs": a.epochs,
        "patience": a.patience,
        "quiet": True,
    }
    print(
        f"config: seq_len={a.seq_len} hidden={a.hidden} lr={a.lr:g} "
        f"epochs<={a.epochs} patience={a.patience}"
    )
    print(f"{a.repeats} repeats per condition, archs={a.archs}\n")

    results: dict[str, dict[str, dict]] = {}
    for arch in a.archs:
        results[arch] = {}

        # Condition 1: identical seed every time -> kernel nondeterminism, if any.
        fixed = [train(arch=arch, seed=a.base_seed, **common)[1] for _ in range(a.repeats)]
        # Condition 2: different seeds -> init + shuffling + train/val split all move.
        varied = [train(arch=arch, seed=a.base_seed + i, **common)[1] for i in range(a.repeats)]

        for label, runs in (("same_seed", fixed), ("different_seeds", varied)):
            results[arch][label] = {
                "rmse": _summarise([r["rmse"] for r in runs]),
                "phm": _summarise([r["phm"] for r in runs]),
                "runs": [{"seed": r.get("seed"), "rmse": r["rmse"], "phm": r["phm"]} for r in runs],
            }
            s = results[arch][label]
            print(
                f"  {arch:5s} {label:16s} RMSE {s['rmse']['mean']:.3f} "
                f"±{s['rmse']['std']:.3f} (spread {s['rmse']['spread']:.3f})   "
                f"PHM {s['phm']['mean']:.1f} ±{s['phm']['std']:.1f}",
                flush=True,
            )

    # Raw results persisted before plotting: a rendering fault must not discard
    # expensive computation. See src/interpret.py for the incident that motivated this.
    #
    # MERGED, not overwritten. This file is keyed by architecture and accumulates across
    # runs, because `--archs lstm gru` followed by `--archs attention cnn` used to leave only
    # the second pair — silently deleting an hour of compute and stranding whichever docs
    # section cited the first. Architectures present in this run replace their own entries;
    # everything else is preserved.
    merged = merge_results(OUTPUTS_DIR / "variance.json", results)
    if set(merged) - set(results):
        print(f"merged with existing entries: {sorted(set(merged) - set(results))}", flush=True)

    fig_path = _plot(results, a)
    lines = _report(results, a, fig_path)
    (OUTPUTS_DIR / "variance.md").write_text("\n".join(lines))
    print("\n" + lines[lines.index("## Verdict") + 2])
    print(f"Saved -> outputs/variance.md, {fig_path.name}")


def _report(results, a, fig_path) -> list[str]:
    lines = [
        "# Run-to-run variance",
        "",
        f"**Data: {'real NASA C-MAPSS' if using_real_data() else 'SYNTHETIC (plumbing only)'}**",
        "",
        f"Config held fixed at `seq_len={a.seq_len}, hidden={a.hidden}, lr={a.lr:g}`; "
        f"{a.repeats} runs per condition.",
        "",
        "| arch | condition | RMSE mean | RMSE std | RMSE spread | PHM mean | PHM std |",
        "|---|---|---|---|---|---|---|",
    ]
    for arch, conds in results.items():
        for label, s in conds.items():
            lines.append(
                f"| {arch.upper()} | {label.replace('_', ' ')} | {s['rmse']['mean']} | "
                f"{s['rmse']['std']} | {s['rmse']['spread']} | {s['phm']['mean']} | "
                f"{s['phm']['std']} |"
            )
    lines += ["", "## Verdict", ""]

    archs = list(results)
    if len(archs) < 2:
        lines.append(
            "Only one architecture measured, so no between-model claim is tested here. The "
            "spread column is still the precision with which any single number should be "
            "quoted."
        )
    else:
        a1, a2 = archs[0], archs[1]
        m1 = results[a1]["different_seeds"]["rmse"]["mean"]
        m2 = results[a2]["different_seeds"]["rmse"]["mean"]
        gap = abs(m1 - m2)
        noise = max(
            results[a1]["different_seeds"]["rmse"]["spread"],
            results[a2]["different_seeds"]["rmse"]["spread"],
        )
        better = a1 if m1 < m2 else a2
        if gap > noise:
            lines.append(
                f"The {better.upper()}'s RMSE advantage ({gap:.3f}) **exceeds** the "
                f"across-seed spread ({noise:.3f}), so the ranking survives re-running and "
                "is worth reporting as a real difference."
            )
        else:
            lines.append(
                f"The gap between {a1.upper()} and {a2.upper()} ({gap:.3f} RMSE) is **smaller "
                f"than** the across-seed spread ({noise:.3f}). On this data the two are not "
                "distinguishable, and any single-run table that ranks one above the other is "
                "reporting noise. The comparison in `docs/benchmarks.md` should be read with "
                "that caveat."
            )
    # Stability is a result in its own right, not a footnote. Two architectures can share a
    # mean while one is far riskier to train, and for a maintenance model that matters more
    # than a best case you cannot reliably reproduce.
    if len(archs) >= 2:
        spreads = {a: results[a]["different_seeds"]["rmse"]["spread"] for a in archs}
        steady = min(spreads, key=lambda k: spreads[k])
        jumpy = max(spreads, key=lambda k: spreads[k])
        if spreads[jumpy] > 2 * spreads[steady]:
            ratio = spreads[jumpy] / max(spreads[steady], 1e-9)
            lines += [
                "",
                f"**{jumpy.upper()} is markedly less stable than {steady.upper()}** — "
                f"across-seed spread {spreads[jumpy]:.3f} versus {spreads[steady]:.3f}, a factor "
                f"of {ratio:.1f}. Its single-seed number is a correspondingly weaker guide to "
                "what a fresh training run will produce, which is a liability in its own right: "
                "a best case you cannot reliably reproduce is worth less than a slightly worse "
                "one you can.",
            ]

    same_spreads = [c["same_seed"]["rmse"]["spread"] for c in results.values()]
    lines += [
        "",
        (
            "Same-seed repeats came out **identical** (spread 0.0), so seeding Python, NumPy "
            "and torch is sufficient for this workload on this device — measured, not assumed. "
            "MPS and cuDNN give no general bitwise guarantee, so the condition is kept in the "
            "study to catch that changing."
            if max(same_spreads, default=0.0) == 0.0
            else (
                f"Same-seed repeats still vary (spread up to {max(same_spreads):.3f} RMSE), so "
                "seeding alone does not pin this workload down on this device — that spread is "
                "the floor on how precisely any single number can be quoted."
            )
        ),
        "",
        f"Figure: `{fig_path.name}`",
        "",
    ]
    return lines


def _plot(results, a):
    archs = list(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for ax, metric, label in ((ax1, "rmse", "test RMSE"), (ax2, "phm", "test PHM")):
        positions, ticks = [], []
        for i, arch in enumerate(archs):
            for j, cond in enumerate(("same_seed", "different_seeds")):
                pos = i * 2.5 + j
                vals = [r[metric] for r in results[arch][cond]["runs"]]
                ax.scatter(
                    [pos] * len(vals),
                    vals,
                    s=45,
                    alpha=0.8,
                    edgecolor="k",
                    color="tab:blue" if cond == "same_seed" else "tab:orange",
                )
                mean = statistics.fmean(vals)
                ax.plot([pos - 0.28, pos + 0.28], [mean, mean], "k-", lw=2)
                positions.append(pos)
                ticks.append(f"{arch}\n{cond.replace('_', ' ')}")
        ax.set_xticks(positions)
        ax.set_xticklabels(ticks, fontsize=7)
        ax.set_ylabel(label)
        ax.set_title(f"{label} across repeated runs (bar = mean)")
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("twin-turbofan — run-to-run variance at a fixed configuration")
    fig.tight_layout()
    path = OUTPUTS_DIR / "variance.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
