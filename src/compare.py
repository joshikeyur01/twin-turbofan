"""Model × dataset comparison table.

Runs the RandomForest baseline and every sequence architecture over whichever
C-MAPSS subsets are present, and writes one table of RMSE + PHM per pair.

Run:
    python -m src.compare                      # all archs, all available subsets
    python -m src.compare --archs lstm cnn     # subset of models
    python -m src.compare --epochs 40          # shorter runs

Writes outputs/comparison.md, outputs/comparison.json

The report is explicitly stamped with whether it ran on real or synthetic data —
synthetic numbers validate plumbing and must not be read as benchmark results.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .ablation import run_arm
from .ci import mean_ci95, overlaps, t_crit_95
from .data_loader import available_subsets, load_cmapss, using_real_data
from .paths import OUTPUTS_DIR
from .seq_models import ARCHITECTURES

SEQ_ARCHS = list(ARCHITECTURES)


def run_rf(subset: str, seed: int = 42) -> dict:
    """RandomForest baseline with the project's default feature set."""
    train, test = load_cmapss(subset)
    res = run_arm(train, test, use_rolling=True, window=5, seed=seed)
    return {
        "model": "RandomForest",
        "subset": subset,
        "seed": seed,
        "rmse": res["rmse"],
        "phm": res["phm"],
        "pct_late": res["pct_late"],
        "n_params": None,
        "train_s": res["fit_s"],
    }


def swept_config(arch: str) -> tuple[dict, str] | None:
    """Best config for ``arch``, preferring a **seed-averaged** selection.

    Reading this per-architecture matters. The first version of this script applied one
    config to all three models, chosen by the LSTM's sweep, and the CNN looked far worse
    than it is: swept on its own grid the CNN prefers the *opposite* corner
    (seq_len=20, hidden=32 rather than 50/128). A shared config is not a fair comparison
    between architectures with different inductive biases.

    ``rerank_<arch>.json`` is preferred over ``sweep_<arch>.json`` when present, because the
    sweep ranks on a single seed and `src/rerank.py` showed that ranking is noise: for the
    GRU the seed-averaged winner is a *different* learning rate whose 3-seed test RMSE is
    6.820 against the single-seed pick's 8.312. Selecting a configuration on one seed is the
    same error as reporting a metric on one, a level up — so if the seed-averaged ranking
    exists, it wins.
    """
    rerank = OUTPUTS_DIR / f"rerank_{arch}.json"
    if rerank.exists():
        rows = json.loads(rerank.read_text())
        if rows:
            best = min(rows, key=lambda r: r["val_mean"])
            return (
                {
                    "seq_len": int(best["seq_len"]),
                    "hidden": int(best["hidden"]),
                    "lr": float(best["lr"]),
                },
                f"rerank_{arch}.json (seed-averaged)",
            )

    path = OUTPUTS_DIR / f"sweep_{arch}.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    if not rows:
        return None
    best = min(rows, key=lambda r: r["val_rmse"])
    return (
        {
            "seq_len": int(best["seq_len"]),
            "hidden": int(best["hidden"]),
            "lr": float(best["lr"]),
        },
        f"sweep_{arch}.json (single seed)",
    )


def _aggregate(df):
    """Collapse per-seed rows into `mean ±95% CI` tables.

    Reporting a single seed was the project's biggest honesty problem: the variance study
    (`src/variance.py`) found the across-seed RMSE spread reaches 1.8 for the GRU and 6.1 for
    the attention model, while the between-architecture gaps being ranked are often under 1.0.
    A single-seed table therefore ranks noise, and — because seed 42 turned out to be a
    favourable draw — reports numbers better than a fresh run would produce.

    **Why the ± is now a confidence interval, not a half-range.** The earlier table reported
    `±(max−min)/2` on the argument that three seeds do not support a standard deviation. The
    reasoning was sound but the statistic was the wrong one: half-range describes *the runs that
    happened* and, being an extremum, drifts wider as seeds are added — so it could never settle
    whether the GRU actually beats the LSTM no matter how much compute was spent. A Student-t
    interval on the **mean** answers the question that is actually being asked of this table, and
    tightens as ~1/√n. The raw range is kept as its own column, because it still says something
    the interval does not: how badly a *single* run can land.
    """
    raw = {
        (model, subset): {
            "rmse": mean_ci95(list(g["rmse"])),
            "phm": mean_ci95(list(g["phm"])),
        }
        for (model, subset), g in df.groupby(["model", "subset"])
    }

    agg = (
        df.groupby(["model", "subset"])
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_min=("rmse", "min"),
            rmse_max=("rmse", "max"),
            phm_mean=("phm", "mean"),
            phm_min=("phm", "min"),
            phm_max=("phm", "max"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    def cell(row, metric, dp):
        return raw[(row["model"], row["subset"])][metric].fmt(dp)

    def ci(row, metric, dp):
        iv = raw[(row["model"], row["subset"])][metric]
        return "n/a" if iv.lo is None else f"[{iv.lo:.{dp}f}, {iv.hi:.{dp}f}]"

    agg["rmse_str"] = agg.apply(lambda r: cell(r, "rmse", 3), axis=1)
    agg["phm_str"] = agg.apply(lambda r: cell(r, "phm", 1), axis=1)
    agg["rmse_ci95"] = agg.apply(lambda r: ci(r, "rmse", 3), axis=1)
    agg["rmse_range"] = (agg["rmse_max"] - agg["rmse_min"]).round(3)

    rmse_tbl = agg.pivot(index="model", columns="subset", values="rmse_str")
    phm_tbl = agg.pivot(index="model", columns="subset", values="phm_str")
    return rmse_tbl, phm_tbl, agg, raw


def _separation(raw, subset: str, metric: str = "rmse"):
    """Which models this many seeds can actually tell apart, on one subset.

    Returns ``(ordered_models, ties)`` where ``ties[m]`` lists the models whose 95% interval
    overlaps ``m``'s. Overlap is deliberately the *conservative* direction: see `src/ci.py`.
    A `separated` verdict here is evidence; a `tied` one is only an absence of it.
    """
    models = sorted(
        (m for (m, s) in raw if s == subset), key=lambda m: raw[(m, subset)][metric].mean
    )
    ties = {
        m: [
            o
            for o in models
            if o != m and overlaps(raw[(m, subset)][metric], raw[(o, subset)][metric])
        ]
        for m in models
    }
    return models, ties


def _separation_section(raw, subsets, seeds) -> list[str]:
    """The 'who is actually distinguishable' block — the point of raising the seed count."""
    lines = [
        "## Statistical separation (95% CI on the mean RMSE)",
        "",
        f"With **{len(seeds)} seeds** the interval is `mean ± t·s/√n` with t="
        f"{t_crit_95(len(seeds) - 1):g} (df={len(seeds) - 1}). Models whose intervals overlap "
        "are marked **indistinguishable at 95% CI**.",
        "",
    ]
    for subset in subsets:
        models, ties = _separation(raw, subset)
        if not models:
            continue
        if len(subsets) > 1:
            lines += [f"### {subset}", ""]
        lines += [
            "| rank | model | RMSE mean ±95% CI | 95% CI | across-seed range | verdict |",
            "|---|---|---|---|---|---|",
        ]
        for i, m in enumerate(models, start=1):
            iv = raw[(m, subset)]["rmse"]
            span = "n/a" if iv.lo is None else f"[{iv.lo:.3f}, {iv.hi:.3f}]"
            verdict = (
                f"**indistinguishable at 95% CI** from {', '.join(ties[m])}"
                if ties[m]
                else "separated from every other model"
            )
            lines += [f"| {i} | {m} | {iv.fmt()} | {span} | {iv.range:.3f} | {verdict} |"]

        best = models[0]
        clear = [m for m in models if not ties[m]]
        if not ties[best]:
            headline = (
                f"**{best} is a clear winner on {subset}**: its interval "
                f"{raw[(best, subset)]['rmse'].fmt()} overlaps no other model's, so its lead is "
                f"not a seed artefact."
            )
        else:
            headline = (
                f"**{best} leads on the mean but is not separated from "
                f"{', '.join(ties[best])} at 95% CI** on {subset}. On this evidence the top of "
                "the table is a tie, not a ranking."
            )
        lines += [
            "",
            headline,
            "",
            (
                f"Fully separated from everything else: {', '.join(clear)}."
                if clear
                else "No model on this subset is separated from every other."
            ),
            "",
            "Overlap is a conservative test: non-overlapping intervals do imply a significant "
            "difference, but overlapping ones do **not** prove equivalence — two means can "
            "overlap and still differ at p<0.05. Read `indistinguishable` as *unresolved by "
            f"{len(seeds)} seeds*, not as *proven equal*.",
            "",
        ]
    return lines


def run_seq(arch: str, subset: str, **kw) -> dict:
    """One sequence architecture. Imported lazily so RF-only runs need no torch."""
    from .train_seq import train

    _, res, _ = train(arch=arch, subset=subset, quiet=True, **kw)
    return {
        "model": arch.upper(),
        "subset": subset,
        "seed": res["seed"],
        "rmse": res["rmse"],
        "phm": res["phm"],
        "pct_late": res["pct_late"],
        "n_params": res["n_params"],
        "seq_len": res["seq_len"],
        "hidden": res["hidden"],
        "lr": res["lr"],
        "train_s": res["train_s"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="*", default=SEQ_ARCHS, choices=SEQ_ARCHS)
    p.add_argument("--subsets", nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=50)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--skip-rf", action="store_true")
    p.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="number of seeds to average over (1 reproduces the old single-seed table). "
        "Default raised 3->5 to match src/rerank.py: the 95%% CI on the mean scales as "
        "~1/sqrt(n), and at 3 seeds it was too wide to separate any two architectures",
    )
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument(
        "--shared-config",
        action="store_true",
        help="force one config for every arch instead of each arch's own swept best "
        "(kept only for reproducing the earlier, unfair table)",
    )
    p.add_argument(
        "--published",
        nargs="?",
        const="ATTENTION",
        default=None,
        metavar="MODEL",
        help="append a comparison against published C-MAPSS FD001 results for MODEL "
        "(default ATTENTION). Off by default: the published rows are real FD001 and these "
        "are not, so the section is opt-in rather than something a reader trips over. "
        "See src/compare_published.py and outputs/published_comparison.md",
    )
    a = p.parse_args()

    subsets = a.subsets or available_subsets()
    if not subsets:
        raise SystemExit("no C-MAPSS subsets found; see data/README.md")

    real = using_real_data()
    print(f"subsets: {subsets}  data: {'REAL' if real else 'SYNTHETIC (plumbing only)'}")

    seeds = [a.base_seed + i for i in range(a.seeds)]
    print(f"seeds: {seeds}")

    rows = []
    for subset in subsets:
        for seed in seeds:
            if not a.skip_rf:
                r = run_rf(subset, seed=seed)
                rows.append(r)
                # flush: this loop runs for tens of minutes, and without it a backgrounded
                # or piped run shows nothing at all until the process exits.
                print(
                    f"  {subset} seed={seed} RandomForest  "
                    f"rmse={r['rmse']:8.3f}  phm={r['phm']:9.1f}",
                    flush=True,
                )
            for arch in a.archs:
                hp = {"seq_len": a.seq_len, "hidden": a.hidden, "lr": a.lr}
                source = "shared (never swept)"
                if not a.shared_config:
                    swept = swept_config(arch)
                    if swept:
                        hp, source = swept
                r = run_seq(arch, subset, epochs=a.epochs, patience=a.patience, seed=seed, **hp)
                r["config_source"] = source
                rows.append(r)
                print(
                    f"  {subset} seed={seed} {r['model']:12s}  "
                    f"rmse={r['rmse']:8.3f}  phm={r['phm']:9.1f}"
                    f"   [seq={hp['seq_len']} hid={hp['hidden']} lr={hp['lr']:g}]",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    rmse_tbl, phm_tbl, per_seed, intervals = _aggregate(df)

    lines = [
        "# Model × dataset comparison",
        "",
        f"**Data: {'real NASA C-MAPSS' if real else 'SYNTHETIC fallback'}**",
        "",
    ]
    if not real:
        lines += [
            "> These numbers exercise the pipeline; they are **not** benchmark results.",
            "> The synthetic generator produces smooth monotonic drift with no real fault",
            "> modes, so absolute values and model ranking may both differ on real data.",
            "",
        ]
    lines += [
        "Protocol: split by engine, score each engine's last cycle, RUL capped at 125.",
        f"Sequence models: up to {a.epochs} epochs, early stopping patience {a.patience}, "
        "best-val checkpoint restored. Each architecture uses **its own** "
        "validation-selected hyperparameters where a sweep exists (see the "
        "`config_source` column) — a single shared config penalises architectures whose "
        "inductive bias wants a different window or capacity.",
        "",
        f"Averaged over **{len(seeds)} seeds** ({seeds}); cells are "
        "`mean ±95% CI` — a Student-t interval on the mean, not the older half-range. "
        "Single-seed numbers are not reported as headline results: the across-seed spread "
        "(see `outputs/variance.md`) is comparable to or larger than the gaps between "
        "architectures, so one seed ranks noise — and seed 42 happens to be a favourable draw. "
        "Half-range was retired as the headline ± because it describes the runs rather than "
        "the mean, and widens rather than tightens as seeds are added; it survives as the "
        "`across-seed range` column, which answers a different and still-useful question.",
        "",
        "## RMSE (lower is better)",
        "",
        rmse_tbl.to_markdown(),
        "",
        "## PHM score (lower is better; asymmetric, punishes late predictions)",
        "",
        phm_tbl.to_markdown(),
        "",
        *_separation_section(intervals, subsets, seeds),
        "## Per-seed detail",
        "",
        per_seed.round(3).to_markdown(index=False),
        "",
        "## Every run",
        "",
        df.to_markdown(index=False),
        "",
    ]
    # JSON first: the published section re-reads it, so it reports the run that just
    # finished rather than whichever one happened to be on disk beforehand.
    (OUTPUTS_DIR / "comparison.json").write_text(json.dumps(rows, indent=2))

    if a.published:
        from .compare_published import render_section

        lines += render_section(a.published, subsets[0])

    (OUTPUTS_DIR / "comparison.md").write_text("\n".join(lines))
    print("\nSaved -> outputs/comparison.md, outputs/comparison.json")


if __name__ == "__main__":
    main()
