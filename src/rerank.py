"""Does the sweep's chosen configuration survive re-seeding?

    python -m src.rerank --arch cnn --top 3 --seeds 5

Writes outputs/rerank_<arch>.md, outputs/rerank_<arch>.json

**Why this exists.** `src/sweep.py` ranks configurations on a single seed, and `src/variance.py`
showed the across-seed RMSE spread is 1.7–5.7. Comparing those two numbers is uncomfortable:

| arch | top-3 val RMSE spread | across-seed spread |
|---|---|---|
| LSTM | 1.412 | 1.723 |
| GRU | 1.101 | 1.767 |
| CNN | **0.194** | 2.235 |

Every architecture's finalists sit within its own re-run noise, and the CNN's within a *tenth*
of it. So the "own best config" each architecture was given in the comparison may itself be a
lucky draw — a second-order version of the seed bias that already cost two published claims.

Rather than re-running whole grids at every seed (4 archs × 27 configs × 3 seeds is hours), this
re-runs only the **finalists**: the top *N* configurations from an existing sweep, each at *M*
seeds, ranked by mean validation RMSE. That is enough to answer the actual question — *does the
winner change?* — at a fraction of the cost.

**Why the default is 5 seeds, not 3.** Three seeds could not settle the ranking: the across-seed
range (1.7–6.1 RMSE) swamped the gaps between finalists, so every verdict landed on "inside the
noise" and stayed there no matter how the numbers fell. Three samples cannot fix that, because
the *range* they report does not shrink with more of them. A confidence interval on the **mean**
does — as ~1/√n — so 5 seeds both narrows the interval and buys a usable t multiplier (2.776 at
df=4 against 4.303 at df=2). The cost is linear in seeds and the payoff is a verdict that can
actually come back "separated". See `src/ci.py` for the interval maths and its caveats.

Reads the sweep JSON, so `python -m src.sweep --arch <a>` must have run first.
"""

from __future__ import annotations

import argparse
import json

from .ci import Interval, as_dict, mean_ci95, overlaps, t_crit_95
from .data_loader import using_real_data
from .paths import OUTPUTS_DIR


def load_finalists(arch: str, top: int) -> list[dict]:
    """Top ``top`` configurations from ``outputs/sweep_<arch>.json`` by validation RMSE."""
    path = OUTPUTS_DIR / f"sweep_{arch}.json"
    if not path.exists():
        raise SystemExit(f"no sweep results at {path}. Run: python -m src.sweep --arch {arch}")
    rows = json.loads(path.read_text())
    if not rows:
        raise SystemExit(f"{path} is empty")
    rows.sort(key=lambda r: r["val_rmse"])
    return rows[: max(1, top)]


def summarise(
    rank: int,
    val_single: float,
    key: tuple[int, int, float],
    vals: list[float],
    tests: list[float],
    phms: list[float],
    seeds: list[int],
) -> dict:
    """One finalist's row: the config, its per-seed samples, and intervals over them.

    Separated from ``main`` so the output *shape* is testable without training anything —
    the JSON is consumed by `src/compare.py` and `src/validate_docs.py`, and a silently
    renamed key there breaks the comparison table rather than this script.
    """
    return {
        "original_rank": rank,
        "seq_len": key[0],
        "hidden": key[1],
        "lr": key[2],
        "val_single": val_single,
        **as_dict(mean_ci95(vals), "val"),
        **as_dict(mean_ci95(tests), "test"),
        **as_dict(mean_ci95(phms), "phm", dp=1),
        "n_seeds": len(seeds),
        "seeds": seeds,
        # Per-seed values, not just their summary. Every earlier version stored mean and range
        # only, which made the 3->5 seed extension a full re-run: you cannot recover samples
        # from a range, so nothing could be reused and no other interval could be computed
        # after the fact. Storing the samples makes this file re-analysable without a GPU.
        "val_by_seed": [round(v, 3) for v in vals],
        "test_by_seed": [round(v, 3) for v in tests],
        "phm_by_seed": [round(v, 1) for v in phms],
    }


def main():
    from .seq_models import ARCH_NAMES

    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="cnn", choices=ARCH_NAMES)
    p.add_argument("--subset", default="FD001")
    p.add_argument("--top", type=int, default=3, help="how many sweep finalists to re-run")
    p.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="seeds per finalist; the 95%% CI on the mean tightens as ~1/sqrt(n), so 3 "
        "rarely separates finalists and 5-7 sometimes does (default 5)",
    )
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    a = p.parse_args()

    from .train_seq import train

    finalists = load_finalists(a.arch, a.top)
    seeds = [a.base_seed + i for i in range(a.seeds)]
    original_winner = (
        int(finalists[0]["seq_len"]),
        int(finalists[0]["hidden"]),
        float(finalists[0]["lr"]),
    )

    print(f"re-ranking {len(finalists)} finalists for {a.arch} over seeds {seeds}")
    print(
        f"single-seed winner: seq={original_winner[0]} hid={original_winner[1]} "
        f"lr={original_winner[2]:g}"
    )
    print()

    results = []
    for rank, cfg in enumerate(finalists, start=1):
        key = (int(cfg["seq_len"]), int(cfg["hidden"]), float(cfg["lr"]))
        vals, tests, phms = [], [], []
        for seed in seeds:
            _, res, _ = train(
                arch=a.arch,
                subset=a.subset,
                seq_len=key[0],
                hidden=key[1],
                lr=key[2],
                epochs=a.epochs,
                patience=a.patience,
                seed=seed,
                quiet=True,
            )
            vals.append(res["val_rmse"])
            tests.append(res["rmse"])
            phms.append(res["phm"])

        row = summarise(rank, cfg["val_rmse"], key, vals, tests, phms, seeds)
        results.append(row)
        val_iv = _iv(row, "val")
        ci = f"±{val_iv.half:.3f}" if val_iv.half is not None else "n/a"
        print(
            f"  rank{rank}  seq={key[0]:>3} hid={key[1]:>4} lr={key[2]:<7g}  "
            f"val single={cfg['val_rmse']:7.3f} -> mean={row['val_mean']:7.3f} {ci:>8s} "
            f"(range {row['val_range']:.3f})   test mean={row['test_mean']:7.3f}",
            flush=True,
        )

    # Persist before rendering the report — see src/interpret.py for why.
    (OUTPUTS_DIR / f"rerank_{a.arch}.json").write_text(json.dumps(results, indent=2))

    reranked = sorted(results, key=lambda r: r["val_mean"])
    new_winner = (reranked[0]["seq_len"], reranked[0]["hidden"], reranked[0]["lr"])
    held = new_winner == original_winner

    ties = [r for r in reranked[1:] if overlaps(_iv(reranked[0], "val"), _iv(r, "val"))]
    lines = _report(a, results, reranked, original_winner, new_winner, held, seeds, ties)
    (OUTPUTS_DIR / f"rerank_{a.arch}.md").write_text("\n".join(lines))
    print()
    print(lines[lines.index("## Verdict") + 2])
    print(f"Saved -> outputs/rerank_{a.arch}.md")


def _iv(row: dict, prefix: str) -> Interval:
    """Rebuild the stored interval for ``prefix`` from a result row."""
    return Interval(
        n=int(row.get("n_seeds") or len(row["seeds"])),
        mean=row[f"{prefix}_mean"],
        sd=row.get(f"{prefix}_sd"),
        half=row.get(f"{prefix}_ci95"),
        lo=row.get(f"{prefix}_lo"),
        hi=row.get(f"{prefix}_hi"),
        range=row[f"{prefix}_range"],
    )


def _cfg(r: dict) -> str:
    return f"seq={r['seq_len']}, hidden={r['hidden']}, lr={r['lr']:g}"


def _report(a, results, reranked, original_winner, new_winner, held, seeds, ties) -> list[str]:
    max_range = max(r["val_range"] for r in results)
    max_ci = max(r["val_ci95"] for r in results if r["val_ci95"] is not None)
    finalist_gap = max(r["val_mean"] for r in results) - min(r["val_mean"] for r in results)
    best, n = reranked[0], len(seeds)
    tie_list = "; ".join(_cfg(r) for r in ties)

    if held and not ties:
        verdict = (
            f"**The selection holds, and is separable at 95% CI.** Re-ranking on mean validation "
            f"RMSE keeps the same winner ({_cfg(best)}), and its interval "
            f"[{best['val_lo']:.3f}, {best['val_hi']:.3f}] over {n} seeds overlaps no other "
            f"finalist's. The gap between finalists ({finalist_gap:.3f}) is larger than the "
            f"widest interval half-width ({max_ci:.3f}), so this is a real ordering rather than "
            "a lucky draw — the first verdict in this project that survives a stated error bar."
        )
    elif held:
        verdict = (
            f"**The winner survives, but {len(ties)} of {len(results) - 1} rivals are "
            f"indistinguishable from it at 95% CI.** The same configuration ({_cfg(best)}) still "
            f"ranks first on the mean over {n} seeds, but its interval "
            f"[{best['val_lo']:.3f}, {best['val_hi']:.3f}] overlaps: {tie_list}. Finalists are "
            f"separated by {finalist_gap:.3f} validation RMSE against interval half-widths of up "
            f"to {max_ci:.3f} (raw across-seed range up to {max_range:.3f}). Read this as 'the "
            "evidence does not separate these configurations', not as 'the sweep picked the best "
            "one'."
        )
    else:
        verdict = (
            f"**The selection does not survive re-seeding.** The single-seed sweep chose "
            f"seq={original_winner[0]}, hidden={original_winner[1]}, lr={original_winner[2]:g}; "
            f"averaged over {n} seeds the best configuration is instead {_cfg(best)}"
            + (
                f", though {len(ties)} finalist(s) remain indistinguishable from it at 95% CI "
                f"({tie_list})"
                if ties
                else ", and its 95% CI overlaps no other finalist's"
            )
            + f". Finalists differ by {finalist_gap:.3f} validation RMSE against interval "
            f"half-widths of up to {max_ci:.3f}, so the original single-seed ranking was reading "
            "noise. Any downstream result that used the single-seed winner inherits that "
            "arbitrariness."
        )

    header = (
        "| original rank | seq_len | hidden | lr | val (1 seed) | val mean ±95% CI "
        "| val 95% CI | val range | test mean ±95% CI | PHM mean | vs. winner |"
    )

    def row_cells(r):
        iv = _iv(r, "val")
        if r is best:
            sep = "— (winner)"
        elif overlaps(_iv(best, "val"), iv):
            sep = "**tied** (CI overlaps)"
        else:
            sep = "separated"
        lo, hi = r["val_lo"], r["val_hi"]
        span = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "n/a"
        return (
            f"| {r['original_rank']} | {r['seq_len']} | {r['hidden']} | {r['lr']:g} | "
            f"{r['val_single']} | **{iv.fmt()}** | {span} | {r['val_range']} | "
            f"{_iv(r, 'test').fmt()} | {r['phm_mean']} | {sep} |"
        )

    return [
        f"# Sweep finalists re-ranked across seeds — {a.arch.upper()} ({a.subset})",
        "",
        f"**Data: {'real NASA C-MAPSS' if using_real_data() else 'SYNTHETIC (plumbing only)'}**",
        "",
        f"Top {len(results)} configurations from `sweep_{a.arch}.json`, each re-run at "
        f"{n} seeds {seeds}. Ranked by **mean** validation RMSE, with a Student-t 95% "
        f"confidence interval on that mean (t={t_crit_95(n - 1):g} at df={n - 1}).",
        "",
        header,
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *[row_cells(r) for r in results],
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "### Reading the overlap column",
        "",
        "Overlap is a **conservative** test. Non-overlapping 95% intervals do imply a "
        "significant difference; overlapping ones do **not** prove the configurations are "
        "equivalent — two means can overlap and still differ at p<0.05. `separated` is therefore "
        "a stronger claim than `tied` is, and `tied` should be read as *unresolved by "
        f"{n} seeds*, not as *proven equal*.",
        "",
        "Why this check exists: `src/sweep.py` ranks on one seed, and `src/variance.py` measured "
        "across-seed RMSE spreads of 1.7–6.1 — larger than the gaps between these finalists. "
        "Selecting a configuration on a single seed is the same mistake as reporting a metric on "
        "one, one level up. The interval, unlike that raw spread, tightens as seeds are added, "
        "which is what makes a `separated` verdict reachable at all.",
        "",
    ]


if __name__ == "__main__":
    main()
