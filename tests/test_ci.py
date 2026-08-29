"""Tests for the confidence-interval helper and the shape of `rerank_<arch>.json`.

Two separable things are pinned down here.

**The maths.** A wrong t multiplier or a √n in the wrong place produces intervals that look
entirely plausible and quietly change which models the comparison calls "tied". So the interval
is checked against a hand-computed example, and — more importantly — against the *property* the
whole 3→5 seed change rests on: the interval on the mean must **tighten** as samples are added.
Half-range does not have that property, which is why it was replaced; a regression back to a
range-like statistic would break this test rather than pass silently.

**The output shape.** `outputs/rerank_<arch>.json` is read by `src/compare.py` (to select each
architecture's configuration) and by `src/validate_docs.py` (to check the documented numbers).
Both consume it by key. Renaming or dropping a key there does not fail here first — it fails as
a wrong comparison table two steps downstream — so the contract is asserted explicitly, including
the legacy keys that predate the CI columns.
"""

from __future__ import annotations

import json
import math
import statistics

import pytest

from src.ci import Interval, as_dict, mean_ci95, overlaps, t_crit_95
from src.paths import OUTPUTS_DIR
from src.rerank import summarise

ARCHS = ["lstm", "gru", "cnn", "attention"]

# Decimal places each metric is stored at, and the tolerance that implies when re-deriving
# one stored number from another. PHM is written to 1dp, so `phm_sd` and `phm_ci95` are each
# rounded by up to 0.05 before anyone can compare them — recomputing t·s/√n from the rounded
# sd can therefore legitimately miss the rounded interval by ~0.11. Asserting tighter than
# the stored precision tests the rounding, not the statistics.
STORED_DP = {"val": 3, "test": 3, "phm": 1}


def _tol(metric: str, n: int) -> float:
    unit = 10.0 ** -STORED_DP[metric]
    return t_crit_95(n - 1) * (unit / 2) / math.sqrt(n) + unit / 2 + 1e-9


# Keys written before the CI columns existed. src/compare.py selects on `val_mean`, and
# src/validate_docs.py reads `test_mean` and `val_range` for published claims.
LEGACY_KEYS = ["original_rank", "seq_len", "hidden", "lr", "val_single", "seeds"]
LEGACY_STATS = ["val_mean", "val_range", "test_mean", "test_range", "phm_mean"]
CI_STATS = [f"{m}_{s}" for m in ("val", "test", "phm") for s in ("sd", "ci95", "lo", "hi")]
PER_SEED = ["val_by_seed", "test_by_seed", "phm_by_seed"]


# -- the maths ---------------------------------------------------------------------------


def test_t_crit_matches_published_table():
    """Two-sided 95% values; df=4 (n=5) is the one this project now runs at."""
    assert t_crit_95(1) == pytest.approx(12.706)
    assert t_crit_95(2) == pytest.approx(4.303)
    assert t_crit_95(4) == pytest.approx(2.776)
    assert t_crit_95(6) == pytest.approx(2.447)
    # Beyond the table, fall back to the normal multiplier rather than extrapolating.
    assert t_crit_95(500) == pytest.approx(1.960)


def test_t_crit_rejects_a_single_sample():
    with pytest.raises(ValueError):
        t_crit_95(0)


def test_interval_is_t_not_normal():
    """Using 1.96 at n=5 would understate the interval by ~30% — the exact error this
    module exists to avoid, and an invisible one if never asserted."""
    v = [7.0, 7.4, 6.6, 7.2, 6.8]
    iv = mean_ci95(v)
    sd = statistics.stdev(v)
    assert iv.mean == pytest.approx(7.0)
    assert iv.sd == pytest.approx(sd)
    assert iv.half == pytest.approx(2.776 * sd / math.sqrt(5))
    assert iv.half > 1.96 * sd / math.sqrt(5)
    assert (iv.lo, iv.hi) == pytest.approx((7.0 - iv.half, 7.0 + iv.half))


def test_interval_tightens_with_more_seeds():
    """The premise of raising the seed count: same spread, more samples, narrower interval.

    The raw range does not shrink — asserted alongside, because it is precisely why the
    range was retired as the headline ±.
    """
    base = [7.0, 7.4, 6.6, 7.2, 6.8]
    three, five = mean_ci95(base[:3]), mean_ci95(base)
    assert five.half < three.half
    assert five.range >= three.range


def test_single_sample_has_no_interval():
    """±0 from one run would be the most misleading possible rendering."""
    iv = mean_ci95([5.0])
    assert (iv.sd, iv.half, iv.lo, iv.hi) == (None, None, None, None)
    assert iv.fmt() == "5.000"


def test_overlap_is_symmetric_and_conservative():
    a = mean_ci95([1.0, 1.1, 0.9, 1.05, 0.95])
    b = mean_ci95([5.0, 5.1, 4.9, 5.05, 4.95])
    c = mean_ci95([1.2, 1.0, 1.1, 0.95, 1.15])
    assert not overlaps(a, b) and not overlaps(b, a)
    assert overlaps(a, c) and overlaps(c, a)
    # One sample rules nothing out, so it must overlap everything.
    assert overlaps(mean_ci95([99.0]), a)


def test_as_dict_rounds_and_prefixes():
    d = as_dict(mean_ci95([1.0, 2.0, 3.0]), "val")
    assert set(d) == {"val_mean", "val_sd", "val_ci95", "val_lo", "val_hi", "val_range"}
    assert d["val_mean"] == 2.0
    assert d["val_range"] == 2.0
    assert d["val_lo"] == pytest.approx(d["val_mean"] - d["val_ci95"])


def test_fmt_matches_the_reports():
    assert mean_ci95([7.0, 7.4, 6.6, 7.2, 6.8]).fmt() == "7.000 ±0.393"
    assert mean_ci95([50.0, 54.0, 52.0]).fmt(1) == "52.0 ±5.0"


# -- the row contract, without training anything -----------------------------------------


def _row(n=5):
    seeds = [42 + i for i in range(n)]
    vals = [7.0, 7.4, 6.6, 7.2, 6.8][:n]
    tests = [8.0, 8.6, 7.9, 8.3, 8.1][:n]
    phms = [50.0, 58.0, 47.0, 53.0, 51.0][:n]
    return summarise(1, 7.841, (50, 128, 3e-4), vals, tests, phms, seeds)


def test_summarised_row_carries_every_consumed_key():
    row = _row()
    for key in LEGACY_KEYS + LEGACY_STATS + CI_STATS + PER_SEED + ["n_seeds"]:
        assert key in row, f"{key} missing — a downstream consumer reads it"


def test_per_seed_samples_are_kept_and_agree_with_their_summary():
    """A mean that disagrees with its own samples means the file cannot be re-analysed."""
    row = _row()
    assert row["n_seeds"] == len(row["seeds"]) == 5
    for metric in ("val", "test", "phm"):
        samples = row[f"{metric}_by_seed"]
        assert len(samples) == row["n_seeds"]
        assert row[f"{metric}_mean"] == pytest.approx(statistics.fmean(samples), abs=0.05)
        assert row[f"{metric}_range"] == pytest.approx(max(samples) - min(samples), abs=0.05)
        assert row[f"{metric}_lo"] == pytest.approx(
            row[f"{metric}_mean"] - row[f"{metric}_ci95"], abs=0.05
        )


def test_row_survives_a_single_seed():
    """`--seeds 1` must degrade to "no interval", not crash or invent one."""
    row = _row(n=1)
    assert row["val_ci95"] is None and row["val_lo"] is None
    assert row["val_by_seed"] == [7.0]


# -- the artifacts actually on disk ------------------------------------------------------


def _artifact(arch):
    path = OUTPUTS_DIR / f"rerank_{arch}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not generated; run `make rerank ARCH={arch}`")
    return json.loads(path.read_text())


@pytest.mark.parametrize("arch", ARCHS)
def test_artifact_has_one_row_per_finalist(arch):
    rows = _artifact(arch)
    assert rows, "empty rerank artifact"
    ranks = [r["original_rank"] for r in rows]
    assert ranks == sorted(ranks), "rows must stay in the sweep's original ranking order"
    assert len(set(ranks)) == len(ranks), "duplicate finalist rows"
    configs = {(r["seq_len"], r["hidden"], r["lr"]) for r in rows}
    assert len(configs) == len(rows), "the same configuration was re-run as two finalists"


@pytest.mark.parametrize("arch", ARCHS)
def test_artifact_carries_five_seed_statistics(arch):
    rows = _artifact(arch)
    for row in rows:
        for key in LEGACY_KEYS + LEGACY_STATS + CI_STATS + PER_SEED:
            assert key in row, f"{arch}: {key} missing from rerank artifact"
        assert row["n_seeds"] >= 5, f"{arch}: expected >=5 seeds, got {row['n_seeds']}"
        assert len(row["seeds"]) == len(set(row["seeds"])) == row["n_seeds"]
        for metric in ("val", "test", "phm"):
            samples = row[f"{metric}_by_seed"]
            n, tol = row["n_seeds"], _tol(metric, row["n_seeds"])
            assert len(samples) == n
            assert row[f"{metric}_mean"] == pytest.approx(statistics.fmean(samples), abs=tol)
            iv = Interval(
                n=n,
                mean=row[f"{metric}_mean"],
                sd=row[f"{metric}_sd"],
                half=row[f"{metric}_ci95"],
                lo=row[f"{metric}_lo"],
                hi=row[f"{metric}_hi"],
                range=row[f"{metric}_range"],
            )
            # The strong check: the published interval must be reproducible from the
            # published samples. If these disagree, the file is not self-consistent and
            # nobody downstream can re-derive it.
            expected = t_crit_95(n - 1) * statistics.stdev(samples) / math.sqrt(n)
            assert iv.half == pytest.approx(
                expected, abs=tol
            ), f"{arch}: {metric} interval is not a t interval over its own samples"
            assert iv.lo == pytest.approx(iv.mean - iv.half, abs=tol)
            assert iv.hi == pytest.approx(iv.mean + iv.half, abs=tol)


@pytest.mark.parametrize("arch", ARCHS)
def test_artifact_seeds_are_consistent_across_finalists(arch):
    """Finalists compared against each other must have seen the same seeds."""
    rows = _artifact(arch)
    assert len({tuple(r["seeds"]) for r in rows}) == 1
