"""Where would this project sit against the published C-MAPSS FD001 literature?

    python -m src.compare_published                 # attention, the model the brief names
    python -m src.compare_published --model GRU     # the model that is actually best here
    python -m src.compare_published --all-models    # every row of comparison.json

Writes outputs/published_comparison.md, outputs/published_comparison.json

**Why this exists.** Every metric this project reports is stamped "synthetic data", which is
honest but leaves the reader with no scale: is 8.2 RMSE good? The literature answers that —
but only if the comparison is set up so the answer means something. Setting it up is most of
the work here, and the setup is where the findings are.

**The baselines below are hardcoded from memory, not scraped.** That is a deliberate constraint
of the brief, and it is a real limitation: each row carries a ``confidence`` field, and the
``medium`` ones must be re-read from the paper before being quoted anywhere else. No number in
this module was fetched, and none of it should be treated as verified.

**Three things make the naive comparison invalid**, and the report says so rather than
publishing a table that implies otherwise:

1. **The PHM score is a sum, not a mean** (see ``src/evaluate.py``). Real FD001 has 100 test
   engines; the synthetic fallback has 50. A PHM score computed here is therefore roughly
   *half* a published one before any modelling difference is considered. The report normalises
   per engine and prints both columns.
2. **The test sets are not the same data.** The synthetic generator emits smooth monotonic
   drift with no fault modes, no operating-condition switching and no sensor pathology. It is
   a strictly easier regression problem than FD001, so a favourable RMSE here is the expected
   result, not an achievement.
3. **Parameter counts are almost never reported** in the RUL literature. The brief asks for a
   parameter-count delta; what can honestly be given is this project's counts against a column
   of "not reported", which is itself the finding.

**On the model this leads with.** The brief names the attention model as "your best". It owns
the best *single-seed* number in the project but not the best seed-averaged one — the GRU does.
Since preferring seed-averaged over single-seed selection is the correction this project has
already made twice (`src/variance.py`, `src/rerank.py`), the report leads with attention as
asked and prints the GRU beside it rather than quietly substituting one for the other.

Intervals come from `src/ci.py` — the same Student-t machinery `src/compare.py` uses, so the
number quoted against the literature is the number the comparison table already reports.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ci import Interval, mean_ci95, t_crit_95
from .data_loader import using_real_data
from .paths import OUTPUTS_DIR

# ---------------------------------------------------------------------------
# published baselines
#
# All rows are FD001 under the protocol this project also uses: train on the run-to-failure
# engines, score each *test* engine at its last recorded cycle, piecewise-linear RUL target
# capped at 125 (a few of these papers cap at 130 — flagged per row where it matters).
#
# References, cited by memory as the brief requires — nothing here was fetched:
#
#   [1] A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft
#       Engine Run-to-Failure Simulation", IEEE Int. Conf. on Prognostics and Health
#       Management (PHM), 2008.  The C-MAPSS datasets and the asymmetric score come from
#       here. It is a *dataset* paper: it does not report an FD001 last-cycle RMSE under the
#       protocol everyone later adopted, so it anchors the score definition, not a baseline.
#   [2] G. S. Babu, P. Zhao, X.-L. Li, "Deep Convolutional Neural Network Based Regression
#       Approach for Estimation of Remaining Useful Life", DASFAA, 2016.
#   [3] C. Zhang, P. Lim, A. K. Qin, K. C. Tan, "Multiobjective Deep Belief Networks Ensemble
#       for Remaining Useful Life Estimation in Prognostics", IEEE TNNLS, 2016.  The RF and
#       SVR rows below are from this paper's comparison table, not from separate papers.
#   [4] S. Zheng, K. Ristovski, A. Farahat, C. Gupta, "Long Short-Term Memory Network for
#       Remaining Useful Life Estimation", IEEE ICPHM, 2017.
#   [5] X. Li, Q. Ding, J.-Q. Sun, "Remaining Useful Life Estimation in Prognostics Using
#       Deep Convolution Neural Networks", Reliability Engineering & System Safety, 2018.
#   [6] J. Li, X. Li, D. He, "A Directed Acyclic Graph Network Combined with CNN and LSTM for
#       Remaining Useful Life Prediction", IEEE Access, 2019.
#   [7] H. Liu, Z. Liu, W. Jia, X. Lin, "Remaining Useful Life Prediction Using a Novel
#       Feature-Attention-Based End-to-End Approach" (AGCNN), IEEE Trans. Industrial
#       Informatics, 2020.
# ---------------------------------------------------------------------------

#: Test engines in the real FD001 test split. The published PHM scores below are sums over
#: this many engines; the synthetic fallback has 50, which is the whole normalisation problem.
FD001_TEST_ENGINES = 100


@dataclass(frozen=True)
class Published:
    """One published FD001 result, as recalled — never as verified.

    ``confidence`` is not decoration. ``high`` means the pair of numbers is one I can state
    without hedging; ``medium`` means the method and the rough magnitude are right but the
    digits should be re-read from the paper before being quoted. Nothing here is ``verified``,
    because verification requires the paper and this module is not allowed to fetch it.
    """

    label: str
    year: int
    family: str  # "classical" | "early-deep" | "modern-deep" | "reference"
    rmse: float | None
    phm: float | None
    ref: str
    confidence: str
    note: str = ""
    n_params: int | None = None  # essentially never reported; see module docstring
    n_test_engines: int = FD001_TEST_ENGINES

    @property
    def phm_per_engine(self) -> float | None:
        if self.phm is None:
            return None
        return round(self.phm / self.n_test_engines, 3)


PUBLISHED: list[Published] = [
    Published(
        label="Saxena et al. (C-MAPSS + score definition)",
        year=2008,
        family="reference",
        rmse=None,
        phm=None,
        ref="[1]",
        confidence="high",
        note=(
            "Dataset and asymmetric-score paper. Reports no FD001 last-cycle RMSE under the "
            "protocol later work adopted, so it cannot supply a comparable row"
        ),
    ),
    Published(
        label="SVR",
        year=2016,
        family="classical",
        rmse=20.96,
        phm=1381.5,
        ref="[3]",
        confidence="medium",
        note="Baseline in the MODBNE comparison table, not a standalone paper",
    ),
    Published(
        label="Random Forest",
        year=2016,
        family="classical",
        rmse=17.91,
        phm=479.75,
        ref="[3]",
        confidence="medium",
        note=(
            "Baseline in the MODBNE comparison table — the closest published analogue to "
            "this project's own forest"
        ),
    ),
    Published(
        label="Deep CNN (Babu et al.)",
        year=2016,
        family="early-deep",
        rmse=18.45,
        phm=1286.7,
        ref="[2]",
        confidence="high",
        note="First deep-learning result on C-MAPSS; decent RMSE for its time, very poor score",
    ),
    Published(
        label="MODBNE (deep belief net ensemble)",
        year=2016,
        family="early-deep",
        rmse=15.04,
        phm=334.23,
        ref="[3]",
        confidence="high",
    ),
    Published(
        label="Deep LSTM (Zheng et al.)",
        year=2017,
        family="early-deep",
        rmse=16.14,
        phm=338.0,
        ref="[4]",
        confidence="high",
        note="The canonical LSTM-on-C-MAPSS reference the brief asks for",
    ),
    Published(
        label="DCNN, time-window input (Li et al.)",
        year=2018,
        family="modern-deep",
        rmse=12.61,
        phm=273.7,
        ref="[5]",
        confidence="high",
        note="RUL cap 125, window 30 — the protocol closest to this project's",
    ),
    Published(
        label="DAG network (CNN + LSTM)",
        year=2019,
        family="modern-deep",
        rmse=11.96,
        phm=229.0,
        ref="[6]",
        confidence="medium",
    ),
    Published(
        label="AGCNN (feature attention)",
        year=2020,
        family="modern-deep",
        rmse=12.42,
        phm=225.5,
        ref="[7]",
        confidence="medium",
        note=(
            "Attention-based, so the nearest architectural analogue to this project's "
            "attention model"
        ),
    ),
]

#: What the brief supplied, kept verbatim so the divergence from the rows above is visible
#: rather than silently overwritten. All three are optimistic; §6 of the report says by how
#: much, and why the PHM one is the more serious error.
BRIEF_ESTIMATES: dict[str, dict[str, str]] = {
    "Saxena et al. (original 2008)": {
        "rmse": "~18.5",
        "phm": "~200",
        "assessment": (
            "**no such row exists.** [1] defines the dataset and the score; it reports no "
            "FD001 last-cycle RMSE. ~18.5 matches Babu et al. 2016 (18.45) closely enough "
            "that the figure is probably that one, misattributed"
        ),
    },
    "Recent deep learning (e.g. LSTM)": {
        "rmse": "~8–12",
        "phm": "~50–80",
        "assessment": (
            "RMSE optimistic — best published FD001 is ~11.9–12.6, so 8 is below anything I "
            "can cite. **PHM off by roughly 4x**: real values are ~225–340"
        ),
    },
    "Random Forest / XGBoost typical": {
        "rmse": "~10–14",
        "phm": "~60–100",
        "assessment": (
            "optimistic — the published RF row is 17.91 / 479.75 [3]. A tree ensemble at "
            "10–14 RMSE on FD001 would be near state of the art"
        ),
    },
}


def scored() -> list[Published]:
    """Baselines carrying comparable numbers — everything but the reference-only row."""
    return [e for e in PUBLISHED if e.rmse is not None and e.phm is not None]


def best_published() -> Published:
    """Lowest published RMSE on record here. The bar the project is measured against."""
    return min(scored(), key=lambda e: e.rmse)  # type: ignore[arg-type,return-value]


# ---------------------------------------------------------------------------
# this project's side of the table
# ---------------------------------------------------------------------------


@dataclass
class Ours:
    """One of this project's models, aggregated over the seeds in ``comparison.json``."""

    model: str
    subset: str
    rmse: Interval
    phm: Interval
    n_params: int | None
    n_test_engines: int
    seeds: list[int] = field(default_factory=list)

    @property
    def phm_per_engine(self) -> float:
        return round(self.phm.mean / self.n_test_engines, 3)

    def span(self, metric: str = "rmse", dp: int = 3) -> str:
        iv = self.rmse if metric == "rmse" else self.phm
        if iv.lo is None or iv.hi is None:
            return "n/a (1 seed)"
        return f"[{iv.lo:.{dp}f}, {iv.hi:.{dp}f}]"


def load_comparison(path: Path | None = None) -> list[dict]:
    src = path or (OUTPUTS_DIR / "comparison.json")
    if not src.exists():
        raise SystemExit(f"no comparison results at {src}. Run: python -m src.compare")
    rows = json.loads(src.read_text())
    if not rows:
        raise SystemExit(f"{src} is empty")
    return rows


def collect(rows: list[dict], model: str, subset: str, n_test_engines: int) -> Ours:
    """Aggregate every seed of one (model, subset) pair from ``comparison.json``."""
    matching = [r for r in rows if r["model"].lower() == model.lower() and r["subset"] == subset]
    if not matching:
        available = sorted({r["model"] for r in rows})
        raise SystemExit(
            f"comparison.json has no rows for model={model} subset={subset}; "
            f"available models: {', '.join(available)}"
        )
    params = matching[0].get("n_params")
    return Ours(
        model=matching[0]["model"],
        subset=subset,
        rmse=mean_ci95([float(r["rmse"]) for r in matching]),
        phm=mean_ci95([float(r["phm"]) for r in matching]),
        n_params=int(params) if params is not None else None,
        n_test_engines=n_test_engines,
        seeds=sorted(int(r["seed"]) for r in matching),
    )


def test_engine_count(default: int = 50) -> int:
    """How many engines the PHM sum was taken over — the divisor the comparison needs.

    Read from ``metrics.json`` when the baseline has been run, because the whole point of the
    normalisation is that this number is *not* the published 100, and assuming either value
    silently would defeat it.
    """
    path = OUTPUTS_DIR / "metrics.json"
    if path.exists():
        try:
            return int(json.loads(path.read_text())["n_test_engines"])
        except (KeyError, ValueError, TypeError):
            pass
    return default


# ---------------------------------------------------------------------------
# deltas
# ---------------------------------------------------------------------------


def pct_gap(published: float, ours: float) -> float:
    """Percent by which ``ours`` improves on ``published``; negative means worse."""
    return round((published - ours) / published * 100.0, 1)


def deltas(ours: Ours) -> list[dict]:
    """Per-baseline gap, reported raw *and* per-engine.

    Both columns are given because they disagree in magnitude and only one of them is
    defensible. The raw PHM gap flatters this project by roughly 2x purely because its test
    set is half the size of FD001's; the per-engine column removes that and leaves whatever is
    actually attributable to the model and the data.

    ``beaten_at_ci_upper`` is the stricter question the point estimates cannot answer: does the
    baseline still lose to the *worst* end of our confidence interval? On a handful of seeds
    that is a materially different verdict, and it is the one worth quoting.
    """
    out = []
    for entry in scored():
        assert entry.rmse is not None and entry.phm is not None
        assert entry.phm_per_engine is not None
        out.append(
            {
                "label": entry.label,
                "year": entry.year,
                "family": entry.family,
                "ref": entry.ref,
                "confidence": entry.confidence,
                "pub_rmse": entry.rmse,
                "pub_phm": entry.phm,
                "pub_phm_per_engine": entry.phm_per_engine,
                "our_rmse": round(ours.rmse.mean, 3),
                "our_phm": round(ours.phm.mean, 1),
                "our_phm_per_engine": ours.phm_per_engine,
                "our_rmse_ci95": [
                    None if ours.rmse.lo is None else round(ours.rmse.lo, 3),
                    None if ours.rmse.hi is None else round(ours.rmse.hi, 3),
                ],
                "rmse_gap_pct": pct_gap(entry.rmse, ours.rmse.mean),
                "phm_gap_raw": round(entry.phm - ours.phm.mean, 1),
                "phm_gap_pct_per_engine": pct_gap(entry.phm_per_engine, ours.phm_per_engine),
                "beaten_at_ci_upper": (
                    None if ours.rmse.hi is None else bool(entry.rmse > ours.rmse.hi)
                ),
            }
        )
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *["| " + " | ".join(r) + " |" for r in rows],
    ]


def render_baselines() -> list[str]:
    rows = []
    for e in PUBLISHED:
        rows.append(
            [
                f"{e.label} {e.ref}",
                str(e.year),
                e.family,
                "—" if e.rmse is None else f"{e.rmse:.2f}",
                "—" if e.phm is None else f"{e.phm:.1f}",
                "—" if e.phm_per_engine is None else f"{e.phm_per_engine:.2f}",
                "not reported" if e.n_params is None else f"{e.n_params:,}",
                e.confidence,
            ]
        )
    return _table(
        ["source", "year", "family", "RMSE ↓", "PHM ↓", "PHM/engine ↓", "params", "confidence"],
        rows,
    )


def render_side_by_side(ours: list[Ours]) -> list[str]:
    rows = []
    for e in scored():
        assert e.rmse is not None and e.phm is not None and e.phm_per_engine is not None
        rows.append(
            [
                f"{e.label} {e.ref}",
                f"{e.rmse:.2f}",
                "—",
                f"{e.phm:.1f}",
                f"{e.phm_per_engine:.2f}",
                str(e.n_test_engines),
            ]
        )
    for o in ours:
        rows.append(
            [
                f"**this project — {o.model}** (seed-avg)",
                f"{o.rmse.mean:.2f}",
                o.span("rmse", 2),
                f"{o.phm.mean:.1f}",
                f"{o.phm_per_engine:.2f}",
                str(o.n_test_engines),
            ]
        )
    return _table(["model", "RMSE ↓", "RMSE 95% CI", "PHM ↓", "PHM/engine ↓", "test engines"], rows)


def render_deltas(rows: list[dict]) -> list[str]:
    body = [
        [
            f"{d['label']} {d['ref']}",
            f"{d['pub_rmse']:.2f}",
            f"{d['rmse_gap_pct']:+.1f}%",
            f"{d['phm_gap_raw']:+.1f}",
            f"{d['phm_gap_pct_per_engine']:+.1f}%",
            {True: "yes", False: "**no**", None: "n/a"}[d["beaten_at_ci_upper"]],
        ]
        for d in rows
    ]
    return _table(
        [
            "baseline",
            "their RMSE",
            "RMSE gap",
            "PHM gap (raw)",
            "PHM gap (per engine)",
            "still beaten at our CI upper bound?",
        ],
        body,
    )


def _ci_paragraph(ours: list[Ours], delta_rows: list[dict]) -> list[str]:
    """The finding the point estimates hide: how much of the table the interval swallows.

    Written as its own paragraph because it is the one conclusion here that does not depend on
    the synthetic-data caveat and could be acted on today — the fix is seeds, not a dataset.
    """
    lead = ours[0]
    if lead.rmse.hi is None:
        return [
            "**First, there is only one seed, so there is no interval and no claim.** A single "
            "run cannot be compared to a published number at all; re-run `python -m src.compare` "
            "with more seeds before reading the table above as a ranking."
        ]

    unresolved = [d for d in delta_rows if d["beaten_at_ci_upper"] is False]
    total = len(delta_rows)
    lines = [
        f"**First, the interval swallows most of the table.** The 95% CI on this project's mean "
        f"is {lead.span('rmse', 2)} over {lead.rmse.n} seeds — wide enough that "
        f"**{len(unresolved)} of the {total} published baselines are not beaten at its upper "
        f"end**, including every result from 2017 onward. The point estimate says "
        f"{lead.rmse.mean:.2f}; the evidence says somewhere between {lead.rmse.lo:.2f} and "
        f"{lead.rmse.hi:.2f}, and a good part of that span is ordinary territory for a 2017 "
        "paper. This is a compute problem, not a data problem, and it is fixable today: more "
        "seeds tighten the interval as ~1/√n."
    ]

    contenders = [
        o
        for o in ours[1:]
        if o.rmse.hi is not None and o.rmse.hi < min(d["pub_rmse"] for d in delta_rows)
    ]
    if contenders and lead.rmse.hi >= min(d["pub_rmse"] for d in delta_rows):
        names = ", ".join(f"{o.model} {o.span('rmse', 2)}" for o in contenders)
        lines += [
            "",
            f"Worth noting which model this bites. The brief names the attention model as the "
            f"project's best, and it is the one whose interval fails — while {names} clears "
            "every published row in the table at its upper bound. That is the same pattern "
            "`src/variance.py` already found: attention owns the best single run and the worst "
            "stability, and picking a headline model on a single seed picks the wrong one.",
        ]
    return lines


def render(ours: list[Ours], delta_rows: list[dict], real: bool) -> list[str]:
    """The full standalone report."""
    lead = ours[0]
    best = best_published()
    assert best.rmse is not None and best.phm is not None and best.phm_per_engine is not None
    n = lead.rmse.n

    lines = [
        "# This project vs published C-MAPSS FD001 results",
        "",
        f"**Data: {'real NASA C-MAPSS' if real else 'SYNTHETIC fallback'}** — our side of every "
        f"table below comes from `outputs/comparison.json`, seeds {lead.seeds}.",
        "",
    ]
    if not real:
        lines += [
            "> **This comparison does not establish that the models here are competitive.**",
            "> The published rows are real FD001. Our rows are a synthetic generator emitting",
            "> smooth monotonic drift with no fault modes, no operating-condition switching and",
            "> no sensor pathology — a strictly easier regression problem. A favourable number",
            "> here is the *expected* consequence of an easier test set, not evidence of a",
            "> better model. This report exists to position the work and to mark exactly which",
            "> comparisons will and will not become valid once `data/CMAPSSData/` is populated.",
            "",
        ]

    lines += [
        "## 1. Published baselines",
        "",
        "FD001, scored at each test engine's last cycle, RUL capped at 125–130 depending on the",
        "paper. **Every figure is recalled from memory and none was fetched** — read §6 before",
        "quoting any of them.",
        "",
        *render_baselines(),
        "",
        "The `PHM/engine` column is the one to read. The PHM score is a **sum** over test",
        f"engines (`src/evaluate.py`), and real FD001 has {FD001_TEST_ENGINES} test engines",
        f"against this project's {lead.n_test_engines}. Comparing raw sums across differently",
        "sized test sets is a unit error, not a result.",
        "",
        "## 2. Side by side",
        "",
        *render_side_by_side(ours),
        "",
        (
            f"The interval is a two-sided 95% Student-t interval on the mean over {n} seeds "
            f"(t={t_crit_95(n - 1):g}), computed by `src/ci.py` — the same machinery "
            "`src/compare.py` uses, so this is the project's own headline number rather than a "
            "second opinion about it. It is wide because a handful of runs cannot pin a mean "
            "tightly; narrowing it is a matter of compute, not of method."
            if n > 1
            else "Only one seed is present, so no interval is defined — run "
            "`python -m src.compare` with more seeds before reading any gap below as real."
        ),
        "",
        "## 3. Deltas",
        "",
        f"Lead model: **{lead.model}**, seed-averaged over {n} seed(s).",
        "",
        *render_deltas(delta_rows),
        "",
        "`RMSE gap` is positive when this project's mean is lower. The last column is the",
        "stricter test — does the baseline still lose to the **upper** end of our confidence",
        "interval? — and it is the column that would flip first on real data.",
        "",
        "### Parameter count",
        "",
        *_table(
            ["model", "params"],
            [
                *[
                    [
                        f"this project — {o.model}",
                        "n/a (forest)" if o.n_params is None else f"{o.n_params:,}",
                    ]
                    for o in ours
                ],
                ["every published row above", "**not reported**"],
            ],
        ),
        "",
        "The brief asks for a parameter-count delta and it cannot be computed. RUL papers on",
        "C-MAPSS report RMSE and score and almost never report model size, so there is no",
        "published denominator. Worth stating plainly rather than estimating: an invented",
        "baseline count would make the number here look either efficient or bloated purely by",
        "choice of fiction.",
        "",
        "## 4. Interpretation",
        "",
        f"**On the means, RMSE is ahead of everything published** — {lead.rmse.mean:.2f} against",
        f"a best published {best.rmse:.2f} ({best.label} {best.ref}), a "
        f"{pct_gap(best.rmse, lead.rmse.mean):+.1f}% gap. It should not be believed, for two",
        "independent reasons, and they fail in different directions.",
        "",
        *_ci_paragraph(ours, delta_rows),
        "",
        "**Second, and this one would survive any number of seeds: the test set is not the same",
        "problem.** The generator's degradation is monotonic and low-noise, so late-life RUL is",
        "nearly a deterministic function of the sensor trajectory. That is not true of FD001's",
        "HPC-degradation fault modes, and no amount of compute on this data will tell us how",
        "much of the margin survives contact with them.",
        "",
        f"**On PHM the raw numbers also look ahead ({lead.phm.mean:.1f} vs {best.phm:.1f}), and",
        "most of that gap is a unit error.** Normalise for the engine count and the comparison",
        f"becomes {lead.phm_per_engine:.2f} against {best.phm_per_engine:.2f} per engine —",
        "still favourable, and still for the same reason. PHM's exponential late-prediction",
        "penalty is dominated by the worst few engines, and the synthetic test set has no",
        "genuinely hard engines to produce them. `outputs/error_analysis.md` already shows 57%",
        "of this project's PHM concentrated in six engines on data with *no* fault modes; on",
        "FD001 that tail is the thing the score is built to measure.",
        "",
        "**So the honest summary is not 'beat on RMSE, lag on PHM'.** It is: the pipeline",
        "produces numbers in a plausible range, under a protocol matching the literature's, on a",
        "test set easier than the literature's by an unknown factor, against baselines recalled",
        "rather than verified. Three of those four clauses have to be fixed before there is a",
        "comparison claim at all — and the one that is already sound (the protocol) is the one",
        "that took the most work.",
        "",
        "## 5. What would change on real FD001",
        "",
        "Written down now so it can be scored later rather than rationalised afterwards.",
        "",
        "- **RMSE rises to roughly 12–16.** FD001 carries irreducible noise this generator does",
        "  not: the true RUL of an engine at a fixed sensor state is not a point. Landing under",
        "  12 would put this project at the 2019–2020 state of the art, which a ~200k-parameter",
        "  recurrent model trained on a laptop over a few seeds should not be expected to do.",
        "- **PHM rises superlinearly, to roughly 250–450 over 100 engines.** A factor of 2 comes",
        "  from the engine count alone, and the exponential penalty means the extra hard engines",
        "  cost disproportionately more than the easy ones. This is where the gap to published",
        "  work will actually show, because the score punishes exactly the tail behaviour that",
        "  synthetic data lacks.",
        "- **The neural margin over the forest should widen — and if it does not, that is a",
        "  finding.** On smooth monotonic drift the sequence models have little temporal",
        "  structure to exploit that the rolling features do not already capture, which is the",
        "  most likely reason the RandomForest stays within ~3 RMSE of them here. Real fault",
        "  modes are where a sequence model earns its parameters.",
        "- **The attention model's instability should get worse, not better.** Its across-seed",
        "  RMSE range is already the widest in the project on easy data. Harder data with a",
        "  longer tail gives the seed more to disagree about, so the gap between its best and",
        "  typical run should grow.",
        "",
        "## 6. Data sources and caveats",
        "",
        f"**Sources.** {len(scored())} comparable baselines drawn from "
        f"{len({e.ref for e in PUBLISHED})} papers, all cited in the header comment of "
        "`src/compare_published.py`. All were written",
        "from memory under the brief's no-scraping constraint. Each row carries a `confidence`",
        "field; `medium` means the method and magnitude are right but the digits need re-reading",
        "from the paper. **No row is marked verified, because none is.**",
        "",
        "**Where the brief's supplied numbers and these disagree.**",
        "",
        *_table(
            ["brief's estimate", "RMSE", "PHM", "assessment"],
            [[k, v["rmse"], v["phm"], v["assessment"]] for k, v in BRIEF_ESTIMATES.items()],
        ),
        "",
        "The PHM divergence is the important one, and it is the same unit error the rest of this",
        "report is built to avoid: 50–100 is the range of a PHM *sum over ~50 engines*, or of a",
        "per-engine mean, but not of a published FD001 score. Had the brief's figures been",
        'hardcoded as given, this report would have concluded "competitive on RMSE, roughly',
        'level on PHM" — and been wrong on both counts.',
        "",
        "**Remaining caveats.**",
        "",
        f"- Our numbers are {n} seed(s). The intervals are wide; treat any ordering inside them",
        "  as unresolved rather than close.",
        "- Published papers differ on RUL cap (125 vs 130) and input window length. Both move",
        "  RMSE by a few tenths — negligible against the gaps here, not negligible against the",
        "  intervals.",
        "- Papers report their best run and rarely say over how many. A seed-averaged number is",
        "  being compared against what is probably a favourable single draw, which biases every",
        "  comparison in this report *against* this project. Of the biases present here, that is",
        "  the one direction that is safe.",
        "- Nothing in this file is checked by `src/validate_docs.py` against a source of truth,",
        "  because the published rows have no artifact to be checked against — they are inputs,",
        "  not outputs. The derived cells are recomputed from `outputs/comparison.json` on every",
        "  run.",
        "",
    ]
    return lines


def render_section(model: str = "ATTENTION", subset: str | None = None) -> list[str]:
    """Condensed version for embedding in ``outputs/comparison.md`` via ``--published``."""
    rows = load_comparison()
    subset = subset or rows[0]["subset"]
    lead = collect(rows, model, subset, test_engine_count())
    best = best_published()
    assert best.rmse is not None
    return [
        "## Against published FD001 results",
        "",
        *render_side_by_side([lead]),
        "",
        f"Best published RMSE is {best.rmse:.2f} ({best.label} {best.ref}); ours is "
        f"{lead.rmse.mean:.3f} (95% CI {lead.span('rmse', 2)} over {lead.rmse.n} seeds). "
        "**The gap is not a result.** The published rows are real FD001 with "
        f"{FD001_TEST_ENGINES} test engines; ours is synthetic data with "
        f"{lead.n_test_engines}, and the PHM score is a sum over engines — so the raw PHM "
        "column is off by roughly 2x on units alone. See `outputs/published_comparison.md` "
        "for the normalised comparison, the per-baseline deltas and the sourcing caveats; "
        "every published figure there is recalled from memory, not verified.",
        "",
    ]


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def build_payload(ours: list[Ours], delta_rows: list[dict], real: bool) -> dict:
    lead = ours[0]
    return {
        "subset": lead.subset,
        "real_data": real,
        "n_test_engines": lead.n_test_engines,
        "published_test_engines": FD001_TEST_ENGINES,
        "ci": {
            "kind": "two-sided 95% Student-t on the mean",
            "n": lead.rmse.n,
            "t": t_crit_95(lead.rmse.n - 1) if lead.rmse.n > 1 else None,
            "source": "src/ci.py",
        },
        "published": [asdict(e) | {"phm_per_engine": e.phm_per_engine} for e in PUBLISHED],
        "brief_estimates": BRIEF_ESTIMATES,
        "ours": [
            {
                "model": o.model,
                "subset": o.subset,
                "seeds": o.seeds,
                "n_params": o.n_params,
                "n_test_engines": o.n_test_engines,
                "rmse": asdict(o.rmse),
                "phm": asdict(o.phm),
                "phm_per_engine": o.phm_per_engine,
            }
            for o in ours
        ],
        "deltas": delta_rows,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m src.compare_published",
        description="position this project's models against published C-MAPSS FD001 results",
    )
    p.add_argument(
        "--model",
        default="ATTENTION",
        help="model from comparison.json to lead with (default: ATTENTION, as the brief names)",
    )
    p.add_argument(
        "--also",
        nargs="*",
        default=["GRU"],
        help="additional models to print beside it (default: GRU, the seed-averaged best)",
    )
    p.add_argument("--all-models", action="store_true", help="include every model in the run")
    p.add_argument("--subset", default=None)
    p.add_argument(
        "--test-engines",
        type=int,
        default=None,
        help="engines the PHM sum covers (default: read from metrics.json, else 50)",
    )
    a = p.parse_args(argv)

    rows = load_comparison()
    subset = a.subset or rows[0]["subset"]
    n_engines = a.test_engines or test_engine_count()

    wanted = [a.model]
    if a.all_models:
        seen = dict.fromkeys(r["model"] for r in rows)
        wanted += [m for m in seen if m.lower() != a.model.lower()]
    else:
        wanted += [m for m in a.also if m.lower() != a.model.lower()]
    ours = [collect(rows, m, subset, n_engines) for m in wanted]
    lead = ours[0]
    delta_rows = deltas(lead)
    real = using_real_data()

    (OUTPUTS_DIR / "published_comparison.md").write_text("\n".join(render(ours, delta_rows, real)))
    (OUTPUTS_DIR / "published_comparison.json").write_text(
        json.dumps(build_payload(ours, delta_rows, real), indent=2)
    )

    best = best_published()
    assert best.rmse is not None
    print(f"lead model: {lead.model}  seeds={lead.seeds}  data={'REAL' if real else 'SYNTHETIC'}")
    print(f"  RMSE {lead.rmse.fmt()}   95% CI {lead.span('rmse')}")
    print(f"  PHM  {lead.phm.fmt(1)}   {lead.phm_per_engine:.2f} per engine over {n_engines}")
    print(f"  best published: {best.rmse:.2f} RMSE ({best.label} {best.ref})")
    print("\nSaved -> outputs/published_comparison.md, outputs/published_comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
