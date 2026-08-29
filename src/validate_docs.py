"""Do the numbers in README.md and docs/benchmarks.md still match ``outputs/``?

    python -m src.validate_docs            # report drift, exit 1 if any
    python -m src.validate_docs --strict   # warnings count as failures too
    python -m src.validate_docs -v         # also list the checks that passed

**Why this exists.** Every headline figure in the docs was copied by hand out of an
artifact in ``outputs/``. Sweeps get re-run, a configuration selection changes, a model
gains parameters — and the prose keeps quoting the old number. That failure is silent:
nothing crashes, the tables still look authoritative, and the drift is only found by
someone re-deriving a number by hand. This module re-derives them all, every ``make check``.

The checks are declarative. Each entry in :data:`SPECS` says where a number lives in a doc
and how to recompute it from an artifact; running a spec yields one finding per cell that
disagrees, with ``file:line`` and expected-vs-actual.

**Three severities**, because "different" is not one thing:

- ``ok``    — the doc agrees with the artifact at the precision it prints.
- ``warn``  — off by one unit in the last printed digit. Usually a rounding path that
  changed (a mean of rounded values vs a rounded mean), not a stale number. Reported,
  but does not fail the build unless ``--strict``.
- ``error`` — off by more than that, or a non-numeric mismatch (a configuration, a
  parameter count). This is drift.

**Artifact coverage.** Anything number-bearing in ``outputs/`` that no spec reads is a
warning: a new sweep or a renamed artifact must either get checked or be excused in
:data:`UNCHECKED` with a reason. Otherwise new numbers enter the docs unguarded, which is
the exact hole this module exists to close.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ci import mean_ci95

# ---------------------------------------------------------------------------
# value types
#
# A spec's expected value is a plain Python object. Its *type* decides how the
# corresponding doc cell is parsed and compared, so specs stay declarative.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """A sequence-model configuration, however the doc happens to spell it.

    Written as ``50/128/1e-3`` in README tables and ``seq=50 hidden=128 lr=3e-4`` in
    benchmarks; both parse to the same triple so either form can be checked.
    """

    seq_len: int
    hidden: int
    lr: float

    def __str__(self) -> str:
        return f"{self.seq_len}/{self.hidden}/{self.lr:g}"


@dataclass(frozen=True)
class MeanSpread:
    """A ``mean ±uncertainty`` cell, as ``src/compare.py`` renders them.

    The uncertainty is now a 95% confidence interval on the mean; it was a half-range until
    the seed count went from 3 to 5. The parsing is identical either way — only the expected
    value changes — so this class stayed put while ``mean_ci`` replaced ``mean_spread`` as the
    thing that produces it.
    """

    mean: float
    half: float

    def __str__(self) -> str:
        return f"{self.mean} ±{self.half}"


@dataclass(frozen=True)
class Ratio:
    """A ``best / total`` cell, e.g. the sweep's ``21 / 29`` best-epoch column."""

    num: int
    den: int

    def __str__(self) -> str:
        return f"{self.num} / {self.den}"


Expected = float | int | str | Config | MeanSpread | Ratio


# ---------------------------------------------------------------------------
# row keys
#
# Identifying "the doc row this artifact row is about" is the one genuinely fiddly part.
# Prefix matching is not enough: `s2` also prefixes `s20`, `0.9` prefixes `0.95`, and the
# sweep grid writes the same learning rate as `3e-4` in the doc and `0.0003` in the JSON.
# Each key type below states exactly what identity means for one table.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Label:
    """Cell 0 equals this text once emphasis and parentheticals are stripped."""

    text: str

    def matches(self, row: Row) -> bool:
        return clean(row.cells[0]).lower() == self.text.lower()

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class NumCells:
    """The first number of each leading cell equals these, compared numerically.

    Numeric comparison is the point: the doc writes ``3e-4`` where the JSON holds
    ``0.0003``, and ``0.9`` must not match the ``0.95`` row.
    """

    values: tuple[float, ...]

    def matches(self, row: Row) -> bool:
        if len(row.cells) < len(self.values):
            return False
        for cell, want in zip(row.cells, self.values, strict=False):
            got = parse_float(clean(cell))
            if got is None or abs(got - want) > 1e-9:
                return False
        return True

    def __str__(self) -> str:
        return "/".join(f"{v:g}" for v in self.values)


@dataclass(frozen=True)
class BinUpper:
    """The *last* number in cell 0 equals this — for bins written ``25–50`` / ``(25.0, 50.0]``."""

    value: float

    def matches(self, row: Row) -> bool:
        found = numbers_in(clean(row.cells[0]))
        return bool(found) and abs(float(found[-1]) - self.value) < 1e-9

    def __str__(self) -> str:
        return f"≤{self.value:g}"


@dataclass(frozen=True)
class Tokens:
    """Every part appears as a whole token somewhere in the row.

    For tables where cell 0 alone is ambiguous — §8c keys on arch *and* condition.
    """

    parts: tuple[str, ...]

    def matches(self, row: Row) -> bool:
        haystack = " ".join(clean(c) for c in row.cells).lower()
        return all(
            re.search(rf"(?<![\w.]){re.escape(p.lower())}(?![\w.])", haystack) for p in self.parts
        )

    def __str__(self) -> str:
        return " ".join(self.parts)


Key = Label | NumCells | BinUpper | Tokens


# ---------------------------------------------------------------------------
# markdown parsing
# ---------------------------------------------------------------------------

_MINUS = "−"  # docs use a real minus sign in residual columns
_DELIM = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")


@dataclass(frozen=True)
class Row:
    cells: tuple[str, ...]
    line: int


@dataclass(frozen=True)
class Table:
    header: tuple[str, ...]
    rows: tuple[Row, ...]
    section: str
    preamble: str
    line: int

    @property
    def context(self) -> str:
        """Text a spec's ``anchor`` is matched against to identify this table."""
        return "\n".join([self.section, self.preamble, " | ".join(self.header)])


def split_row(line: str) -> list[str]:
    """Cells of a markdown table row, outer pipes dropped."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [c.strip() for c in stripped.split("|")]


def parse_tables(text: str) -> list[Table]:
    """Every pipe table in ``text``, tagged with the heading and prose above it.

    Line numbers are 1-based so findings can be pasted straight into an editor.
    """
    lines = text.splitlines()
    tables: list[Table] = []
    section = ""
    since_heading: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            section = line.strip()
            since_heading = []
            i += 1
            continue
        is_table = (
            "|" in line
            and line.strip().startswith("|")
            and i + 1 < len(lines)
            and _DELIM.match(lines[i + 1])
            and "|" in lines[i + 1]
        )
        if not is_table:
            since_heading.append(line)
            i += 1
            continue

        header = tuple(split_row(line))
        start = i
        i += 2  # header + delimiter
        rows: list[Row] = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            rows.append(Row(cells=tuple(split_row(lines[i])), line=i + 1))
            i += 1
        tables.append(
            Table(
                header=header,
                rows=tuple(rows),
                section=section,
                preamble="\n".join(since_heading),
                line=start + 1,
            )
        )
        since_heading = []
    return tables


def clean(cell: str) -> str:
    """Strip markdown emphasis, code ticks, arrows and parenthetical notes from a cell.

    ``**seq=20 hidden=32** lr=3e-4`` and ``**1.0 *(GRU only — selected)***`` are both real
    cells from these docs; the annotations are commentary, not part of the value.
    """
    out = cell
    out = re.sub(r"\*\(.*?\)\*", " ", out)  # *(shipped default)*
    out = re.sub(r"\((?:[^()]*)\)", " ", out)  # (GRU only — selected)
    out = out.replace("**", " ").replace("*", " ").replace("`", " ")
    out = out.replace("↓", " ").replace("↑", " ").replace(_MINUS, "-")
    return " ".join(out.split())


_NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def numbers_in(text: str) -> list[str]:
    """Every numeric literal in ``text``, in order, as written."""
    return _NUM.findall(text)


def parse_float(text: str) -> float | None:
    found = numbers_in(text)
    if not found:
        return None
    return float(found[0].replace(",", ""))


def decimals(literal: str) -> int:
    """Digits printed after the decimal point — the precision the doc is claiming."""
    if "e" in literal.lower():
        return 0
    _, _, frac = literal.partition(".")
    return len(frac)


def parse_config(text: str) -> Config | None:
    """``50/128/1e-3`` or ``seq=50 hidden=128 lr=3e-4`` -> :class:`Config`."""
    body = clean(text)
    kw = re.search(r"seq(?:_len)?\s*=\s*(\d+).*?hidden\s*=\s*(\d+).*?lr\s*=\s*([\d.eE+-]+)", body)
    if kw:
        return Config(int(kw.group(1)), int(kw.group(2)), float(kw.group(3)))
    slash = re.search(r"(\d+)\s*/\s*(\d+)\s*/\s*([\d.eE+-]+)", body)
    if slash:
        return Config(int(slash.group(1)), int(slash.group(2)), float(slash.group(3)))
    return None


def parse_ratio(text: str) -> Ratio | None:
    found = numbers_in(clean(text))
    if len(found) < 2:
        return None
    return Ratio(int(float(found[0])), int(float(found[1])))


def parse_mean_spread(text: str) -> MeanSpread | None:
    found = numbers_in(clean(text))
    if len(found) < 2:
        return None
    return MeanSpread(float(found[0].replace(",", "")), float(found[1].replace(",", "")))


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


class Artifacts:
    """Lazy reader for ``outputs/`` that records which files were actually consulted.

    The usage record is what powers the coverage audit: a spec cannot claim to cover an
    artifact it never opened, and an artifact nothing opened is flagged.
    """

    def __init__(self, outputs: Path) -> None:
        self.outputs = outputs
        self.used: set[str] = set()
        self._cache: dict[str, Any] = {}

    def _read(self, name: str) -> str:
        self.used.add(name)
        path = self.outputs / name
        if not path.exists():
            raise MissingArtifact(name)
        return path.read_text()

    def json(self, name: str) -> Any:
        if name not in self._cache:
            self._cache[name] = json.loads(self._read(name))
        else:
            self.used.add(name)
        return self._cache[name]

    def md_frontmatter(self, name: str) -> dict[str, Any]:
        """Header metadata of a generated report, above its first table.

        These reports carry no YAML frontmatter; the equivalent metadata is prose the
        generator writes deterministically ("27 configurations, up to 60 epochs each with
        early-stopping patience 8"), so it is parsed out of that.
        """
        key = f"frontmatter:{name}"
        if key in self._cache:
            self.used.add(name)
            return dict(self._cache[key])

        text = self._read(name)
        head = text.split("\n|", 1)[0]
        meta: dict[str, Any] = {}
        if title := re.search(r"^#\s+(.*)$", head, re.M):
            meta["title"] = title.group(1).strip()
        if data := re.search(r"\*\*Data:\s*([^*]+)\*\*", head):
            meta["data"] = data.group(1).strip()
        if n := re.search(r"(\d+)\s+configurations", head):
            meta["n_configs"] = int(n.group(1))
        if ep := re.search(r"up to (\d+) epochs", head):
            meta["epochs"] = int(ep.group(1))
        if pat := re.search(r"patience (\d+)", head):
            meta["patience"] = int(pat.group(1))
        if seeds := re.search(r"seeds \[([\d,\s]+)\]", head):
            meta["seeds"] = [int(s) for s in seeds.group(1).split(",")]
        self._cache[key] = meta
        return dict(meta)

    def md_metrics(self, name: str) -> dict[str, float]:
        """``- Label: **12.3**`` bullets from a generated report.

        ``src/error_analysis.py`` writes only markdown and PNGs — no JSON — so its
        numbers are only checkable through the report it emits.
        """
        key = f"metrics:{name}"
        if key in self._cache:
            self.used.add(name)
            return dict(self._cache[key])

        text = self._read(name)
        out: dict[str, float] = {}
        for label, value in re.findall(r"^-\s+([^:]+):\s*\*\*([^*]+)\*\*", text, re.M):
            parsed = parse_float(value.replace(_MINUS, "-"))
            if parsed is not None:
                out[label.strip()] = parsed
        self._cache[key] = out
        return dict(out)


class MissingArtifact(Exception):
    """A spec asked for an artifact that is not in ``outputs/``."""

    def __init__(self, name: str) -> None:
        super().__init__(f"missing artifact outputs/{name}")
        self.name = name


# ---------------------------------------------------------------------------
# derived quantities
#
# Everything a spec quotes is recomputed here from the raw artifact, so a spec never
# hardcodes a number: the artifact is the single source of truth.
# ---------------------------------------------------------------------------


def comparison_by_model(art: Artifacts) -> dict[str, list[dict[str, Any]]]:
    rows = art.json("comparison.json")
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row["model"], []).append(row)
    return out


def mean_spread(values: list[float]) -> MeanSpread:
    """``mean ±half-range`` — the pre-CI rendering.

    Kept because several derived claims (`_gru_rmse_gain_pct` and friends) only want ``.mean``
    and are unaffected by which uncertainty the table prints. Table cells now use ``mean_ci``.
    """
    return MeanSpread(
        round(sum(values) / len(values), 3), round((max(values) - min(values)) / 2, 3)
    )


def mean_ci(values: list[float], dp: int = 3) -> MeanSpread:
    """``src/compare.py`` renders ``mean ±95% CI``; this reproduces it exactly.

    Delegates to `src.ci` rather than re-deriving the interval: a checker that computes the
    number a second way validates its own arithmetic, not the document.
    """
    iv = mean_ci95(values)
    return MeanSpread(round(iv.mean, dp), round(iv.half, dp) if iv.half is not None else 0.0)


def spread(values: list[float]) -> float:
    return round(max(values) - min(values), 3)


def sweep_rows(art: Artifacts, arch: str) -> list[dict[str, Any]]:
    return list(art.json(f"sweep_{arch}.json"))


def sweep_best(art: Artifacts, arch: str) -> dict[str, Any]:
    """The configuration ``src/sweep.py`` selects: lowest validation RMSE."""
    return min(sweep_rows(art, arch), key=lambda r: r["val_rmse"])


def sweep_at(art: Artifacts, arch: str, cfg: Config) -> dict[str, Any]:
    for row in sweep_rows(art, arch):
        if (row["seq_len"], row["hidden"], row["lr"]) == (cfg.seq_len, cfg.hidden, cfg.lr):
            return row
    raise KeyError(f"sweep_{arch}.json has no cell {cfg}")


def rerank_best(art: Artifacts, arch: str) -> dict[str, Any]:
    """The configuration ``src/rerank.py`` selects: lowest *seed-averaged* validation RMSE."""
    return min(art.json(f"rerank_{arch}.json"), key=lambda r: r["val_mean"])


def rerank_config(art: Artifacts, arch: str) -> Config:
    best = rerank_best(art, arch)
    return Config(best["seq_len"], best["hidden"], best["lr"])


def variance_block(art: Artifacts, arch: str, condition: str) -> dict[str, Any]:
    return art.json("variance.json")[arch][condition]


def uncertainty_row(art: Artifacts, quantile: float) -> dict[str, Any]:
    for row in art.json("uncertainty.json")["quantiles"]:
        if abs(row["quantile"] - quantile) < 1e-9:
            return row
    raise KeyError(f"uncertainty.json has no quantile {quantile}")


def per_engine_row(art: Artifacts, k: float) -> dict[str, Any]:
    for row in art.json("uncertainty_per_engine.json")["rows"]:
        if abs(row["k"] - k) < 1e-9:
            return row
    raise KeyError(f"uncertainty_per_engine.json has no k={k}")


def ensemble_row(art: Artifacts, w: float) -> dict[str, Any]:
    for row in art.json("ensemble.json")["rows"]:
        if abs(row["w_seq"] - w) < 1e-9:
            return row
    raise KeyError(f"ensemble.json has no w={w}")


def pct_better(worse: float, better: float) -> float:
    """The "N% better than" figure the docs quote, as a percentage."""
    return (worse - better) / worse * 100.0


# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------

README = "README.md"
BENCH = "docs/benchmarks.md"
FIDELITY = "outputs/synthetic_fidelity.md"

#: C-MAPSS ships 21 sensor channels in every subset. Spelled out rather than imported from
#: ``data_loader`` to keep this module importable with nothing but the standard library.
N_CMAPSS_SENSORS = 21


@dataclass(frozen=True)
class TableSpec:
    """A markdown table whose every checked cell is recomputed from artifacts.

    ``expected`` returns one dict per row. The dict's ``key`` entry identifies the row by
    its first cell; the remaining entries map a column-header regex to the value that
    column should hold. Rows the artifact does not describe are left alone — docs are
    allowed to carry commentary rows the pipeline never produced.
    """

    id: str
    doc: str
    anchor: str
    expected: Callable[[Artifacts], list[dict[str, Any]]]
    key: str = "key"


@dataclass(frozen=True)
class ProseSpec:
    """A number embedded in prose, located by a regex with a ``value`` group."""

    id: str
    doc: str
    pattern: str
    expected: Callable[[Artifacts], Expected]
    occurrences: int = 1


Spec = TableSpec | ProseSpec


def _comparison_rows(art: Artifacts) -> list[dict[str, Any]]:
    """The seed-averaged headline table, rebuilt from the per-seed rows."""
    out: list[dict[str, Any]] = []
    for model, rows in comparison_by_model(art).items():
        rmse = [r["rmse"] for r in rows]
        phm = [r["phm"] for r in rows]
        entry: dict[str, Any] = {
            "key": Label(model),
            r"RMSE": mean_ci(rmse),
            r"PHM": mean_ci(phm, dp=1),
            r"range": spread(rmse),
        }
        if rows[0].get("n_params") is not None:
            entry[r"params"] = int(rows[0]["n_params"])
        if rows[0].get("seq_len") is not None:
            entry[r"^config$"] = rerank_config(art, model.lower())
        out.append(entry)
    return out


def _ablation_rows(art: Artifacts) -> list[dict[str, Any]]:
    return [
        {
            "key": Label(row["arm"]),
            r"features": int(row["n_features"]),
            r"RMSE": float(row["rmse"]),
            r"PHM": float(row["phm"]),
            r"late": float(row["pct_late"]),
        }
        for row in art.json("ablation.json")
    ]


def _uncertainty_rows(art: Artifacts) -> list[dict[str, Any]]:
    return [
        {
            "key": NumCells((float(row["quantile"]),)),
            r"offset": float(row["offset"]),
            r"RMSE": float(row["rmse"]),
            r"PHM": float(row["phm"]),
            r"late": float(row["pct_late"]),
        }
        for row in art.json("uncertainty.json")["quantiles"]
    ]


def _coverage_rows(art: Artifacts) -> list[dict[str, Any]]:
    return [
        {
            "key": NumCells((float(row["nominal"]) * 100,)),
            r"empirical": float(row["empirical"]),
            r"width": float(row["mean_width"]),
        }
        for row in art.json("uncertainty.json")["coverage"]
    ]


def _per_engine_rows(art: Artifacts) -> list[dict[str, Any]]:
    return [
        {
            "key": NumCells((float(row["k"]),)),
            r"shift": float(row["mean_shift"]),
            r"per-engine PHM": float(row["pe_phm"]),
            r"uniform PHM": float(row["uni_phm"]),
        }
        for row in art.json("uncertainty_per_engine.json")["rows"]
    ]


def _ensemble_rows(art: Artifacts) -> list[dict[str, Any]]:
    return [
        {
            "key": NumCells((float(row["w_seq"]),)),
            r"val RMSE": float(row["val_rmse"]),
            r"test RMSE": float(row["rmse"]),
            r"test PHM": float(row["phm"]),
        }
        for row in art.json("ensemble.json")["rows"]
    ]


def _bias_rows(art: Artifacts) -> list[dict[str, Any]]:
    """Bias-by-life-stage, keyed by the bin's upper bound so both docs' labels match."""
    tables = parse_tables((art.outputs / "error_analysis.md").read_text())
    art.used.add("error_analysis.md")
    bias = next(t for t in tables if "Bias by life stage" in t.section)
    out: list[dict[str, Any]] = []
    for row in bias.rows:
        upper = numbers_in(row.cells[0].replace(_MINUS, "-"))[-1]
        out.append(
            {
                "key": BinUpper(float(upper)),
                r"^n$|count": int(float(row.cells[1])),
                r"mean": float(row.cells[2].replace(_MINUS, "-")),
                r"std": float(row.cells[3].replace(_MINUS, "-")),
            }
        )
    return out


def _sweep_lstm_rows(art: Artifacts) -> list[dict[str, Any]]:
    """Benchmarks §4 quotes six cells of the LSTM grid, keyed by seq_len."""
    out: list[dict[str, Any]] = []
    for row in sweep_rows(art, "lstm"):
        out.append(
            {
                "key": NumCells((row["seq_len"], row["hidden"], row["lr"])),
                r"val RMSE": float(row["val_rmse"]),
                r"test RMSE": float(row["rmse"]),
                r"test PHM": float(row["phm"]),
            }
        )
    return out


def _sweep_selected_rows(art: Artifacts) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for arch in ("lstm", "gru", "cnn"):
        best = sweep_best(art, arch)
        out.append(
            {
                "key": Label(arch.upper()),
                r"selected config": Config(best["seq_len"], best["hidden"], best["lr"]),
                r"val RMSE": float(best["val_rmse"]),
                r"test RMSE": float(best["rmse"]),
                r"test PHM": float(best["phm"]),
                r"best epoch": Ratio(int(best["best_epoch"]), int(best["epochs_run"])),
            }
        )
    return out


def _variance_rows(art: Artifacts) -> list[dict[str, Any]]:
    """§8c: one row per (arch, condition); keyed by arch, disambiguated by the condition."""
    out: list[dict[str, Any]] = []
    for arch in ("lstm", "gru"):
        for label, cond in (("same seed", "same_seed"), ("different seeds", "different_seeds")):
            try:
                block = variance_block(art, arch, cond)
            except KeyError:
                continue
            out.append(
                {
                    "key": Tokens((arch.upper(), label)),
                    r"RMSE mean": round(float(block["rmse"]["mean"]), 3),
                    r"RMSE std": round(float(block["rmse"]["std"]), 3),
                    r"RMSE spread": round(float(block["rmse"]["spread"]), 3),
                    r"PHM mean": round(float(block["phm"]["mean"]), 1),
                    r"PHM std": round(float(block["phm"]["std"]), 1),
                }
            )
    return out


def _attention_stability_rows(art: Artifacts) -> list[dict[str, Any]]:
    """The attention-vs-GRU stability table, in both docs (README drops the std column)."""
    out: list[dict[str, Any]] = []
    for arch in ("attention", "gru"):
        same = variance_block(art, arch, "same_seed")
        diff = variance_block(art, arch, "different_seeds")
        out.append(
            {
                "key": Label(arch),
                r"seed 42": round(float(same["rmse"]["mean"]), 3),
                r"seed-mean": round(float(diff["rmse"]["mean"]), 3),
                r"RMSE std": round(float(diff["rmse"]["std"]), 3),
                r"(RMSE )?spread": round(float(diff["rmse"]["spread"]), 3),
                r"PHM std": round(float(diff["phm"]["std"]), 1),
            }
        )
    return out


def _importance_rows(art: Artifacts) -> list[dict[str, Any]]:
    importance = art.json("interpretability.json")["importance"]
    return [
        {"key": Label(name), r"RMSE": round(float(value), 3)} for name, value in importance.items()
    ]


def _epoch_budget_rows(art: Artifacts) -> list[dict[str, Any]]:
    """§9's two-budget table: the sweep's GRU run vs the comparison's."""
    swept = sweep_best(art, "gru")
    same = variance_block(art, "gru", "same_seed")
    return [
        {
            "key": NumCells((60,)),
            r"RMSE": float(swept["rmse"]),
            r"PHM": float(swept["phm"]),
            r"epochs run": int(swept["epochs_run"]),
        },
        {
            "key": NumCells((80,)),
            r"RMSE": round(float(same["rmse"]["mean"]), 3),
            r"PHM": round(float(same["phm"]["mean"]), 1),
        },
    ]


def _mean_residual(art: Artifacts) -> float:
    """``src/error_analysis.py`` writes no JSON; its markdown bullet is the only source."""
    return art.md_metrics("error_analysis.md")["Mean residual (pred − true)"]


def _fidelity_rows(art: Artifacts) -> list[dict[str, Any]]:
    """The v2 column of the generator-v2 before/after table.

    Only the v2 column is checkable: the v1 column records numbers produced by a generator
    that no longer exists in the tree, so there is no artifact to re-derive it from. That
    column is a historical record and is deliberately left unguarded; the live one is not,
    because it duplicates ``metrics.json`` and would otherwise go stale on the next
    ``make baseline``.
    """
    metrics = art.json("metrics.json")
    art.used.add("synthetic_fidelity.md")
    return [
        {"key": Label("informative sensors"), r"v2": _n_sensors(art)},
        {"key": Label("feature columns"), r"v2": 3 * _n_sensors(art)},
        {"key": Label("RMSE"), r"v2": float(metrics["rmse"])},
        {"key": Label("PHM score"), r"v2": float(metrics["phm_score"])},
        {"key": Label("mean residual"), r"v2": _mean_residual(art)},
    ]


def _baseline_rows(art: Artifacts) -> list[dict[str, Any]]:
    metrics = art.json("metrics.json")
    return [
        {
            "key": Label("RandomForestRegressor"),
            r"RMSE": float(metrics["rmse"]),
            r"PHM": float(metrics["phm_score"]),
            r"engines": int(metrics["n_test_engines"]),
        }
    ]


# -- claims restated in both docs, so their derivation lives in one place ----


def _gru_rmse_gain_pct(art: Artifacts) -> int:
    by = comparison_by_model(art)
    return round(
        pct_better(
            mean_spread([r["rmse"] for r in by["RandomForest"]]).mean,
            mean_spread([r["rmse"] for r in by["GRU"]]).mean,
        )
    )


def _gru_phm_gain_pct(art: Artifacts) -> int:
    by = comparison_by_model(art)
    return round(
        pct_better(
            mean_spread([r["phm"] for r in by["RandomForest"]]).mean,
            mean_spread([r["phm"] for r in by["GRU"]]).mean,
        )
    )


def _model_range(art: Artifacts, model: str) -> float:
    return spread([r["rmse"] for r in comparison_by_model(art)[model]])


def _neural_ranges(art: Artifacts) -> list[float]:
    by = comparison_by_model(art)
    return sorted(_model_range(art, m) for m in by if m != "RandomForest")


def _global_phm_gain(art: Artifacts) -> float:
    base = art.json("uncertainty.json")["baseline"]["phm"]
    return round(pct_better(base, uncertainty_row(art, 0.7)["phm"]), 1)


def _global_rmse_cost(art: Artifacts) -> float:
    base = art.json("uncertainty.json")["baseline"]["rmse"]
    return round(-pct_better(base, uncertainty_row(art, 0.7)["rmse"]), 1)


def _n_sensors(art: Artifacts) -> int:
    return len(art.json("feature_spec.json")["sensors"])


def _n_dropped_sensors(art: Artifacts) -> int:
    """How many of the 21 the selector threw away — the claim both docs actually make.

    Checked as *dropped* rather than *kept* because that is the number the prose quotes,
    and because it is the one that says whether the synthetic fallback takes the same
    code path as the real data (6 there; 0 on generator v1).
    """
    return N_CMAPSS_SENSORS - _n_sensors(art)


def _n_test_engines(art: Artifacts) -> int:
    return int(art.json("metrics.json")["n_test_engines"])


def _attention_val_spread(art: Artifacts) -> float:
    return float(rerank_best(art, "attention")["val_range"])


def _seed42_penalty(art: Artifacts, arch: str, dp: int) -> float:
    """How much seed 42 flatters an architecture against its own across-seed mean."""
    diff = (
        variance_block(art, arch, "different_seeds")["rmse"]["mean"]
        - variance_block(art, arch, "same_seed")["rmse"]["mean"]
    )
    return round(diff, dp)


def build_specs() -> list[Spec]:
    """The claim registry: every doc number this module knows how to re-derive."""
    specs: list[Spec] = [
        # -- tables -------------------------------------------------------------
        TableSpec("readme.comparison", README, r"Model comparison", _comparison_rows),
        TableSpec("readme.ablation", README, r"Feature ablation", _ablation_rows),
        TableSpec(
            "readme.attention", README, r"best number, worse engineering", _attention_stability_rows
        ),
        TableSpec("readme.uncertainty", README, r"residual quantile", _uncertainty_rows),
        TableSpec("readme.coverage", README, r"nominal.*empirical", _coverage_rows),
        TableSpec("bench.baseline", BENCH, r"^## 1\. Baseline", _baseline_rows),
        TableSpec("fidelity.baseline", FIDELITY, r"before vs after", _fidelity_rows),
        TableSpec("bench.bias", BENCH, r"Bias by life stage", _bias_rows),
        TableSpec("bench.ablation", BENCH, r"^## 3\. Feature ablation", _ablation_rows),
        TableSpec("bench.sweep_lstm", BENCH, r"27 configurations", _sweep_lstm_rows),
        TableSpec("bench.sweep_selected", BENCH, r"each on its own grid", _sweep_selected_rows),
        TableSpec("bench.comparison", BENCH, r"^## 5\. Model comparison", _comparison_rows),
        TableSpec("bench.uncertainty", BENCH, r"^## 6\. Uncertainty", _uncertainty_rows),
        TableSpec("bench.coverage", BENCH, r"Interval coverage", _coverage_rows),
        TableSpec("bench.per_engine", BENCH, r"per-engine PHM", _per_engine_rows),
        TableSpec("bench.ensemble", BENCH, r"w on GRU", _ensemble_rows),
        TableSpec("bench.epoch_budget", BENCH, r"budget.*best epoch", _epoch_budget_rows),
        # The benchmarks variance table was removed with §8c: the numbers now live only in
        # outputs/variance.md, generated from the artifact. Nothing here to cross-check.
        TableSpec("bench.attention", BENCH, r"^## 8d\.", _attention_stability_rows),
        TableSpec("bench.importance", BENCH, r"Permutation importance", _importance_rows),
        # -- prose: model comparison -------------------------------------------
        ProseSpec(
            "gru_vs_forest_rmse",
            README,
            r"(?P<value>\d+)% better RMSE",
            _gru_rmse_gain_pct,
        ),
        ProseSpec(
            "gru_vs_forest_phm",
            README,
            r"(?P<value>\d+)% better PHM",
            _gru_phm_gain_pct,
        ),
        ProseSpec(
            "gru_lr_correction",
            README,
            r"worth (?P<value>[\d.]+) RMSE",
            _gru_lr_gain,
        ),
        ProseSpec(
            "gru_lstm_gap",
            README,
            r"leads the LSTM by (?P<value>[\d.]+) RMSE",
            lambda a: _model_gap(a, "LSTM", "GRU"),
            occurrences=2,
        ),
        ProseSpec(
            "gru_lstm_gap_ranges",
            README,
            r"ranges\s+of (?P<value>[\d.]+) and [\d.]+",
            lambda a: _model_range(a, "GRU"),
            occurrences=2,
        ),
        ProseSpec(
            "lstm_range",
            README,
            r"ranges\s+of [\d.]+ and (?P<value>[\d.]+)",
            lambda a: _model_range(a, "LSTM"),
            occurrences=2,
        ),
        ProseSpec(
            "forest_range",
            README,
            r"across-seed range is \*?\*?(?P<value>[\d.]+)",
            lambda a: _model_range(a, "RandomForest"),
        ),
        ProseSpec(
            "attention_val_spread",
            README,
            r"varies by \*\*(?P<value>[\d.]+)\*\* validation RMSE",
            _attention_val_spread,
        ),
        ProseSpec(
            "attention_swept_optimum",
            README,
            r"its swept optimum\s+is (?P<value>[\d/e-]+)",
            lambda a: _config_of(sweep_best(a, "attention")),
        ),
        # -- prose: stability ---------------------------------------------------
        ProseSpec(
            "attention_instability_factor",
            README,
            r"(?P<value>[\d.]+)×\s*\n?\s*less stable",
            lambda a: round(
                variance_block(a, "attention", "different_seeds")["rmse"]["spread"]
                / variance_block(a, "gru", "different_seeds")["rmse"]["spread"],
                1,
            ),
        ),
        ProseSpec(
            "attention_seed42_win",
            README,
            r"the (?P<value>\d+)% win is not architectural",
            lambda a: round(
                pct_better(
                    variance_block(a, "gru", "same_seed")["rmse"]["mean"],
                    variance_block(a, "attention", "same_seed")["rmse"]["mean"],
                )
            ),
        ),
        ProseSpec(
            "attention_seed42_flatter",
            README,
            r"flatters it by (?P<value>[\d.]+) RMSE",
            lambda a: _seed42_penalty(a, "attention", 2),
        ),
        ProseSpec(
            "gru_seed42_flatter",
            README,
            r"versus\s*\n?\s*(?P<value>[\d.]+) for the GRU",
            lambda a: _seed42_penalty(a, "gru", 2),
        ),
        ProseSpec(
            "attention_phm_range",
            README,
            r"PHM range spans (?P<value>[\d.]+) to [\d.]+",
            lambda a: float(variance_block(a, "attention", "different_seeds")["phm"]["min"]),
        ),
        ProseSpec(
            "attention_phm_range_hi",
            README,
            r"PHM range spans [\d.]+ to (?P<value>[\d.]+)",
            lambda a: float(variance_block(a, "attention", "different_seeds")["phm"]["max"]),
        ),
        # -- prose: ensemble ----------------------------------------------------
        ProseSpec(
            "ensemble_weight",
            README,
            r"selects \*\*w=(?P<value>[\d.]+)\*\*",
            lambda a: float(a.json("ensemble.json")["chosen_w"]),
        ),
        ProseSpec(
            "ensemble_test_rmse",
            README,
            r"`w=0\.8` scores (?P<value>[\d.]+) RMSE",
            lambda a: float(ensemble_row(a, 0.8)["rmse"]),
        ),
        ProseSpec(
            "ensemble_test_phm",
            README,
            r"`w=0\.8` scores [\d.]+ RMSE / (?P<value>[\d.]+) PHM",
            lambda a: float(ensemble_row(a, 0.8)["phm"]),
        ),
        ProseSpec(
            "ensemble_weights_offered",
            README,
            r"(?P<value>\w+) weights on offer",
            lambda a: _spell(len(a.json("ensemble.json")["rows"])),
        ),
        ProseSpec(
            "ensemble_test_engines",
            README,
            r"With (?P<value>\d+) test engines",
            _n_test_engines,
        ),
        # -- prose: uncertainty -------------------------------------------------
        ProseSpec(
            "global_offset_phm_gain",
            README,
            r"recovers \*\*(?P<value>[\d.]+)% of PHM",
            _global_phm_gain,
        ),
        ProseSpec(
            "global_offset_rmse_cost",
            README,
            r"for a (?P<value>[\d.]+)% RMSE cost",
            _global_rmse_cost,
        ),
        ProseSpec(
            "per_engine_k",
            README,
            r"k·sigma \(k=(?P<value>[\d.]+)\)",
            lambda a: float(a.json("uncertainty_per_engine.json")["k_selected"]),
        ),
        ProseSpec(
            "per_engine_gain",
            README,
            r"\*\*−(?P<value>[\d.]+)%\*\*",
            lambda a: round(
                pct_better(
                    a.json("uncertainty_per_engine.json")["baseline"]["phm"],
                    per_engine_row(a, 0.25)["pe_phm"],
                ),
                1,
            ),
        ),
        ProseSpec(
            "per_engine_mean_shift",
            README,
            r"\*same\* mean amount \((?P<value>[\d.]+) cycles\)",
            lambda a: float(per_engine_row(a, 0.25)["mean_shift"]),
        ),
        ProseSpec(
            "per_engine_vs_uniform",
            README,
            r"cycles\) by (?P<value>[\d.]+) PHM",
            lambda a: round(
                per_engine_row(a, 0.25)["uni_phm"] - per_engine_row(a, 0.25)["pe_phm"], 1
            ),
        ),
        ProseSpec(
            "sigma_error_corr",
            README,
            r"correlates \*\*\+(?P<value>[\d.]+)\*\*",
            lambda a: float(a.json("uncertainty_per_engine.json")["corr_sigma_abserr"]),
        ),
        # -- prose: baseline, features, error analysis --------------------------
        ProseSpec(
            "minimal_rmse",
            README,
            r"RMSE (?P<value>[\d.]+), alert at cycle",
            lambda a: float(a.json("metrics.json")["rmse"]),
        ),
        ProseSpec(
            "mean_residual",
            README,
            r"\(\*\*(?P<value>[-+−][\d.]+)\*\* cycles\)",
            _mean_residual,
        ),
        ProseSpec(
            "sensor_count",
            README,
            r"drops\s+\*\*(?P<value>\d+) of 21\*\*",
            _n_dropped_sensors,
            occurrences=2,
        ),
        ProseSpec(
            "rul_cap",
            README,
            r"capped at (?P<value>\d+)",
            lambda a: 125,
        ),
        ProseSpec(
            "rolling_rmse_gain",
            README,
            r"only ~(?P<value>\d+)% RMSE for",
            lambda a: round(_ablation_gain(a)),
        ),
        # -- prose: interpretability --------------------------------------------
        ProseSpec(
            "attention_recent_share",
            BENCH,
            r"hold \*\*(?P<value>[\d.]+)%\*\* of the total",
            lambda a: round(a.json("interpretability.json")["recent_quarter_share"] * 100, 1),
        ),
        ProseSpec(
            "attention_window",
            BENCH,
            r"most recent (?P<value>\d+) of \d+ cycles",
            lambda a: _recent_cycles(a),
        ),
        ProseSpec(
            "attention_seq_len",
            BENCH,
            r"most recent \d+ of (?P<value>\d+) cycles",
            lambda a: len(a.json("interpretability.json")["attention_profile"]),
        ),
        ProseSpec(
            "attention_uniform_share",
            BENCH,
            r"against \*\*(?P<value>[\d.]+)%\*\* for uniform",
            lambda a: round(
                _recent_cycles(a) / len(a.json("interpretability.json")["attention_profile"]) * 100,
                1,
            ),
        ),
        # -- prose: sweep -------------------------------------------------------
        ProseSpec(
            "sweep_lstm_n_configs",
            BENCH,
            r"(?P<value>\d+) configurations \(seq_len",
            lambda a: a.md_frontmatter("sweep_lstm.md")["n_configs"],
        ),
        ProseSpec(
            "sweep_lstm_selected",
            BENCH,
            r"Selected:\n`(?P<value>seq_len=\d+, hidden=\d+, lr=[\de.-]+)`",
            lambda a: _config_of(sweep_best(a, "lstm")),
        ),
        ProseSpec(
            "cnn_at_lstm_config_rmse",
            BENCH,
            r"LSTM's config it scored (?P<value>[\d.]+) RMSE",
            lambda a: float(sweep_at(a, "cnn", _config_of(sweep_best(a, "lstm")))["rmse"]),
        ),
        ProseSpec(
            "cnn_at_lstm_config_phm",
            BENCH,
            r"LSTM's config it scored [\d.]+ RMSE / (?P<value>[\d.]+) PHM",
            lambda a: float(sweep_at(a, "cnn", _config_of(sweep_best(a, "lstm")))["phm"]),
        ),
        ProseSpec(
            "cnn_own_config_rmse",
            BENCH,
            r"on its own it scores\n?\*?\*?(?P<value>[\d.]+) / [\d.]+\*?\*?",
            lambda a: float(sweep_best(a, "cnn")["rmse"]),
        ),
        ProseSpec(
            "cnn_tuning_rmse_gain",
            BENCH,
            r"a (?P<value>\d+)% RMSE and \d+% PHM improvement",
            lambda a: round(
                pct_better(
                    sweep_at(a, "cnn", _config_of(sweep_best(a, "lstm")))["rmse"],
                    sweep_best(a, "cnn")["rmse"],
                )
            ),
        ),
        ProseSpec(
            "cnn_tuning_phm_gain",
            BENCH,
            r"a \d+% RMSE and (?P<value>\d+)% PHM improvement",
            lambda a: round(
                pct_better(
                    sweep_at(a, "cnn", _config_of(sweep_best(a, "lstm")))["phm"],
                    sweep_best(a, "cnn")["phm"],
                )
            ),
        ),
        # -- prose: variance ----------------------------------------------------
        # `variance_gru_lstm_gap` and `variance_single_seed_gap` were removed with the prose
        # they checked. Benchmarks §8c no longer restates the variance table by hand — it
        # points at `outputs/variance.md`, which src/variance.py generates from the artifact.
        # A number that lives in exactly one place cannot drift, so there is nothing to check.
        ProseSpec(
            "attention_means_gap",
            BENCH,
            r"a (?P<value>[\d.]+) gap against a [\d.]+ spread",
            lambda a: round(
                variance_block(a, "gru", "different_seeds")["rmse"]["mean"]
                - variance_block(a, "attention", "different_seeds")["rmse"]["mean"],
                3,
            ),
        ),
        ProseSpec(
            "seeds_averaged",
            BENCH,
            r"averaged over \*\*(?P<value>\d+) seeds\*\*",
            lambda a: len({r["seed"] for r in a.json("comparison.json")}),
        ),
        # -- prose: the same claims, restated in benchmarks ---------------------
        # Deliberately duplicated specs rather than one spec spanning both files: the
        # wording differs, and a claim that survives in one doc while going stale in the
        # other is exactly the drift worth catching.
        ProseSpec(
            "bench.gru_vs_forest_rmse", BENCH, r"(?P<value>\d+)% better RMSE", _gru_rmse_gain_pct
        ),
        ProseSpec(
            "bench.gru_vs_forest_phm", BENCH, r"(?P<value>\d+)% better PHM", _gru_phm_gain_pct
        ),
        ProseSpec(
            "bench.gru_lr_correction",
            BENCH,
            r"worth (?P<value>[\d.]+) RMSE",
            _gru_lr_gain,
            occurrences=2,
        ),
        ProseSpec(
            "bench.gru_lr_before",
            BENCH,
            r"worth [\d.]+ RMSE \((?P<value>[\d.]+) → [\d.]+\)",
            lambda a: float(
                next(r for r in a.json("rerank_gru.json") if r["original_rank"] == 1)["test_mean"]
            ),
        ),
        ProseSpec(
            "bench.gru_lr_after",
            BENCH,
            r"worth [\d.]+ RMSE \([\d.]+ → (?P<value>[\d.]+)\)",
            lambda a: float(rerank_best(a, "gru")["test_mean"]),
        ),
        ProseSpec(
            "bench.gru_lstm_gap",
            BENCH,
            r"lead over the LSTM is (?P<value>[\d.]+) RMSE",
            lambda a: _model_gap(a, "LSTM", "GRU"),
        ),
        ProseSpec(
            "bench.gru_range",
            BENCH,
            r"ranges\s+of (?P<value>[\d.]+) and [\d.]+",
            lambda a: _model_range(a, "GRU"),
        ),
        ProseSpec(
            "bench.lstm_range",
            BENCH,
            r"ranges\s+of [\d.]+ and (?P<value>[\d.]+)",
            lambda a: _model_range(a, "LSTM"),
        ),
        ProseSpec(
            "bench.forest_range",
            BENCH,
            r"across-seed range is \*?\*?(?P<value>[\d.]+)",
            lambda a: _model_range(a, "RandomForest"),
        ),
        ProseSpec(
            "bench.attention_val_spread",
            BENCH,
            r"varies by \*\*(?P<value>[\d.]+)\*\* validation RMSE",
            _attention_val_spread,
        ),
        ProseSpec(
            "bench.global_offset_phm_gain",
            BENCH,
            r"recovers \*\*(?P<value>[\d.]+)% of PHM",
            _global_phm_gain,
        ),
        ProseSpec(
            "bench.global_offset_rmse_cost",
            BENCH,
            r"for a (?P<value>[\d.]+)% RMSE cost",
            _global_rmse_cost,
        ),
        ProseSpec(
            "bench.sensor_count", BENCH, r"drops \*\*(?P<value>\d+) of 21\*\*", _n_dropped_sensors
        ),
        ProseSpec(
            "bench.test_engines",
            BENCH,
            r"with (?P<value>\d+) test engines",
            _n_test_engines,
        ),
        ProseSpec(
            "bench.per_engine_k",
            BENCH,
            r"independently selected \*\*k=(?P<value>[\d.]+)\*\*",
            lambda a: float(a.json("uncertainty_per_engine.json")["k_selected"]),
        ),
        ProseSpec(
            "bench.per_engine_gain",
            BENCH,
            r"a \*\*(?P<value>[\d.]+)%\*\* gain",
            lambda a: round(
                pct_better(
                    a.json("uncertainty_per_engine.json")["baseline"]["phm"],
                    per_engine_row(a, 0.25)["pe_phm"],
                ),
                1,
            ),
        ),
        ProseSpec(
            "bench.sigma_error_corr",
            BENCH,
            r"corr\(sigma, \|error\|\) = \*\*\+(?P<value>[\d.]+)",
            lambda a: float(a.json("uncertainty_per_engine.json")["corr_sigma_abserr"]),
        ),
        ProseSpec(
            "bench.attention_spread_vs_gap",
            BENCH,
            r"a [\d.]+ gap against a (?P<value>[\d.]+) spread",
            lambda a: round(
                float(variance_block(a, "attention", "different_seeds")["rmse"]["spread"]), 3
            ),
        ),
        ProseSpec(
            "bench.attention_seed42_flatter",
            BENCH,
            r"flatters it by (?P<value>[\d.]+) RMSE",
            lambda a: _seed42_penalty(a, "attention", 3),
        ),
        ProseSpec(
            "bench.gru_seed42_flatter",
            BENCH,
            r"equivalent penalty is (?P<value>[\d.]+)",
            lambda a: _seed42_penalty(a, "gru", 3),
        ),
        ProseSpec(
            "bench.attention_instability_factor",
            BENCH,
            r"attention is (?P<value>[\d.]+)× less stable",
            lambda a: round(
                variance_block(a, "attention", "different_seeds")["rmse"]["spread"]
                / variance_block(a, "gru", "different_seeds")["rmse"]["spread"],
                1,
            ),
        ),
        # -- prose: the neural-vs-forest stability contrast, in both docs -------
        ProseSpec(
            "neural_range_low",
            README,
            r"against (?P<value>[\d.]+)–[\d.]+ for every neural model",
            lambda a: round(_neural_ranges(a)[0], 1),
        ),
        ProseSpec(
            "neural_range_high",
            README,
            r"against [\d.]+–(?P<value>[\d.]+) for every neural model",
            lambda a: round(_neural_ranges(a)[-1], 1),
        ),
        ProseSpec(
            "forest_tighter_low",
            README,
            r"(?P<value>\d+)× to \d+×\s+tighter",
            lambda a: round(_neural_ranges(a)[0] / _model_range(a, "RandomForest")),
        ),
        ProseSpec(
            "forest_tighter_high",
            README,
            r"\d+× to (?P<value>\d+)×\s+tighter",
            lambda a: round(_neural_ranges(a)[-1] / _model_range(a, "RandomForest")),
        ),
        ProseSpec(
            "bench.neural_range_low",
            BENCH,
            r"against (?P<value>[\d.]+)–[\d.]+ for\nevery neural model",
            lambda a: round(_neural_ranges(a)[0], 1),
        ),
        ProseSpec(
            "bench.forest_tighter_low",
            BENCH,
            r"(?P<value>\d+)× to \d+×\s+tighter",
            lambda a: round(_neural_ranges(a)[0] / _model_range(a, "RandomForest")),
        ),
        ProseSpec(
            "bench.forest_tighter_high",
            BENCH,
            r"\d+× to (?P<value>\d+)×\s+tighter",
            lambda a: round(_neural_ranges(a)[-1] / _model_range(a, "RandomForest")),
        ),
    ]
    return specs


# -- small derived helpers used by the prose specs above ---------------------


def _config_of(row: dict[str, Any]) -> Config:
    return Config(row["seq_len"], row["hidden"], row["lr"])


def _model_gap(art: Artifacts, worse: str, better: str) -> float:
    by = comparison_by_model(art)
    return round(
        mean_spread([r["rmse"] for r in by[worse]]).mean
        - mean_spread([r["rmse"] for r in by[better]]).mean,
        3,
    )


def _gru_lr_gain(art: Artifacts) -> float:
    """What correcting the GRU's configuration selection was worth, per rerank_gru.json."""
    rows = art.json("rerank_gru.json")
    single = next(r for r in rows if r["original_rank"] == 1)
    chosen = rerank_best(art, "gru")
    return round(single["test_mean"] - chosen["test_mean"], 2)


def _ablation_gain(art: Artifacts) -> float:
    rows = art.json("ablation.json")
    raw = next(r for r in rows if "raw" in r["arm"])
    best = min(rows, key=lambda r: r["rmse"])
    return pct_better(raw["rmse"], best["rmse"])


def _recent_cycles(art: Artifacts) -> int:
    """How many trailing cycles ``recent_quarter_share`` covers — a quarter, rounded up."""
    profile = art.json("interpretability.json")["attention_profile"]
    return -(-len(profile) // 4)


def _lstm_seed_mean(art: Artifacts) -> float:
    """The LSTM's across-seed mean. variance.json only holds attention+gru, so use the
    comparison run, which is the same quantity from the same seeds."""
    return mean_spread([r["rmse"] for r in comparison_by_model(art)["LSTM"]]).mean


def _lstm_same_seed(art: Artifacts) -> float:
    rows = comparison_by_model(art)["LSTM"]
    return float(next(r for r in rows if r["seed"] == 42)["rmse"])


_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}


def _spell(n: int) -> str:
    return _WORDS.get(n, str(n))


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

OK, WARN, ERROR = "ok", "warn", "error"


@dataclass(frozen=True)
class Finding:
    severity: str
    doc: str
    line: int
    spec: str
    what: str
    expected: str
    actual: str

    @property
    def location(self) -> str:
        return f"{self.doc}:{self.line}"


def compare_number(expected: float, literal: str) -> str:
    """``ok`` / ``warn`` / ``error`` for a number printed at the precision in ``literal``.

    A doc that prints 3 decimals is claiming the value to ±0.0005. One extra unit in that
    last digit is a rounding-path difference (a mean of rounded per-seed values is not the
    same as the rounded mean) and gets a warning; anything wider is drift.
    """
    actual = float(literal.replace(",", ""))
    unit = 10.0 ** -decimals(literal)
    diff = abs(expected - actual)
    if diff <= unit / 2 + 1e-9:
        return OK
    if diff <= unit + 1e-9:
        return WARN
    return ERROR


def compare(expected: Expected, cell: str) -> tuple[str, str]:
    """Grade a doc cell against its expected value; returns ``(severity, actual_str)``."""
    text = clean(cell)
    if isinstance(expected, Config):
        actual_cfg = parse_config(text)
        if actual_cfg is None:
            return ERROR, cell.strip() or "(empty)"
        return (OK if actual_cfg == expected else ERROR), str(actual_cfg)
    if isinstance(expected, Ratio):
        actual_ratio = parse_ratio(text)
        if actual_ratio is None:
            return ERROR, cell.strip() or "(empty)"
        return (OK if actual_ratio == expected else ERROR), str(actual_ratio)
    if isinstance(expected, MeanSpread):
        actual_ms = parse_mean_spread(text)
        if actual_ms is None:
            return ERROR, cell.strip() or "(empty)"
        literals = numbers_in(text)
        worst = _worst(
            compare_number(expected.mean, literals[0]),
            compare_number(expected.half, literals[1]),
        )
        return worst, str(actual_ms)
    if isinstance(expected, bool):  # pragma: no cover - not used, but keeps bool != int
        return (OK if str(expected).lower() in text.lower() else ERROR), text
    if isinstance(expected, int):
        literals = numbers_in(text)
        if not literals:
            return ERROR, cell.strip() or "(empty)"
        actual_int = int(float(literals[0].replace(",", "")))
        return (OK if actual_int == expected else ERROR), str(actual_int)
    if isinstance(expected, float):
        literals = numbers_in(text)
        if not literals:
            return ERROR, cell.strip() or "(empty)"
        return compare_number(expected, literals[0]), literals[0]
    return (OK if text.strip().lower() == str(expected).strip().lower() else ERROR), text


def _worst(*severities: str) -> str:
    for level in (ERROR, WARN):
        if level in severities:
            return level
    return OK


def format_expected(expected: Expected) -> str:
    if isinstance(expected, float):
        return f"{expected:g}"
    return str(expected)


# ---------------------------------------------------------------------------
# running the specs
# ---------------------------------------------------------------------------


def _match_column(header: tuple[str, ...], pattern: str) -> int | None:
    rx = re.compile(pattern, re.I)
    for idx, cell in enumerate(header):
        if rx.search(clean(cell)):
            return idx
    return None


def run_table_spec(
    spec: TableSpec, art: Artifacts, tables: dict[str, list[Table]]
) -> tuple[list[Finding], int]:
    """Compare a table's cells; returns the findings and how many cells were compared.

    The count matters as much as the findings. A spec whose anchor drifts, or whose row
    keys stop matching, would otherwise report a clean pass while checking nothing — the
    failure mode most likely to make this whole module quietly worthless. Zero comparisons
    is therefore an error, not a pass.
    """
    rows_expected = spec.expected(art)
    anchor = re.compile(spec.anchor, re.I | re.M)
    candidates = [t for t in tables[spec.doc] if anchor.search(t.context)]
    if not candidates:
        return (
            [
                Finding(
                    ERROR,
                    spec.doc,
                    0,
                    spec.id,
                    "table",
                    f"a table matching /{spec.anchor}/",
                    "none",
                )
            ],
            0,
        )

    findings: list[Finding] = []
    checked = 0
    for table in candidates:
        for expected_row in rows_expected:
            key: Key = expected_row[spec.key]
            for doc_row in (r for r in table.rows if key.matches(r)):
                for column, expected in expected_row.items():
                    if column == spec.key:
                        continue
                    idx = _match_column(table.header, column)
                    if idx is None or idx >= len(doc_row.cells):
                        continue
                    checked += 1
                    severity, actual = compare(expected, doc_row.cells[idx])
                    if severity == OK:
                        continue
                    findings.append(
                        Finding(
                            severity,
                            spec.doc,
                            doc_row.line,
                            spec.id,
                            f"{key} / {clean(table.header[idx]) or f'col {idx}'}",
                            format_expected(expected),
                            actual,
                        )
                    )
    if checked == 0:
        findings.append(
            Finding(
                ERROR,
                spec.doc,
                candidates[0].line,
                spec.id,
                "table matched but no rows did",
                "at least one comparable cell",
                "0 cells checked",
            )
        )
    return findings, checked


def run_prose_spec(
    spec: ProseSpec, art: Artifacts, text: dict[str, str]
) -> tuple[list[Finding], int]:
    """Compare a prose claim wherever it appears; returns findings and comparison count.

    ``occurrences`` is checked too: several of these claims are deliberately stated in
    both README and benchmarks, or twice in one file. If a restatement is added or
    dropped, the new copy is unguarded, so the count changing is itself a finding.
    """
    expected = spec.expected(art)
    body = text[spec.doc]
    matches = list(re.finditer(spec.pattern, body, re.I | re.M))
    if not matches:
        return (
            [
                Finding(
                    ERROR, spec.doc, 0, spec.id, "prose", f"/{spec.pattern}/ to match", "no match"
                )
            ],
            0,
        )

    findings: list[Finding] = []
    if len(matches) != spec.occurrences:
        findings.append(
            Finding(
                WARN,
                spec.doc,
                body.count("\n", 0, matches[0].start()) + 1,
                spec.id,
                "claim restated a different number of times",
                f"{spec.occurrences} occurrence(s)",
                f"{len(matches)}",
            )
        )
    for match in matches:
        severity, actual = compare(expected, match.group("value"))
        if severity == OK:
            continue
        findings.append(
            Finding(
                severity,
                spec.doc,
                body.count("\n", 0, match.start()) + 1,
                spec.id,
                match.group(0).strip().replace("\n", " ")[:60],
                format_expected(expected),
                actual,
            )
        )
    return findings, len(matches)


# ---------------------------------------------------------------------------
# artifact coverage
# ---------------------------------------------------------------------------

#: Number-bearing artifacts no spec reads, each with the reason it is excused. Anything
#: not listed here and not consulted by a spec is reported as an uncovered artifact.
UNCHECKED: dict[str, str] = {
    "sweep_gru.md": "same numbers as sweep_gru.json, which is checked",
    "sweep_cnn.md": "same numbers as sweep_cnn.json, which is checked",
    "sweep_attention.md": "same numbers as sweep_attention.json, which is checked",
    "rerank_lstm.md": "rendering of rerank_lstm.json, which is checked",
    "rerank_gru.md": "rendering of rerank_gru.json, which is checked",
    "rerank_cnn.md": "rendering of rerank_cnn.json, which is checked",
    "rerank_attention.md": "rendering of rerank_attention.json, which is checked",
    "comparison.md": "rendering of comparison.json, which is checked",
    # The published baselines are *inputs* hardcoded in src/compare_published.py, not outputs
    # of a run, so there is no artifact to re-derive them from — checking them would only
    # compare the module against itself. The cells that ARE derived (our means, CIs and the
    # deltas) are recomputed from comparison.json, which is checked, on every run of the
    # script. What guards the literature values instead is the per-row `confidence` field and
    # §6 of the report; neither is quoted in README.md or docs/benchmarks.md.
    "published_comparison.md": "rendering of published_comparison.json",
    "published_comparison.json": "hardcoded literature baselines (no artifact to check "
    "against) + cells re-derived from comparison.json, which is checked",
    "ablation.md": "rendering of ablation.json, which is checked",
    "ensemble.md": "rendering of ensemble.json, which is checked",
    "uncertainty.md": "rendering of uncertainty.json, which is checked",
    "uncertainty_per_engine.md": "rendering of uncertainty_per_engine.json, which is checked",
    "variance.md": "rendering of variance.json, which is checked",
    "interpretability.md": "rendering of interpretability.json, which is checked",
}

#: Extensions that can carry a number into the docs. Everything else is a figure or a
#: pickled model: still audited, but only for being a file nobody declared.
NUMBER_BEARING = {".json", ".md"}

#: Non-numeric artifacts with no same-stem JSON/markdown sibling to vouch for them.
KNOWN_BINARIES = {
    "baseline.pkl",
    "residuals.png",
    "trajectories.png",
    "pred_vs_true.png",
    "live_twin_engine48.png",
}


def audit_artifacts(
    art: Artifacts,
    unchecked: dict[str, str] | None = None,
    binaries: set[str] | None = None,
) -> list[Finding]:
    """Warn about anything in ``outputs/`` that no spec looked at.

    This is the half of the problem that a claim registry alone cannot solve: specs check
    that quoted numbers are right, but a *new* artifact can add numbers to the docs that
    no spec knows to look for. Failing loudly on an unrecognised file forces the choice —
    check it, or write down why not.

    The two allowlists are arguments rather than globals so the audit can be exercised
    against a directory other than this repo's.
    """
    unchecked = UNCHECKED if unchecked is None else unchecked
    binaries = KNOWN_BINARIES if binaries is None else binaries
    findings: list[Finding] = []
    stems = {p.stem for p in art.outputs.iterdir() if p.suffix in NUMBER_BEARING}
    for path in sorted(art.outputs.iterdir()):
        if path.is_dir() or path.name.startswith("."):
            continue
        name = path.name
        if name in art.used or name in unchecked:
            continue
        if path.suffix in NUMBER_BEARING:
            findings.append(
                Finding(
                    WARN,
                    f"outputs/{name}",
                    0,
                    "coverage",
                    "uncovered artifact",
                    "a spec that reads it, or an entry in UNCHECKED",
                    "neither",
                )
            )
        elif name not in binaries and path.stem not in stems:
            findings.append(
                Finding(
                    WARN,
                    f"outputs/{name}",
                    0,
                    "coverage",
                    "undeclared artifact",
                    "an entry in KNOWN_BINARIES, or a same-stem json/md",
                    "neither",
                )
            )
    for name in sorted(unchecked):
        if not (art.outputs / name).exists():
            findings.append(
                Finding(
                    WARN,
                    f"outputs/{name}",
                    0,
                    "coverage",
                    "stale UNCHECKED entry",
                    "the file to exist",
                    "missing",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    specs_run: int = 0
    checks_run: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    artifacts_used: set[str] = field(default_factory=set)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARN]

    def failed(self, strict: bool = False) -> bool:
        return bool(self.errors) or (strict and bool(self.warnings))


#: Where the docs quote this module's own totals. Checking them is not a joke at the
#: module's expense: "343 numbers across 94 specs" is a hand-copied figure in a markdown
#: file, which is precisely the thing this module exists to distrust. The counts are read
#: after every other spec has run and are not themselves counted, so they stay stable.
SELF_CLAIMS: list[tuple[str, str]] = [
    (README, r"\*\*(?P<checks>\d+) numbers across (?P<specs>\d+) specs\*\*"),
    (BENCH, r"\| doc numbers \| \*\*(?P<checks>\d+)\*\*"),
    (BENCH, r"`src/validate_docs\.py` recomputes"),
]


def self_check(report: Report, text: dict[str, str]) -> list[Finding]:
    """Grade the docs' claims about this module's own coverage against the live totals."""
    findings: list[Finding] = []
    actuals = {"checks": report.checks_run, "specs": report.specs_run}
    for doc, pattern in SELF_CLAIMS:
        match = re.search(pattern, text.get(doc, ""))
        if match is None:
            findings.append(
                Finding(WARN, doc, 0, "self", "coverage claim", f"/{pattern}/ to match", "no match")
            )
            continue
        line = text[doc].count("\n", 0, match.start()) + 1
        for name, expected in actuals.items():
            claimed = match.groupdict().get(name)
            if claimed is None or int(claimed) == expected:
                continue
            findings.append(
                Finding(ERROR, doc, line, "self", f"{name} claimed here", str(expected), claimed)
            )
    return findings


def validate(
    root: Path, specs: list[Spec] | None = None, require_artifacts: bool = False
) -> Report:
    """Run every spec against the docs under ``root`` and audit ``outputs/`` coverage.

    ``outputs/*.json`` is gitignored in this repo — only the rendered markdown reports are
    committed — so a fresh checkout has no artifacts to check against. Missing artifacts
    therefore *skip* their specs by default rather than failing, and the skip count is
    reported so a run that checked almost nothing cannot be mistaken for a clean one. Pass
    ``require_artifacts`` where ``outputs/`` is known to be populated (after ``make all``)
    to turn absence back into a failure.
    """
    art = Artifacts(root / "outputs")
    specs = build_specs() if specs is None else specs
    docs = sorted({s.doc for s in specs})

    text: dict[str, str] = {}
    tables: dict[str, list[Table]] = {}
    report = Report()
    for doc in docs:
        path = root / doc
        if not path.exists():
            report.findings.append(
                Finding(ERROR, doc, 0, "doc", "file", "the file to exist", "missing")
            )
            text[doc], tables[doc] = "", []
            continue
        text[doc] = path.read_text()
        tables[doc] = parse_tables(text[doc])

    for spec in specs:
        report.specs_run += 1
        try:
            if isinstance(spec, TableSpec):
                findings, checked = run_table_spec(spec, art, tables)
            else:
                findings, checked = run_prose_spec(spec, art, text)
            report.findings.extend(findings)
            report.checks_run += checked
        except MissingArtifact as exc:
            report.skipped.append((spec.id, exc.name))
            if require_artifacts:
                report.findings.append(
                    Finding(ERROR, spec.doc, 0, spec.id, "artifact", exc.name, "missing")
                )
        except (KeyError, StopIteration, IndexError, ValueError) as exc:
            report.findings.append(
                Finding(ERROR, spec.doc, 0, spec.id, "spec failed", "a resolvable value", repr(exc))
            )

    report.findings.extend(audit_artifacts(art))
    if not report.skipped:
        # Only meaningful when every spec actually ran; a sparse outputs/ makes the totals
        # legitimately smaller than what the docs claim.
        report.findings.extend(self_check(report, text))
    report.artifacts_used = set(art.used)
    report.findings.sort(key=lambda f: (f.severity != ERROR, f.doc, f.line, f.spec))
    return report


def render(report: Report, strict: bool, verbose: bool) -> Iterator[str]:
    if report.errors:
        yield "DRIFT — docs disagree with outputs/"
        yield ""
        for f in report.errors:
            yield f"  {f.location}  [{f.spec}] {f.what}"
            yield f"      expected {f.expected}   actual {f.actual}"
        yield ""
    if report.warnings:
        yield "warnings"
        yield ""
        for f in report.warnings:
            suffix = f" — expected {f.expected}, actual {f.actual}" if f.line else ""
            location = f.location if f.line else f.doc
            yield f"  {location}  [{f.spec}] {f.what}{suffix}"
        yield ""
    if report.skipped:
        absent = sorted({artifact for _, artifact in report.skipped})
        yield f"skipped {len(report.skipped)} specs — outputs/ is missing {len(absent)} artifacts"
        yield f"  {', '.join(absent)}"
        yield "  run `make all` to regenerate them, then re-run this check"
        yield ""
    if verbose:
        yield f"artifacts read: {', '.join(sorted(report.artifacts_used))}"
        yield ""

    counts = (
        f"{report.checks_run} numbers across {report.specs_run - len(report.skipped)} specs, "
        f"{len(report.errors)} drift, {len(report.warnings)} warnings"
    )
    if report.skipped:
        counts += f", {len(report.skipped)} skipped"
    if report.failed(strict):
        yield f"FAIL — {counts}"
    else:
        yield f"OK — {counts}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.validate_docs",
        description="check README/docs numbers against the artifacts in outputs/",
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings (incl. uncovered artifacts) as errors"
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="fail instead of skipping when outputs/ is missing an artifact a spec needs",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="list artifacts consulted")
    args = parser.parse_args(argv)

    report = validate(args.root, require_artifacts=args.require_artifacts)
    for line in render(report, args.strict, args.verbose):
        print(line)
    return 1 if report.failed(args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
