"""Tests for the doc/artifact consistency checker.

Two things need pinning down here, and they pull in opposite directions.

The first is that the checker *catches* drift: a stale number must produce a finding at
the right line. That much is obvious.

The second matters more. A checker like this fails silently when it stops checking —
an anchor that no longer matches a table, a row key that matches nothing, a regex that
matches nothing — and a silent checker is worse than none, because it looks like a
passing guard. So several tests assert on the *count* of comparisons rather than only on
the findings, and the whole claim registry is exercised against the real repo to prove
every spec still resolves against the real artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.validate_docs import (
    ERROR,
    OK,
    WARN,
    Artifacts,
    BinUpper,
    Config,
    Label,
    MeanSpread,
    NumCells,
    ProseSpec,
    Ratio,
    Row,
    TableSpec,
    Tokens,
    audit_artifacts,
    build_specs,
    clean,
    compare,
    compare_number,
    decimals,
    mean_spread,
    parse_config,
    parse_tables,
    run_prose_spec,
    run_table_spec,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]


def row(*cells: str) -> Row:
    return Row(cells=tuple(cells), line=1)


class TestCellCleaning:
    def test_strips_emphasis_and_ticks(self):
        assert clean("**6.820**") == "6.820"
        assert clean("`s4_rmean`") == "s4_rmean"

    def test_drops_parenthetical_annotations(self):
        """``+ rolling w=5 *(default)*`` names the same arm as ``+ rolling w=5``."""
        assert clean("+ rolling w=5 *(default)*") == "+ rolling w=5"
        assert clean("**1.0 *(GRU only — selected)***") == "1.0"

    def test_normalises_unicode_minus(self):
        """Residual columns use a real minus sign, which ``float()`` will not parse."""
        assert clean("−1.53") == "-1.53"

    def test_drops_direction_arrows(self):
        assert clean("RMSE ↓") == "RMSE"


class TestNumberPrecision:
    @pytest.mark.parametrize(
        ("literal", "expected"), [("10.309", 3), ("85.0", 1), ("50", 0), ("3e-4", 0)]
    )
    def test_decimals_counts_printed_digits(self, literal, expected):
        assert decimals(literal) == expected

    def test_exact_at_printed_precision_is_ok(self):
        """The doc prints 3 decimals, so it is claiming the value only to ±0.0005."""
        assert compare_number(10.3094, "10.309") == OK

    def test_one_unit_out_is_a_warning_not_drift(self):
        """2.672 vs 2.673 is a rounding path, not a stale number — worth saying, not failing."""
        assert compare_number(2.673, "2.672") == WARN

    def test_further_out_is_drift(self):
        assert compare_number(2.673, "2.680") == ERROR

    def test_precision_scales_the_tolerance(self):
        """A number printed to 1dp claims less, so it is graded against a wider band."""
        assert compare_number(85.04, "85.0") == OK
        assert compare_number(85.1, "85.0") == WARN
        assert compare_number(86.0, "85.0") == ERROR


class TestCompare:
    def test_parameter_counts_are_exact(self):
        """A parameter count has no rounding to forgive; the CNN's drifted by 16k."""
        assert compare(26881, "10,401") == (ERROR, "10401")
        assert compare(26881, "26,881")[0] == OK

    def test_config_survives_either_spelling(self):
        """README writes ``50/128/1e-3``; benchmarks writes ``seq=50 hidden=128 lr=1e-3``."""
        cfg = Config(50, 128, 1e-3)
        assert compare(cfg, "50/128/1e-3")[0] == OK
        assert compare(cfg, "**seq=50 hidden=128** lr=1e-3")[0] == OK
        assert compare(cfg, "50/128/3e-4")[0] == ERROR

    def test_mean_spread_grades_both_halves(self):
        assert compare(MeanSpread(6.820, 1.336), "**6.820** ±1.336")[0] == OK
        assert compare(MeanSpread(6.820, 1.336), "**6.820** ±9.999")[0] == ERROR

    def test_ratio(self):
        assert compare(Ratio(60, 60), "**60 / 60**")[0] == OK
        assert compare(Ratio(60, 60), "21 / 29")[0] == ERROR

    def test_missing_number_is_drift_not_a_crash(self):
        assert compare(1.0, "—")[0] == ERROR


class TestRowKeys:
    """Row identity is where a checker like this silently goes wrong.

    Every case below is a bug this module actually had: prefix matching made ``s2`` match
    the ``s20`` row, ``0.9`` match the ``0.95`` row, and ``0.0`` match every weight in the
    ensemble table — each one comparing a real number against the wrong row's value.
    """

    def test_label_is_exact_not_a_prefix(self):
        assert Label("s2").matches(row("`s2`", "+0.561"))
        assert not Label("s2").matches(row("`s20`", "+2.383"))

    def test_label_ignores_emphasis_and_annotations(self):
        assert Label("GRU").matches(row("**GRU**", "50/128/1e-3"))
        assert Label("+ rolling w=5").matches(row("+ rolling w=5 *(shipped default)*", "63"))

    def test_numcells_compares_numerically(self):
        assert NumCells((0.7,)).matches(row("**0.70**", "+1.86"))
        assert not NumCells((0.9,)).matches(row("0.95", "+22.22"))

    def test_numcells_matches_across_notations(self):
        """The sweep grid writes ``3e-4`` where the JSON holds ``0.0003``."""
        assert NumCells((50, 128, 0.0003)).matches(row("**50**", "**128**", "**3e-4**", "7.841"))
        assert not NumCells((50, 128, 0.001)).matches(row("50", "128", "3e-4", "7.841"))

    def test_bin_upper_keys_on_the_bin_boundary(self):
        """The two docs spell the same bin ``0–25`` and ``(-0.001, 25.0]``."""
        assert BinUpper(25.0).matches(row("0–25 (near failure)", "5"))
        assert BinUpper(25.0).matches(row("(-0.001, 25.0]", "5"))
        assert not BinUpper(25.0).matches(row("25–50", "12"))

    def test_tokens_requires_every_part(self):
        assert Tokens(("GRU", "same seed")).matches(row("GRU", "same seed", "7.292"))
        assert not Tokens(("GRU", "same seed")).matches(row("GRU", "different seeds", "8.312"))


class TestTableParsing:
    DOC = """# Report

## Results

Some preamble text.

| model | RMSE |
|---|---|
| GRU | 6.820 |
| LSTM | 8.302 |

## Other
"""

    def test_finds_the_table_with_1_based_line_numbers(self):
        (table,) = parse_tables(self.DOC)
        assert table.header == ("model", "RMSE")
        assert [r.cells[0] for r in table.rows] == ["GRU", "LSTM"]
        assert self.DOC.splitlines()[table.rows[0].line - 1] == "| GRU | 6.820 |"

    def test_context_carries_heading_and_preamble_for_anchoring(self):
        (table,) = parse_tables(self.DOC)
        assert "## Results" in table.context
        assert "Some preamble text." in table.context

    def test_a_heading_resets_the_preamble(self):
        """Otherwise §4's preamble would anchor §4b's table too."""
        doc = "## A\n\nanchor-a\n\n## B\n\n| x |\n|---|\n| 1 |\n"
        (table,) = parse_tables(doc)
        assert "anchor-a" not in table.context


# --------------------------------------------------------------------------
# fixture: a miniature repo, so spec behaviour is testable without the real one
# --------------------------------------------------------------------------

FIXTURE_DOC = """# Demo

## Results

| model | RMSE ↓ | params |
|---|---|---|
| **GRU** | **6.820** | 177,345 |
| CNN | 11.386 | 10,401 |

The GRU is 40% better than the CNN.
"""

FIXTURE_METRICS = [
    {"model": "GRU", "rmse": 6.820, "n_params": 177345},
    {"model": "CNN", "rmse": 11.386, "n_params": 26881},
]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "demo.json").write_text(json.dumps(FIXTURE_METRICS))
    (tmp_path / "README.md").write_text(FIXTURE_DOC)
    return tmp_path


def demo_rows(art: Artifacts) -> list[dict]:
    return [
        {"key": Label(r["model"]), r"RMSE": float(r["rmse"]), r"params": int(r["n_params"])}
        for r in art.json("demo.json")
    ]


class TestTableSpec:
    def _run(self, repo: Path, spec: TableSpec):
        art = Artifacts(repo / "outputs")
        text = (repo / "README.md").read_text()
        return run_table_spec(spec, art, {"README.md": parse_tables(text)})

    def test_reports_drift_with_the_offending_line(self, repo):
        spec = TableSpec("demo", "README.md", r"## Results", demo_rows)
        findings, checked = self._run(repo, spec)
        assert checked == 4  # two rows x two checked columns
        (finding,) = findings
        assert finding.severity == ERROR
        assert finding.expected == "26881"
        assert finding.actual == "10401"
        assert FIXTURE_DOC.splitlines()[finding.line - 1].startswith("| CNN")

    def test_a_spec_that_checks_nothing_fails(self, repo):
        """The failure mode that matters: an anchor still matches, but no row does.

        Without this the spec reports a clean pass while comparing zero numbers, which is
        indistinguishable from working.
        """

        def unmatchable(art: Artifacts) -> list[dict]:
            return [{"key": Label("TRANSFORMER"), r"RMSE": 1.0}]

        findings, checked = self._run(repo, TableSpec("x", "README.md", r"## Results", unmatchable))
        assert checked == 0
        assert [f.severity for f in findings] == [ERROR]
        assert "no rows" in findings[0].what

    def test_a_missing_table_fails(self, repo):
        findings, checked = self._run(repo, TableSpec("x", "README.md", r"## Nope", demo_rows))
        assert checked == 0
        assert findings[0].severity == ERROR


class TestProseSpec:
    def _run(self, repo: Path, spec: ProseSpec):
        art = Artifacts(repo / "outputs")
        return run_prose_spec(spec, art, {"README.md": (repo / "README.md").read_text()})

    def test_matches_and_grades_a_prose_number(self, repo):
        spec = ProseSpec(
            "gain", "README.md", r"is (?P<value>\d+)% better", lambda a: 40, occurrences=1
        )
        findings, checked = self._run(repo, spec)
        assert checked == 1
        assert findings == []

    def test_a_regex_that_stops_matching_fails(self, repo):
        spec = ProseSpec("gone", "README.md", r"(?P<value>\d+)% faster", lambda a: 40)
        findings, _ = self._run(repo, spec)
        assert findings[0].severity == ERROR
        assert findings[0].actual == "no match"

    def test_a_restated_claim_changing_count_warns(self, repo):
        """Several claims are stated twice on purpose; a new copy would be unguarded."""
        spec = ProseSpec(
            "gain", "README.md", r"is (?P<value>\d+)% better", lambda a: 40, occurrences=2
        )
        findings, _ = self._run(repo, spec)
        assert [f.severity for f in findings] == [WARN]
        assert "restated" in findings[0].what


class TestArtifactCoverage:
    def test_an_unread_artifact_warns(self, repo):
        art = Artifacts(repo / "outputs")
        (findings,) = audit_artifacts(art, unchecked={}, binaries=set())
        assert findings.severity == WARN
        assert "demo.json" in findings.doc

    def test_a_read_artifact_does_not(self, repo):
        art = Artifacts(repo / "outputs")
        art.json("demo.json")
        assert audit_artifacts(art, unchecked={}, binaries=set()) == []

    def test_a_new_artifact_is_flagged_even_though_no_spec_names_it(self, repo):
        """The point of the audit: a new sweep cannot enter outputs/ unnoticed."""
        art = Artifacts(repo / "outputs")
        art.json("demo.json")
        (repo / "outputs" / "sweep_transformer.json").write_text("[]")
        (finding,) = audit_artifacts(art, unchecked={}, binaries=set())
        assert "sweep_transformer.json" in finding.doc


class TestArtifactReader:
    def test_frontmatter_of_a_generated_report(self):
        """The sweep reports carry no YAML; the metadata is prose the generator writes."""
        art = Artifacts(ROOT / "outputs")
        meta = art.md_frontmatter("sweep_lstm.md")
        assert meta["n_configs"] == 27
        assert meta["epochs"] == 60
        assert meta["patience"] == 8
        assert "SYNTHETIC" in meta["data"]

    def test_metrics_bullets_of_a_report_without_json(self):
        """``src/error_analysis.py`` emits no JSON, so its markdown is the only source.

        Checked against ``metrics.json`` rather than a literal: both artifacts come from
        the same run, so a hardcoded number here just goes stale every time the baseline
        is regenerated — which is the failure mode this whole module exists to prevent.
        """
        art = Artifacts(ROOT / "outputs")
        metrics = art.md_metrics("error_analysis.md")
        expected = json.loads((ROOT / "outputs" / "metrics.json").read_text())["rmse"]
        assert metrics["RMSE"] == pytest.approx(expected)
        assert "error_analysis.md" in art.used


class TestDerivations:
    def test_mean_spread_matches_how_compare_renders_it(self):
        assert mean_spread([8.52, 6.094, 5.847]) == MeanSpread(6.820, 1.336)

    def test_parse_config_accepts_both_doc_spellings(self):
        assert parse_config("50/128/1e-3") == Config(50, 128, 1e-3)
        assert parse_config("seq_len=50, hidden=128, lr=3e-4") == Config(50, 128, 3e-4)
        assert parse_config("no config here") is None


@pytest.fixture(scope="module")
def real_report():
    return validate(ROOT, require_artifacts=True)


@pytest.mark.skipif(
    not (ROOT / "outputs" / "comparison.json").exists(),
    reason="outputs/*.json is gitignored; run `make all` to populate it",
)
class TestAgainstTheRealRepo:
    """The registry has to keep working against the artifacts that actually exist.

    These do not assert the docs are *correct* — that is what ``make check-docs`` is for,
    and it is allowed to fail while someone is mid-edit. They assert the checker is still
    able to check: every spec resolves, finds its table or regex, and compares something.
    """

    def test_every_spec_still_resolves(self, real_report):
        """Catches a renamed artifact, a moved heading, or a reworded claim."""
        broken = [
            f
            for f in real_report.findings
            if f.what
            in {"prose", "table", "spec failed", "artifact", "table matched but no rows did"}
        ]
        assert broken == [], "\n".join(f"{f.spec}: {f.expected} -> {f.actual}" for f in broken)

    def test_it_checks_a_substantial_number_of_values(self, real_report):
        """A floor, so the registry cannot quietly shrink to nothing."""
        assert real_report.specs_run == len(build_specs())
        assert real_report.checks_run > 250

    def test_every_json_artifact_is_covered(self, real_report):
        """No artifact in outputs/ carries numbers that nothing validates."""
        uncovered = [f for f in real_report.findings if f.spec == "coverage"]
        assert uncovered == [], "\n".join(f.doc for f in uncovered)
