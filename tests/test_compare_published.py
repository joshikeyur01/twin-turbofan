"""Tests for the published-baseline comparison.

Two kinds of assertion here, and the split is deliberate.

The **arithmetic** — CI plumbing, per-engine normalisation, gap signs — is pinned exactly,
because it is the part that can be wrong silently. The report's whole argument rests on the
claim that a raw PHM comparison across differently sized test sets is a unit error; a bug in
the divisor would make the report state the error it exists to correct.

The **literature values** are not pinned to their digits, only to their internal consistency
and their confidence labelling. Asserting `Zheng et al. == 16.14` would only prove the test
and the module were typed by the same hand on the same day. What is worth guarding is the
property the report depends on: that no row claims to be verified, since none was.
"""

import json
import pathlib

import pytest

from src import compare_published as cp


@pytest.fixture
def rows():
    """Three seeds of two models, with the spread deliberately unequal.

    ATTENTION gets one bad seed so its interval is wide; GRU is tight. The report's headline
    finding is that those two land on different sides of the published bar, so the fixture has
    to be able to express that.
    """
    return [
        {"model": "ATTENTION", "subset": "FD001", "seed": s, "rmse": r, "phm": p, "n_params": 1000}
        for s, r, p in [(42, 6.0, 30.0), (43, 12.0, 100.0), (44, 6.0, 32.0)]
    ] + [
        {"model": "GRU", "subset": "FD001", "seed": s, "rmse": r, "phm": p, "n_params": 500}
        for s, r, p in [(42, 7.0, 44.0), (43, 7.2, 46.0), (44, 6.8, 42.0)]
    ]


class TestBaselines:
    def test_every_row_is_labelled_unverified(self):
        """Nothing was fetched, so nothing may claim to be verified."""
        assert {e.confidence for e in cp.PUBLISHED} <= {"high", "medium"}

    def test_scored_rows_carry_both_metrics(self):
        """A row with an RMSE but no score would silently skew every per-engine comparison."""
        for entry in cp.scored():
            assert entry.rmse is not None and entry.phm is not None

    def test_reference_only_row_is_excluded_from_comparisons(self):
        """Saxena et al. defines the score; it does not supply a comparable FD001 result."""
        assert any(e.family == "reference" for e in cp.PUBLISHED)
        assert all(e.family != "reference" for e in cp.scored())

    def test_published_scores_are_sums_over_the_full_fd001_test_split(self):
        assert all(e.n_test_engines == cp.FD001_TEST_ENGINES for e in cp.PUBLISHED)

    def test_per_engine_normalisation_divides_by_the_engine_count(self):
        entry = next(e for e in cp.scored() if e.phm is not None)
        assert entry.phm_per_engine == pytest.approx(entry.phm / entry.n_test_engines, abs=5e-4)

    def test_best_published_is_the_lowest_rmse(self):
        assert cp.best_published().rmse == min(e.rmse for e in cp.scored())

    def test_no_parameter_counts_are_available(self):
        """The report says the parameter delta cannot be computed — this is why."""
        assert all(e.n_params is None for e in cp.PUBLISHED)

    def test_every_reference_marker_resolves_to_the_header_comment(self):
        """A `[8]` with no matching citation would look sourced without being sourced."""
        text = pathlib.Path(cp.__file__).read_text()
        for entry in cp.PUBLISHED:
            assert f"#   {entry.ref} " in text, f"{entry.ref} is cited but never defined"


class TestCollect:
    def test_aggregates_every_seed(self, rows):
        ours = cp.collect(rows, "GRU", "FD001", n_test_engines=50)
        assert ours.seeds == [42, 43, 44]
        assert ours.rmse.n == 3
        assert ours.rmse.mean == pytest.approx(7.0)

    def test_model_match_is_case_insensitive(self, rows):
        assert cp.collect(rows, "gru", "FD001", 50).model == "GRU"

    def test_phm_is_normalised_by_our_own_engine_count(self, rows):
        """The divisor must be *our* 50, not the published 100 — the report's core claim."""
        ours = cp.collect(rows, "GRU", "FD001", n_test_engines=50)
        assert ours.phm_per_engine == pytest.approx(44.0 / 50)

    def test_unknown_model_names_the_available_ones(self, rows):
        with pytest.raises(SystemExit, match="ATTENTION"):
            cp.collect(rows, "TRANSFORMER", "FD001", 50)

    def test_single_seed_reports_no_interval(self):
        """One run carries no information about its own spread; `±0` would be a lie."""
        one = [{"model": "GRU", "subset": "FD001", "seed": 42, "rmse": 7.0, "phm": 44.0}]
        ours = cp.collect(one, "GRU", "FD001", 50)
        assert ours.rmse.half is None
        assert ours.span("rmse") == "n/a (1 seed)"


class TestDeltas:
    def test_gap_is_positive_when_we_are_lower(self, rows):
        ours = cp.collect(rows, "GRU", "FD001", 50)
        assert all(d["rmse_gap_pct"] > 0 for d in cp.deltas(ours))

    def test_gap_is_negative_when_we_are_worse(self, rows):
        """Sign convention has to survive the case the synthetic data currently hides."""
        worse = [dict(r, rmse=99.0) for r in rows]
        ours = cp.collect(worse, "GRU", "FD001", 50)
        assert all(d["rmse_gap_pct"] < 0 for d in cp.deltas(ours))

    def test_raw_and_per_engine_phm_gaps_disagree(self, rows):
        """If these two ever agreed, the normalisation would not be doing anything."""
        ours = cp.collect(rows, "GRU", "FD001", 50)
        row = cp.deltas(ours)[0]
        raw_pct = (row["pub_phm"] - row["our_phm"]) / row["pub_phm"] * 100
        assert row["phm_gap_pct_per_engine"] < raw_pct

    def test_wide_interval_fails_baselines_a_point_estimate_would_beat(self, rows):
        """The finding §4 leads with: the mean beats a baseline the CI does not."""
        ours = cp.collect(rows, "ATTENTION", "FD001", 50)
        best = cp.best_published()
        assert ours.rmse.mean < best.rmse  # the point estimate wins
        unresolved = [d for d in cp.deltas(ours) if d["beaten_at_ci_upper"] is False]
        assert unresolved, "a 6.0-12.0 spread must not clear the whole published table"

    def test_tight_interval_clears_the_table(self, rows):
        ours = cp.collect(rows, "GRU", "FD001", 50)
        assert all(d["beaten_at_ci_upper"] for d in cp.deltas(ours))


class TestRendering:
    def test_report_marks_synthetic_data_before_any_number(self, rows):
        ours = [cp.collect(rows, "ATTENTION", "FD001", 50)]
        text = "\n".join(cp.render(ours, cp.deltas(ours[0]), real=False))
        assert "does not establish that the models here are competitive" in text
        assert text.index("SYNTHETIC") < text.index("| SVR")

    def test_report_states_the_engine_count_mismatch(self, rows):
        ours = [cp.collect(rows, "ATTENTION", "FD001", 50)]
        text = "\n".join(cp.render(ours, cp.deltas(ours[0]), real=False))
        assert "sum" in text and "100 test engines" in text

    def test_brief_divergences_are_reported_not_silently_overwritten(self, rows):
        ours = [cp.collect(rows, "GRU", "FD001", 50)]
        text = "\n".join(cp.render(ours, cp.deltas(ours[0]), real=True))
        for label in cp.BRIEF_ESTIMATES:
            assert label in text

    def test_single_seed_report_refuses_to_rank(self):
        one = [{"model": "GRU", "subset": "FD001", "seed": 42, "rmse": 7.0, "phm": 44.0}]
        ours = [cp.collect(one, "GRU", "FD001", 50)]
        text = "\n".join(cp.render(ours, cp.deltas(ours[0]), real=False))
        assert "no interval and no claim" in text

    def test_tables_are_rectangular(self, rows):
        """A ragged markdown table renders as garbage rather than failing loudly."""
        ours = [cp.collect(rows, "ATTENTION", "FD001", 50)]
        lines = cp.render(ours, cp.deltas(ours[0]), real=False)
        width = None
        for line in lines:
            if not line.startswith("|"):
                width = None
                continue
            cells = line.count("|")
            if width is None:
                width = cells
            assert cells == width, f"ragged table row: {line}"


class TestEntryPoint:
    @pytest.fixture
    def redirected(self, tmp_path, monkeypatch, rows):
        out = tmp_path / "outputs"
        out.mkdir()
        (out / "comparison.json").write_text(json.dumps(rows))
        monkeypatch.setattr(cp, "OUTPUTS_DIR", out)
        return out

    def test_writes_both_artifacts(self, redirected):
        assert cp.main([]) == 0
        assert (redirected / "published_comparison.md").exists()
        assert (redirected / "published_comparison.json").exists()

    def test_payload_records_how_the_interval_was_built(self, redirected):
        cp.main([])
        payload = json.loads((redirected / "published_comparison.json").read_text())
        assert payload["ci"]["n"] == 3
        assert payload["ci"]["t"] == pytest.approx(4.303)
        assert payload["published_test_engines"] == 100
        assert payload["n_test_engines"] == 50

    def test_lead_model_is_first_and_the_also_model_follows(self, redirected):
        cp.main(["--model", "GRU", "--also", "ATTENTION"])
        payload = json.loads((redirected / "published_comparison.json").read_text())
        assert [o["model"] for o in payload["ours"]] == ["GRU", "ATTENTION"]

    def test_all_models_includes_every_row(self, redirected):
        cp.main(["--all-models"])
        payload = json.loads((redirected / "published_comparison.json").read_text())
        assert {o["model"] for o in payload["ours"]} == {"ATTENTION", "GRU"}

    def test_missing_comparison_is_a_clear_exit_not_a_traceback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cp, "OUTPUTS_DIR", tmp_path)
        with pytest.raises(SystemExit, match="src.compare"):
            cp.main([])

    def test_engine_count_falls_back_when_metrics_is_unreadable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cp, "OUTPUTS_DIR", tmp_path)
        (tmp_path / "metrics.json").write_text("{}")
        assert cp.test_engine_count() == 50

    def test_engine_count_reads_metrics_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cp, "OUTPUTS_DIR", tmp_path)
        (tmp_path / "metrics.json").write_text(json.dumps({"n_test_engines": 100}))
        assert cp.test_engine_count() == 100


class TestEmbeddedSection:
    def test_section_carries_the_caveat_with_the_number(self, tmp_path, monkeypatch, rows):
        """`--published` puts this in comparison.md, where the caveat must travel with it."""
        monkeypatch.setattr(cp, "OUTPUTS_DIR", tmp_path)
        (tmp_path / "comparison.json").write_text(json.dumps(rows))
        text = "\n".join(cp.render_section("ATTENTION", "FD001"))
        assert "The gap is not a result" in text
        assert "recalled from memory, not verified" in text
