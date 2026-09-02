"""Smoke tests for the torch-dependent entry points.

Covers `train_seq`, `model_lstm`, `sweep`, `compare` and `make_demo_gif` — all of which
sat at 0% coverage, so a crash in any of them would only appear when a human ran the
command. Everything here uses one epoch on the miniature fixture and forces `device=cpu`:
the point is that the scripts execute and emit their artifacts, not that a model trained
on three engines learns anything.

Separated from `test_pipeline_smoke.py` so the torch-free suite (and the no-torch CI job)
stays complete on its own.
"""

import json

import pytest

pytest.importorskip("torch", reason="sequence pipeline needs torch")

from src import compare, make_demo_gif, model_lstm, sweep, train_baseline, train_seq  # noqa: E402


@pytest.fixture
def seq_outputs(cmapss_dir, tmp_path, monkeypatch):
    """Redirect every torch-script module's OUTPUTS_DIR at a temp dir."""
    out = tmp_path / "outputs"
    out.mkdir()
    for mod in (train_seq, model_lstm, sweep, compare, train_baseline):
        if hasattr(mod, "OUTPUTS_DIR"):
            monkeypatch.setattr(mod, "OUTPUTS_DIR", out)
    monkeypatch.setattr(make_demo_gif, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr("src.telemetry.OUTPUTS_DIR", out)
    monkeypatch.setattr("src.online.SPEC_PATH", out / "feature_spec.json")
    return out


class TestTrainSeq:
    @pytest.mark.parametrize("arch", ["lstm", "gru", "cnn"])
    def test_each_arch_trains_and_scores(self, seq_outputs, arch):
        model, result, history = train_seq.train(
            arch=arch, seq_len=6, hidden=8, epochs=1, batch=32, device="cpu", quiet=True
        )
        assert result["arch"] == arch
        assert result["rmse"] >= 0 and result["phm"] >= 0
        assert result["n_params"] > 0
        assert len(history) == 1
        assert result["device"] == "cpu"

    def test_early_stopping_obeys_its_own_rule(self, seq_outputs):
        """Assert the stopping *invariant*, not that stopping happens.

        On three fixture engines validation can improve every epoch, so a test demanding
        an early stop would be testing the fixture rather than the rule. What must always
        hold: the run never exceeds the cap, and if it ended short then it did so because
        `patience` epochs passed with no improvement.
        """
        patience, cap = 2, 12
        _, result, history = train_seq.train(
            arch="cnn",
            seq_len=6,
            hidden=8,
            epochs=cap,
            patience=patience,
            batch=32,
            device="cpu",
            quiet=True,
        )
        assert result["epochs_run"] == len(history) <= cap
        assert 1 <= result["best_epoch"] <= result["epochs_run"]
        if result["epochs_run"] < cap:
            assert result["epochs_run"] - result["best_epoch"] >= patience

    def test_result_reports_the_best_epoch_not_the_last(self, seq_outputs):
        _, result, history = train_seq.train(
            arch="gru",
            seq_len=6,
            hidden=8,
            epochs=6,
            patience=6,
            batch=32,
            device="cpu",
            quiet=True,
        )
        best = min(h["val_rmse"] for h in history)
        assert result["val_rmse"] == pytest.approx(round(best, 3))

    def test_seed_makes_the_run_reproducible(self, seq_outputs):
        kw = {
            "arch": "cnn",
            "seq_len": 6,
            "hidden": 8,
            "epochs": 1,
            "batch": 32,
            "device": "cpu",
            "quiet": True,
            "seed": 7,
        }
        a = train_seq.train(**kw)[1]
        b = train_seq.train(**kw)[1]
        assert a["rmse"] == b["rmse"]
        assert a["val_rmse"] == b["val_rmse"]


class TestModelLSTM:
    def test_entry_point_trains_evaluates_and_saves(self, seq_outputs):
        """The scaffold used to train without evaluating; it must now return metrics."""
        result = model_lstm.main(subset="FD001", seq_len=6, epochs=1, batch=32)
        assert (seq_outputs / "lstm_rul.pt").exists()
        assert {"rmse", "phm", "arch"} <= set(result)
        assert result["arch"] == "lstm"

    def test_backwards_compatible_reexports(self):
        """Pre-harness import paths must keep working."""
        assert model_lstm.LSTMRegressor is not None
        assert model_lstm.WindowDataset is not None


class TestSweep:
    def test_writes_report_ranked_by_validation(self, seq_outputs, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "sweep",
                "--arch",
                "cnn",
                "--seq-lens",
                "6",
                "--hiddens",
                "8",
                "--lrs",
                "1e-3",
                "3e-3",
                "--epochs",
                "1",
            ],
        )
        sweep.main()

        rows = json.loads((seq_outputs / "sweep_cnn.json").read_text())
        assert len(rows) == 2
        report = (seq_outputs / "sweep_cnn.md").read_text()
        assert "Selected configuration" in report
        # The selected row must be the best *validation* row, not the best test row.
        best_val = min(rows, key=lambda r: r["val_rmse"])
        assert f"lr={best_val['lr']:g}" in report


class TestCompare:
    def test_builds_the_model_dataset_table(self, seq_outputs, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["compare", "--archs", "cnn", "--epochs", "1", "--seq-len", "6", "--hidden", "8"],
        )
        compare.main()

        rows = json.loads((seq_outputs / "comparison.json").read_text())
        models_seen = {r["model"] for r in rows}
        assert "RandomForest" in models_seen and "CNN" in models_seen

        report = (seq_outputs / "comparison.md").read_text()
        assert "RMSE" in report and "PHM" in report
        # The report must always state its data provenance. The fixture writes into
        # data/CMAPSSData/, so `using_real_data()` legitimately reports "real" here —
        # the synthetic-caveat branch is covered by test_report_flags_synthetic_data.
        assert "**Data:" in report
        assert "Protocol:" in report

    def test_report_flags_synthetic_data(self, seq_outputs, tmp_path, monkeypatch):
        """The synthetic caveat must appear when the real files are absent."""
        monkeypatch.setattr("src.data_loader.using_real_data", lambda: False)
        monkeypatch.setattr(compare, "using_real_data", lambda: False)
        monkeypatch.setattr(
            "sys.argv",
            [
                "compare",
                "--archs",
                "cnn",
                "--epochs",
                "1",
                "--seq-len",
                "6",
                "--hidden",
                "8",
                "--skip-rf",
            ],
        )
        compare.main()
        report = (seq_outputs / "comparison.md").read_text()
        assert "SYNTHETIC" in report
        assert "not** benchmark results" in report

    def test_skip_rf_omits_the_baseline_row(self, seq_outputs, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "compare",
                "--archs",
                "cnn",
                "--epochs",
                "1",
                "--seq-len",
                "6",
                "--hidden",
                "8",
                "--skip-rf",
            ],
        )
        compare.main()
        rows = json.loads((seq_outputs / "comparison.json").read_text())
        assert {r["model"] for r in rows} == {"CNN"}


class TestDemoGif:
    def test_renders_a_multi_frame_gif(self, seq_outputs, tmp_path):
        from PIL import Image

        train_baseline.main("FD001")
        out = make_demo_gif.build(unit=1, frames=4, fps=4)
        assert out.exists() and out.stat().st_size > 0

        with Image.open(out) as im:
            assert im.n_frames > 1, "a single-frame GIF is not an animation"

    def test_collect_returns_aligned_series(self, seq_outputs):
        train_baseline.main("FD001")
        cycles, preds, trues, _ = make_demo_gif.collect(1, "FD001", 25.0)
        assert len(cycles) == len(preds) == len(trues)
        assert (preds >= 0).all(), "predictions must be clipped at zero"


class TestEnsemble:
    def test_writes_report_with_both_endpoints(self, seq_outputs, monkeypatch):
        from src import ensemble

        monkeypatch.setattr(ensemble, "OUTPUTS_DIR", seq_outputs)
        monkeypatch.setattr(
            "sys.argv",
            [
                "ensemble",
                "--arch",
                "cnn",
                "--seq-len",
                "6",
                "--hidden",
                "8",
                "--epochs",
                "1",
                "--val-frac",
                "0.34",
            ],
        )
        ensemble.main()

        payload = json.loads((seq_outputs / "ensemble.json").read_text())
        weights = [r["w_seq"] for r in payload["rows"]]
        # Both endpoints must be present, else "did blending help?" is unanswerable.
        assert 0.0 in weights and 1.0 in weights
        assert payload["chosen_w"] in weights

        report = (seq_outputs / "ensemble.md").read_text()
        assert "Read-out" in report
        assert "never on test" in report

    def test_weight_is_chosen_on_validation_not_test(self, seq_outputs, monkeypatch):
        """The selected weight must minimise val_rmse, whatever test says."""
        from src import ensemble

        monkeypatch.setattr(ensemble, "OUTPUTS_DIR", seq_outputs)
        monkeypatch.setattr(
            "sys.argv",
            [
                "ensemble",
                "--arch",
                "cnn",
                "--seq-len",
                "6",
                "--hidden",
                "8",
                "--epochs",
                "1",
                "--val-frac",
                "0.34",
            ],
        )
        ensemble.main()

        payload = json.loads((seq_outputs / "ensemble.json").read_text())
        rows = payload["rows"]
        best_val = min(rows, key=lambda r: r["val_rmse"])
        assert payload["chosen_w"] == best_val["w_seq"]


class TestSweptConfigSelection:
    """Pure selection logic — no training, so it stays in the fast suite."""

    def test_returns_none_without_a_sweep_file(self, seq_outputs, monkeypatch):
        from src import compare

        monkeypatch.setattr(compare, "OUTPUTS_DIR", seq_outputs)
        assert compare.swept_config("lstm") is None

    def test_picks_the_best_validation_row(self, seq_outputs, monkeypatch):
        from src import compare

        monkeypatch.setattr(compare, "OUTPUTS_DIR", seq_outputs)
        (seq_outputs / "sweep_cnn.json").write_text(
            json.dumps(
                [
                    {"seq_len": 20, "hidden": 32, "lr": 3e-4, "val_rmse": 10.7, "rmse": 99.0},
                    {"seq_len": 50, "hidden": 128, "lr": 1e-3, "val_rmse": 12.0, "rmse": 1.0},
                ]
            )
        )
        # The second row has the far better TEST score; selection must ignore that.
        cfg, source = compare.swept_config("cnn")
        assert cfg == {"seq_len": 20, "hidden": 32, "lr": 3e-4}
        assert "sweep_cnn.json" in source and "single seed" in source

    def test_empty_sweep_file_is_handled(self, seq_outputs, monkeypatch):
        from src import compare

        monkeypatch.setattr(compare, "OUTPUTS_DIR", seq_outputs)
        (seq_outputs / "sweep_gru.json").write_text("[]")
        assert compare.swept_config("gru") is None


@pytest.mark.slow
class TestSweptConfigEndToEnd:
    """Drives compare.main(), which trains — hence slow."""

    def test_comparison_records_where_the_config_came_from(self, seq_outputs, monkeypatch):
        from src import compare

        monkeypatch.setattr(compare, "OUTPUTS_DIR", seq_outputs)
        (seq_outputs / "sweep_cnn.json").write_text(
            json.dumps([{"seq_len": 6, "hidden": 8, "lr": 1e-3, "val_rmse": 1.0}])
        )
        monkeypatch.setattr("sys.argv", ["compare", "--archs", "cnn", "--epochs", "1", "--skip-rf"])
        compare.main()

        rows = json.loads((seq_outputs / "comparison.json").read_text())
        row = next(r for r in rows if r["model"] == "CNN")
        # The "(single seed)" qualifier is load-bearing, not decoration: sweep.py ranks on
        # one seed and rerank.py showed that ranking is mostly noise, so the provenance has
        # to carry the caveat wherever the config is reported.
        assert row["config_source"] == "sweep_cnn.json (single seed)"
        assert row["seq_len"] == 6 and row["hidden"] == 8

    def test_shared_config_flag_ignores_the_sweep(self, seq_outputs, monkeypatch):
        from src import compare

        monkeypatch.setattr(compare, "OUTPUTS_DIR", seq_outputs)
        (seq_outputs / "sweep_cnn.json").write_text(
            json.dumps([{"seq_len": 6, "hidden": 8, "lr": 1e-3, "val_rmse": 1.0}])
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "compare",
                "--archs",
                "cnn",
                "--epochs",
                "1",
                "--skip-rf",
                "--shared-config",
                "--seq-len",
                "7",
                "--hidden",
                "16",
            ],
        )
        compare.main()

        rows = json.loads((seq_outputs / "comparison.json").read_text())
        row = next(r for r in rows if r["model"] == "CNN")
        assert row["config_source"] == "shared (never swept)"
        assert row["seq_len"] == 7 and row["hidden"] == 16


@pytest.mark.slow
class TestVariance:
    def test_same_seed_runs_are_identical(self, seq_outputs, monkeypatch):
        """Measured property, now asserted: seeding pins this workload exactly.

        If a future torch/device change breaks it, this fails and the variance report's
        conclusion (that same-seed spread is 0) stops being quietly wrong.
        """
        from src import variance

        monkeypatch.setattr(variance, "OUTPUTS_DIR", seq_outputs)
        monkeypatch.setattr(
            "sys.argv",
            [
                "variance",
                "--archs",
                "cnn",
                "--repeats",
                "2",
                "--seq-len",
                "6",
                "--hidden",
                "8",
                "--epochs",
                "2",
            ],
        )
        variance.main()

        payload = json.loads((seq_outputs / "variance.json").read_text())
        same = payload["cnn"]["same_seed"]
        assert same["rmse"]["spread"] == 0.0
        assert same["phm"]["spread"] == 0.0

    def test_records_seeds_and_writes_artifacts(self, seq_outputs, monkeypatch):
        from src import variance

        monkeypatch.setattr(variance, "OUTPUTS_DIR", seq_outputs)
        monkeypatch.setattr(
            "sys.argv",
            [
                "variance",
                "--archs",
                "cnn",
                "--repeats",
                "2",
                "--seq-len",
                "6",
                "--hidden",
                "8",
                "--epochs",
                "2",
                "--base-seed",
                "11",
            ],
        )
        variance.main()

        assert (seq_outputs / "variance.png").exists()
        payload = json.loads((seq_outputs / "variance.json").read_text())
        assert [r["seed"] for r in payload["cnn"]["same_seed"]["runs"]] == [11, 11]
        assert [r["seed"] for r in payload["cnn"]["different_seeds"]["runs"]] == [11, 12]
        assert "Verdict" in (seq_outputs / "variance.md").read_text()


class TestRerankFinalists:
    """`src/rerank.py` asks whether a sweep's winner survives re-seeding."""

    def _write_sweep(self, out, arch, rows):
        (out / f"sweep_{arch}.json").write_text(json.dumps(rows))

    def test_load_finalists_orders_by_validation(self, seq_outputs, monkeypatch):
        from src import rerank

        monkeypatch.setattr(rerank, "OUTPUTS_DIR", seq_outputs)
        self._write_sweep(
            seq_outputs,
            "cnn",
            [
                {"seq_len": 50, "hidden": 128, "lr": 1e-3, "val_rmse": 12.0},
                {"seq_len": 20, "hidden": 32, "lr": 3e-4, "val_rmse": 10.7},
                {"seq_len": 30, "hidden": 32, "lr": 1e-3, "val_rmse": 10.8},
            ],
        )
        top = rerank.load_finalists("cnn", 2)
        assert [r["val_rmse"] for r in top] == [10.7, 10.8]

    def test_missing_sweep_gives_actionable_error(self, seq_outputs, monkeypatch):
        from src import rerank

        monkeypatch.setattr(rerank, "OUTPUTS_DIR", seq_outputs)
        with pytest.raises(SystemExit, match="python -m src.sweep"):
            rerank.load_finalists("lstm", 3)

    def test_empty_sweep_gives_actionable_error(self, seq_outputs, monkeypatch):
        from src import rerank

        monkeypatch.setattr(rerank, "OUTPUTS_DIR", seq_outputs)
        self._write_sweep(seq_outputs, "gru", [])
        with pytest.raises(SystemExit, match="empty"):
            rerank.load_finalists("gru", 3)

    def test_top_is_at_least_one(self, seq_outputs, monkeypatch):
        """--top 0 must not silently produce an empty re-rank."""
        from src import rerank

        monkeypatch.setattr(rerank, "OUTPUTS_DIR", seq_outputs)
        self._write_sweep(
            seq_outputs, "cnn", [{"seq_len": 6, "hidden": 8, "lr": 1e-3, "val_rmse": 1.0}]
        )
        assert len(rerank.load_finalists("cnn", 0)) == 1

    @pytest.mark.slow
    def test_end_to_end_reranks_and_reports(self, seq_outputs, monkeypatch):
        from src import rerank

        monkeypatch.setattr(rerank, "OUTPUTS_DIR", seq_outputs)
        self._write_sweep(
            seq_outputs,
            "cnn",
            [
                {"seq_len": 6, "hidden": 8, "lr": 1e-3, "val_rmse": 1.0},
                {"seq_len": 6, "hidden": 8, "lr": 3e-3, "val_rmse": 1.1},
            ],
        )
        monkeypatch.setattr(
            "sys.argv",
            ["rerank", "--arch", "cnn", "--top", "2", "--seeds", "2", "--epochs", "1"],
        )
        rerank.main()

        rows = json.loads((seq_outputs / "rerank_cnn.json").read_text())
        assert len(rows) == 2
        for r in rows:
            # each finalist must actually be re-run at every seed, not reused
            assert r["seeds"] == [42, 43]
            assert r["val_range"] >= 0
        assert "Verdict" in (seq_outputs / "rerank_cnn.md").read_text()


class TestSeedAveragedSelectionWins:
    """A seed-averaged ranking must override a single-seed sweep.

    `src/rerank.py` showed the sweep's single-seed ordering is noise — for the GRU the
    seed-averaged winner was a different learning rate whose 3-seed test RMSE was 6.820
    against the single-seed pick's 8.312. So when both files exist, the rerank wins.
    """

    def test_rerank_is_preferred_over_sweep(self, seq_outputs, monkeypatch):
        from src import compare

        monkeypatch.setattr(compare, "OUTPUTS_DIR", seq_outputs)
        (seq_outputs / "sweep_gru.json").write_text(
            json.dumps([{"seq_len": 50, "hidden": 128, "lr": 3e-4, "val_rmse": 6.865}])
        )
        (seq_outputs / "rerank_gru.json").write_text(
            json.dumps(
                [
                    {"seq_len": 50, "hidden": 128, "lr": 3e-4, "val_mean": 7.222},
                    {"seq_len": 50, "hidden": 128, "lr": 1e-3, "val_mean": 6.033},
                ]
            )
        )
        cfg, source = compare.swept_config("gru")
        assert cfg["lr"] == 1e-3, "should take the seed-averaged winner, not the sweep's"
        assert "seed-averaged" in source

    def test_falls_back_to_sweep_without_a_rerank(self, seq_outputs, monkeypatch):
        from src import compare

        monkeypatch.setattr(compare, "OUTPUTS_DIR", seq_outputs)
        (seq_outputs / "sweep_lstm.json").write_text(
            json.dumps([{"seq_len": 50, "hidden": 128, "lr": 3e-4, "val_rmse": 7.8}])
        )
        cfg, source = compare.swept_config("lstm")
        assert cfg["lr"] == 3e-4
        assert "single seed" in source

    def test_empty_rerank_falls_through_to_sweep(self, seq_outputs, monkeypatch):
        from src import compare

        monkeypatch.setattr(compare, "OUTPUTS_DIR", seq_outputs)
        (seq_outputs / "rerank_cnn.json").write_text("[]")
        (seq_outputs / "sweep_cnn.json").write_text(
            json.dumps([{"seq_len": 20, "hidden": 32, "lr": 3e-4, "val_rmse": 10.7}])
        )
        cfg, source = compare.swept_config("cnn")
        assert cfg["hidden"] == 32
        assert "single seed" in source

    def test_rerank_ranks_by_mean_not_single_seed(self, seq_outputs, monkeypatch):
        """The rerank file also carries the single-seed value; it must be ignored."""
        from src import compare

        monkeypatch.setattr(compare, "OUTPUTS_DIR", seq_outputs)
        (seq_outputs / "rerank_cnn.json").write_text(
            json.dumps(
                [
                    {"seq_len": 20, "hidden": 32, "lr": 3e-4, "val_single": 1.0, "val_mean": 9.0},
                    {"seq_len": 20, "hidden": 64, "lr": 3e-4, "val_single": 99.0, "val_mean": 8.0},
                ]
            )
        )
        cfg, _ = compare.swept_config("cnn")
        assert cfg["hidden"] == 64
