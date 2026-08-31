"""End-to-end smoke tests for the entry-point scripts.

The property-based suites elsewhere cover the library code thoroughly, but the scripts
that actually produce `outputs/` were at 0% coverage — meaning a crash in report
generation, a renamed column, or a missing artifact would only surface when a human ran
`make all`. These tests run each script against the miniature fixture dataset with
`DATA_DIR` and `OUTPUTS_DIR` redirected to `tmp_path`, so they are fast and touch nothing
real.

They assert the pipeline *runs and writes what it claims to write*. They deliberately do
not assert metric values: on three engines of fixture data any particular RMSE is
meaningless, and pinning it would produce a test that fails whenever the fixture changes
without indicating a real defect.
"""

import json

import numpy as np
import pytest

from src import ablation, error_analysis, evaluate, generate_synthetic, models, train_baseline
from src.models import RidgeFallback, make_model


@pytest.fixture
def redirected(cmapss_dir, tmp_path, monkeypatch):
    """Point every module's DATA_DIR / OUTPUTS_DIR at tmp_path.

    Each module imports these names directly, so patching the definition site is not
    enough — the binding in each importing module has to be replaced too.
    """
    out = tmp_path / "outputs"
    out.mkdir()
    for mod in (train_baseline, error_analysis, ablation, generate_synthetic):
        if hasattr(mod, "OUTPUTS_DIR"):
            monkeypatch.setattr(mod, "OUTPUTS_DIR", out)
    monkeypatch.setattr("src.online.SPEC_PATH", out / "feature_spec.json")
    monkeypatch.setattr("src.generate_synthetic.DATA_DIR", tmp_path / "gen")
    return out


class TestModels:
    def test_make_model_prefers_random_forest(self):
        model, name = make_model()
        assert hasattr(model, "fit") and hasattr(model, "predict")
        assert "RandomForest" in name or "Ridge" in name

    def test_ridge_fallback_fits_and_predicts(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 4))
        y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + 3.0
        model = RidgeFallback(alpha=1e-6).fit(X, y)
        assert model.predict(X).shape == (60,)
        # With negligible regularisation it should recover a near-exact linear fit.
        assert evaluate.rmse(y, model.predict(X)) < 0.1

    def test_ridge_fallback_returns_self_from_fit(self):
        """Sklearn convention — allows RidgeFallback().fit(X, y).predict(X)."""
        X = np.arange(20, dtype=float).reshape(10, 2)
        y = np.arange(10, dtype=float)
        assert isinstance(RidgeFallback().fit(X, y), RidgeFallback)

    def test_ridge_regularisation_shrinks_weights(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(40, 3))
        y = X @ np.array([5.0, 5.0, 5.0])
        weak = RidgeFallback(alpha=1e-6).fit(X, y).w
        strong = RidgeFallback(alpha=1e4).fit(X, y).w
        assert np.abs(strong).sum() < np.abs(weak).sum()

    def test_module_exposes_ridge_for_unpickling(self):
        """stream_demo/telemetry import this name so pickled fallbacks reload."""
        assert models.RidgeFallback is RidgeFallback


class TestGenerateSynthetic:
    def test_writes_cmapss_shaped_files(self, redirected, tmp_path):
        generate_synthetic.main()
        out = tmp_path / "gen" / "synthetic"
        for name in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"):
            assert (out / name).exists(), name

        import pandas as pd

        train = pd.read_csv(out / "train_FD001.txt", sep=r"\s+", header=None)
        assert train.shape[1] == 26, "C-MAPSS layout is 26 columns"
        assert train[0].nunique() == generate_synthetic.N_TRAIN

    def test_rul_file_has_one_row_per_test_engine(self, redirected, tmp_path):
        generate_synthetic.main()
        out = tmp_path / "gen" / "synthetic"

        import pandas as pd

        test = pd.read_csv(out / "test_FD001.txt", sep=r"\s+", header=None)
        rul = pd.read_csv(out / "RUL_FD001.txt", sep=r"\s+", header=None)
        assert len(rul) == test[0].nunique() == generate_synthetic.N_TEST


class TestTrainBaseline:
    def test_writes_model_metrics_and_feature_spec(self, redirected):
        train_baseline.main("FD001")
        assert (redirected / "baseline.pkl").exists()
        assert (redirected / "pred_vs_true.png").exists()

        metrics = json.loads((redirected / "metrics.json").read_text())
        assert set(metrics) >= {"model", "subset", "rmse", "phm_score", "n_test_engines"}
        assert metrics["subset"] == "FD001"
        assert metrics["rmse"] >= 0

        spec = json.loads((redirected / "feature_spec.json").read_text())
        assert len(spec["mean"]) == len(spec["feature_cols"])
        assert spec["window"] == train_baseline.ROLLING_WINDOW

    def test_saved_spec_round_trips_into_an_online_builder(self, redirected):
        """The artifact must be directly usable by the live twin, not just well-formed."""
        from src.online import FeatureSpec, OnlineFeatureBuilder

        train_baseline.main("FD001")
        spec = FeatureSpec.load(redirected / "feature_spec.json")
        builder = OnlineFeatureBuilder(spec)
        reading = dict.fromkeys(spec.sensors, 500.0)
        assert builder.update(1, reading).shape == (len(spec.feature_cols),)


class TestErrorAnalysis:
    def test_writes_figures_and_report(self, redirected):
        train_baseline.main("FD001")
        error_analysis.main("FD001")
        for name in ("trajectories.png", "residuals.png", "error_analysis.md"):
            assert (redirected / name).exists(), name

    def test_report_mentions_both_metrics(self, redirected):
        train_baseline.main("FD001")
        error_analysis.main("FD001")
        text = (redirected / "error_analysis.md").read_text()
        assert "RMSE" in text and "PHM" in text

    def test_runs_without_a_pretrained_model(self, redirected):
        """It should train its own model rather than crashing on a missing pickle."""
        assert not (redirected / "baseline.pkl").exists()
        error_analysis.main("FD001")
        assert (redirected / "residuals.png").exists()


class TestAblation:
    def test_writes_report_covering_every_arm(self, redirected):
        ablation.main("FD001")
        assert (redirected / "ablation.png").exists()
        rows = json.loads((redirected / "ablation.json").read_text())
        assert len(rows) == len(ablation.ARMS)
        for row in rows:
            assert row["rmse"] >= 0
            assert {"n_features", "phm", "pct_late", "arm"} <= set(row)

    def test_raw_arm_has_fewer_features_than_rolling_arms(self, redirected):
        ablation.main("FD001")
        rows = json.loads((redirected / "ablation.json").read_text())
        raw = next(r for r in rows if not r["arm"].startswith("+"))
        rolling = [r for r in rows if r["arm"].startswith("+")]
        assert all(r["n_features"] == 3 * raw["n_features"] for r in rolling)


class TestUncertainty:
    def test_writes_report_and_figure(self, cmapss_dir, tmp_path, monkeypatch):
        from src import uncertainty

        out = tmp_path / "outputs"
        out.mkdir()
        monkeypatch.setattr(uncertainty, "OUTPUTS_DIR", out)
        uncertainty.main("FD001")

        assert (out / "uncertainty.png").exists()
        payload = json.loads((out / "uncertainty.json").read_text())
        assert {"baseline", "quantiles", "coverage"} <= set(payload)
        assert len(payload["quantiles"]) == len(uncertainty.QUANTILES)
        assert len(payload["coverage"]) == len(uncertainty.COVERAGES)

    def test_high_quantile_shifts_predictions_earlier(self, cmapss_dir, tmp_path, monkeypatch):
        """Guards the sign convention I originally had inverted."""
        from src import uncertainty

        out = tmp_path / "outputs"
        out.mkdir()
        monkeypatch.setattr(uncertainty, "OUTPUTS_DIR", out)
        uncertainty.main("FD001")

        rows = json.loads((out / "uncertainty.json").read_text())["quantiles"]
        by_q = {r["quantile"]: r for r in rows}
        lo, hi = min(by_q), max(by_q)
        # A higher residual quantile subtracts more, so fewer predictions land late.
        assert by_q[hi]["pct_late"] <= by_q[lo]["pct_late"]
        assert by_q[hi]["offset"] > by_q[lo]["offset"]

    def test_split_holds_out_whole_engines(self, cmapss_dir):
        from src.data_loader import load_cmapss
        from src.uncertainty import split_by_engine

        train, _ = load_cmapss("FD001")
        fit, cal, held = split_by_engine(train, 0.34, seed=0)
        assert set(fit["unit"]) & set(cal["unit"]) == set()
        assert len(fit) + len(cal) == len(train)
        assert sorted(cal["unit"].unique().tolist()) == held


class TestStreamDemo:
    def test_replays_an_engine(self, redirected, capsys, monkeypatch):
        from src import stream_demo

        train_baseline.main("FD001")
        # Patched via monkeypatch, not assigned directly: a bare assignment would persist
        # into every later test in the session.
        monkeypatch.setattr(stream_demo, "OUTPUTS_DIR", redirected)
        stream_demo.main(unit=1)

        printed = capsys.readouterr().out
        assert "streaming engine 1" in printed
        assert "RUL ~" in printed


class TestPerEngineUncertainty:
    def test_writes_report_and_figure(self, cmapss_dir, tmp_path, monkeypatch):
        from src import uncertainty_per_engine as upe

        out = tmp_path / "outputs"
        out.mkdir()
        monkeypatch.setattr(upe, "OUTPUTS_DIR", out)
        upe.main("FD001")

        assert (out / "uncertainty_per_engine.png").exists()
        payload = json.loads((out / "uncertainty_per_engine.json").read_text())
        assert {"baseline", "corr_sigma_abserr", "k_selected", "rows"} <= set(payload)
        assert len(payload["rows"]) == len(upe.K_VALUES)

    def test_k_zero_is_the_unadjusted_baseline(self, cmapss_dir, tmp_path, monkeypatch):
        """Invariant: subtracting 0*sigma must change nothing, for both arms."""
        from src import uncertainty_per_engine as upe

        out = tmp_path / "outputs"
        out.mkdir()
        monkeypatch.setattr(upe, "OUTPUTS_DIR", out)
        upe.main("FD001")

        payload = json.loads((out / "uncertainty_per_engine.json").read_text())
        base = payload["baseline"]
        zero = next(r for r in payload["rows"] if r["k"] == 0.0)
        assert zero["mean_shift"] == 0.0
        assert zero["pe_phm"] == pytest.approx(base["phm"], rel=1e-6)
        assert zero["uni_phm"] == pytest.approx(base["phm"], rel=1e-6)
        assert zero["pe_rmse"] == pytest.approx(base["rmse"], rel=1e-6)

    def test_tree_spread_shapes_and_non_negative_std(self, cmapss_dir):
        from src.data_loader import load_cmapss
        from src.features import build_xy
        from src.models import make_model
        from src.uncertainty_per_engine import tree_spread

        train, test = load_cmapss("FD001")
        train_f, test_f, feat_cols, _ = build_xy(train, test)
        forest, _ = make_model()
        forest.fit(train_f[feat_cols].to_numpy(), train_f["RUL"].to_numpy())

        X = test_f[feat_cols].to_numpy()[:20]
        mean, std = tree_spread(forest, X)
        assert mean.shape == std.shape == (20,)
        assert (std >= 0).all()

    def test_tree_spread_rejects_a_model_without_trees(self):
        """The numpy ridge fallback has no per-tree predictions — fail with guidance."""
        from src.models import RidgeFallback
        from src.uncertainty_per_engine import tree_spread

        model = RidgeFallback().fit(np.zeros((5, 2)), np.zeros(5))
        with pytest.raises(SystemExit, match="scikit-learn"):
            tree_spread(model, np.zeros((3, 2)))
