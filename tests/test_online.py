"""Tests for the streaming feature path.

The headline test is ``test_matches_batch_features_exactly``: the live twin must
produce the same feature vector as the offline pipeline for the same engine. If the
two paths disagree, the model is being served inputs it was never trained on and
every live prediction is quietly wrong — the classic train/serve skew failure, which
produces no error and no crash, just bad numbers.

The interleaving test matters for the MQTT bus specifically: readings from different
engines arrive mixed together, so per-engine state must stay isolated.
"""

import numpy as np
import pytest

from src.features import build_xy, select_informative_sensors
from src.online import FeatureSpec, OnlineFeatureBuilder

WINDOW = 5


@pytest.fixture
def spec_and_frames(small_frames):
    train, test = small_frames
    train_f, test_f, feat_cols, (mean, std) = build_xy(train, test, window=WINDOW)
    spec = FeatureSpec.from_build_xy(
        sensors=select_informative_sensors(train),
        feature_cols=feat_cols,
        window=WINDOW,
        mean=mean,
        std=std,
    )
    return spec, test, test_f, feat_cols


class TestFeatureSpec:
    def test_roundtrips_through_json(self, spec_and_frames, tmp_path):
        spec, *_ = spec_and_frames
        path = spec.save(tmp_path / "spec.json")
        loaded = FeatureSpec.load(path)
        assert loaded.sensors == spec.sensors
        assert loaded.feature_cols == spec.feature_cols
        assert loaded.window == spec.window
        np.testing.assert_allclose(loaded.mean_vec, spec.mean_vec)
        np.testing.assert_allclose(loaded.std_vec, spec.std_vec)

    def test_vectors_follow_feature_col_order(self, spec_and_frames):
        """Order matters: a permuted vector would silently mis-scale every feature."""
        spec, *_ = spec_and_frames
        expected = np.array([spec.mean[c] for c in spec.feature_cols])
        np.testing.assert_allclose(spec.mean_vec, expected)

    def test_missing_spec_raises_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="train_baseline"):
            FeatureSpec.load(tmp_path / "absent.json")

    def test_std_never_zero(self, spec_and_frames):
        """build_xy replaces zero std with 1.0; the spec must inherit that."""
        spec, *_ = spec_and_frames
        assert (spec.std_vec != 0).all()


class TestOnlineFeatureBuilder:
    def test_matches_batch_features_exactly(self, spec_and_frames):
        """Streaming one cycle at a time must equal the batch computation."""
        spec, test, test_f, feat_cols = spec_and_frames
        builder = OnlineFeatureBuilder(spec)

        unit = 1
        raw_eng = test[test["unit"] == unit].sort_values("cycle")
        batch_eng = test_f[test_f["unit"] == unit].sort_values("cycle")

        for (_, raw_row), (_, batch_row) in zip(
            raw_eng.iterrows(), batch_eng.iterrows(), strict=True
        ):
            got = builder.update(unit, raw_row.to_dict())
            expected = batch_row[feat_cols].to_numpy(dtype=float)
            np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)

    def test_first_reading_has_zero_rolling_std(self, spec_and_frames):
        """One observation has undefined sample std; batch fills 0.0, so must we."""
        spec, test, _, feat_cols = spec_and_frames
        builder = OnlineFeatureBuilder(spec)
        row = test[test["unit"] == 1].sort_values("cycle").iloc[0]
        vec = builder.update(1, row.to_dict())

        std_idx = [i for i, c in enumerate(feat_cols) if c.endswith("_rstd")]
        # Standardised, so the raw 0.0 maps to (0 - mean) / std.
        expected = -spec.mean_vec[std_idx] / spec.std_vec[std_idx]
        np.testing.assert_allclose(vec[std_idx], expected, rtol=1e-9)

    def test_interleaved_engines_stay_isolated(self, spec_and_frames):
        """MQTT delivers engines mixed together; buffers must not cross-contaminate."""
        spec, test, test_f, feat_cols = spec_and_frames
        sequential = OnlineFeatureBuilder(spec)
        interleaved = OnlineFeatureBuilder(spec)

        e1 = test[test["unit"] == 1].sort_values("cycle").to_dict("records")
        e2 = test[test["unit"] == 2].sort_values("cycle").to_dict("records")
        n = min(len(e1), len(e2))

        seq1 = [sequential.update(1, r) for r in e1[:n]]
        sequential.reset()
        seq2 = [sequential.update(2, r) for r in e2[:n]]

        mix1, mix2 = [], []
        for r1, r2 in zip(e1[:n], e2[:n], strict=True):
            mix1.append(interleaved.update(1, r1))
            mix2.append(interleaved.update(2, r2))

        np.testing.assert_allclose(np.array(seq1), np.array(mix1), rtol=1e-9)
        np.testing.assert_allclose(np.array(seq2), np.array(mix2), rtol=1e-9)

    def test_window_is_bounded(self, spec_and_frames):
        """Memory must not grow with stream length — the buffer is a ring."""
        spec, test, _, _ = spec_and_frames
        builder = OnlineFeatureBuilder(spec)
        rows = test[test["unit"] == 1].sort_values("cycle").to_dict("records")
        for r in rows:
            builder.update(1, r)
        assert len(builder._buf[1]) == min(spec.window, len(rows))

    def test_reset_clears_history(self, spec_and_frames):
        spec, test, _, _ = spec_and_frames
        builder = OnlineFeatureBuilder(spec)
        rows = test[test["unit"] == 1].sort_values("cycle").to_dict("records")
        for r in rows[:4]:
            builder.update(1, r)
        builder.reset(1)
        after = builder.update(1, rows[0])
        fresh = OnlineFeatureBuilder(spec).update(1, rows[0])
        np.testing.assert_allclose(after, fresh, rtol=1e-9)

    def test_extra_keys_in_reading_are_ignored(self, spec_and_frames):
        """Payloads carry cycle/settings/timestamps; only sensors should be read."""
        spec, test, _, _ = spec_and_frames
        row = test[test["unit"] == 1].sort_values("cycle").iloc[0].to_dict()
        clean = OnlineFeatureBuilder(spec).update(1, row)
        noisy = OnlineFeatureBuilder(spec).update(
            1, {**row, "timestamp": "2026-07-30T00:00:00Z", "site": "hangar-3"}
        )
        np.testing.assert_allclose(clean, noisy, rtol=1e-9)

    def test_missing_sensor_raises(self, spec_and_frames):
        """A truncated payload must fail loudly, not silently impute."""
        spec, test, _, _ = spec_and_frames
        row = test[test["unit"] == 1].sort_values("cycle").iloc[0].to_dict()
        del row[spec.sensors[0]]
        with pytest.raises(KeyError):
            OnlineFeatureBuilder(spec).update(1, row)

    def test_predict_clips_at_zero(self, spec_and_frames):
        """Negative remaining life is meaningless; match the batch path's clip."""
        spec, test, _, _ = spec_and_frames

        class AlwaysNegative:
            def predict(self, X):
                return np.array([-42.0])

        row = test[test["unit"] == 1].sort_values("cycle").iloc[0].to_dict()
        got = OnlineFeatureBuilder(spec).predict(AlwaysNegative(), 1, row)
        assert got == 0.0

    def test_raw_only_spec_skips_rolling(self, small_frames):
        train, test = small_frames
        _, _, feat_cols, (mean, std) = build_xy(train, test, use_rolling=False)
        spec = FeatureSpec.from_build_xy(
            sensors=select_informative_sensors(train),
            feature_cols=feat_cols,
            window=WINDOW,
            mean=mean,
            std=std,
            use_rolling=False,
        )
        row = test.sort_values(["unit", "cycle"]).iloc[0].to_dict()
        vec = OnlineFeatureBuilder(spec).update(1, row)
        assert len(vec) == len(spec.sensors)
