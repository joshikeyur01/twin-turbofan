"""Tests for feature engineering.

Two properties matter most here:

- **No NaNs.** Rolling windows produce NaNs at the start of every engine's series;
  if those leak through, tree models silently degrade or refuse to fit.
- **No leakage.** Standardisation statistics must come from the training split only,
  and rolling windows must never span an engine boundary — a window that mixes two
  engines invents degradation history that never happened.
"""

import numpy as np
import pytest

from src.data_loader import SENSOR_COLS
from src.features import add_rolling_features, build_xy, select_informative_sensors


class TestSensorSelection:
    def test_drops_constant_sensors(self, small_frames):
        """Fixtures hold most sensors constant; only the trending ones should survive."""
        train, _ = small_frames
        chosen = select_informative_sensors(train)
        assert chosen, "should select at least one sensor"
        assert set(chosen) < set(SENSOR_COLS), "constant sensors must be dropped"
        for c in chosen:
            assert train[c].var() > 1e-3

    def test_threshold_is_respected(self, small_frames):
        train, _ = small_frames
        assert select_informative_sensors(train, threshold=1e9) == []

    def test_preserves_canonical_sensor_order(self, small_frames):
        train, _ = small_frames
        chosen = select_informative_sensors(train)
        assert chosen == [c for c in SENSOR_COLS if c in chosen]


class TestRollingFeatures:
    def test_no_nans_introduced(self, small_frames):
        train, _ = small_frames
        cols = select_informative_sensors(train)
        out = add_rolling_features(train, cols)
        assert not out.isna().any().any()

    def test_first_row_std_is_zero_not_nan(self, small_frames):
        """A 1-sample window has undefined std; it must be filled with 0."""
        train, _ = small_frames
        cols = select_informative_sensors(train)
        out = add_rolling_features(train, cols).sort_values(["unit", "cycle"])
        firsts = out.groupby("unit").head(1)
        for c in cols:
            assert (firsts[f"{c}_rstd"] == 0.0).all()

    def test_first_row_mean_equals_raw_value(self, small_frames):
        train, _ = small_frames
        cols = select_informative_sensors(train)
        out = add_rolling_features(train, cols).sort_values(["unit", "cycle"])
        firsts = out.groupby("unit").head(1)
        for c in cols:
            np.testing.assert_allclose(firsts[f"{c}_rmean"], firsts[c])

    def test_windows_do_not_span_engines(self, small_frames):
        """Engine 2's first rolling mean must ignore engine 1's history entirely."""
        train, _ = small_frames
        cols = select_informative_sensors(train)
        out = add_rolling_features(train, cols).sort_values(["unit", "cycle"])
        first_of_2 = out[out["unit"] == 2].iloc[0]
        assert first_of_2[f"{cols[0]}_rmean"] == pytest.approx(first_of_2[cols[0]])

    def test_rolling_mean_matches_manual_window(self, small_frames):
        train, _ = small_frames
        cols = select_informative_sensors(train)
        window = 5
        out = add_rolling_features(train, cols, window=window).sort_values(["unit", "cycle"])
        g_out = out[out["unit"] == 1]
        raw = train[train["unit"] == 1].sort_values("cycle")[cols[0]].to_numpy()
        expected = raw[5:10].mean()  # 10th row, full window
        assert g_out[f"{cols[0]}_rmean"].iloc[9] == pytest.approx(expected)

    def test_does_not_mutate_input(self, small_frames):
        train, _ = small_frames
        before = train.copy()
        add_rolling_features(train, select_informative_sensors(train))
        assert train.equals(before)


class TestBuildXY:
    def test_expected_columns_present(self, small_frames):
        train, test = small_frames
        train_f, test_f, feat_cols, _ = build_xy(train, test)
        sensors = select_informative_sensors(train)
        assert feat_cols == (
            sensors + [f"{c}_rmean" for c in sensors] + [f"{c}_rstd" for c in sensors]
        )
        for df in (train_f, test_f):
            assert set(feat_cols) <= set(df.columns)

    def test_no_nans_in_features(self, small_frames):
        train, test = small_frames
        train_f, test_f, feat_cols, _ = build_xy(train, test)
        assert not train_f[feat_cols].isna().any().any()
        assert not test_f[feat_cols].isna().any().any()

    def test_all_features_finite(self, small_frames):
        train, test = small_frames
        train_f, test_f, feat_cols, _ = build_xy(train, test)
        assert np.isfinite(train_f[feat_cols].to_numpy()).all()
        assert np.isfinite(test_f[feat_cols].to_numpy()).all()

    def test_train_is_standardised(self, small_frames):
        train, test = small_frames
        train_f, _, feat_cols, _ = build_xy(train, test)
        means = train_f[feat_cols].mean().to_numpy()
        np.testing.assert_allclose(means, 0.0, atol=1e-9)

    def test_test_uses_train_statistics_not_its_own(self, small_frames):
        """Leakage guard: test features must NOT be centred on their own mean."""
        train, test = small_frames
        _, test_f, feat_cols, (mean, std) = build_xy(train, test)
        # Re-deriving from the returned train stats must reproduce the test features.
        raw = add_rolling_features(test, select_informative_sensors(train))
        expected = (raw[feat_cols] - mean) / std
        np.testing.assert_allclose(
            test_f.sort_index()[feat_cols].to_numpy(),
            expected.sort_index().to_numpy(),
            rtol=1e-9,
        )

    def test_zero_variance_std_does_not_divide_by_zero(self, small_frames):
        """std==0 is replaced by 1.0, so constant features stay finite."""
        train, test = small_frames
        _, _, _, (_, std) = build_xy(train, test)
        assert (std != 0).all()

    def test_rul_column_survives(self, small_frames):
        train, test = small_frames
        train_f, test_f, _, _ = build_xy(train, test)
        assert "RUL" in train_f.columns
        assert "RUL" in test_f.columns

    def test_row_count_preserved(self, small_frames):
        train, test = small_frames
        train_f, test_f, _, _ = build_xy(train, test)
        assert len(train_f) == len(train)
        assert len(test_f) == len(test)


class TestAblationArm:
    """``use_rolling=False`` is the ablation baseline — raw sensors only."""

    def test_raw_arm_has_only_selected_sensors(self, small_frames):
        train, test = small_frames
        _, _, feat_cols, _ = build_xy(train, test, use_rolling=False)
        assert feat_cols == select_informative_sensors(train)
        assert not any(c.endswith(("_rmean", "_rstd")) for c in feat_cols)

    def test_raw_arm_has_one_third_the_features(self, small_frames):
        """Rolling adds a mean and a std per sensor, so 3x the columns."""
        train, test = small_frames
        _, _, raw_cols, _ = build_xy(train, test, use_rolling=False)
        _, _, roll_cols, _ = build_xy(train, test, use_rolling=True)
        assert len(roll_cols) == 3 * len(raw_cols)

    def test_raw_arm_still_standardised_on_train(self, small_frames):
        train, test = small_frames
        train_f, _, feat_cols, _ = build_xy(train, test, use_rolling=False)
        np.testing.assert_allclose(train_f[feat_cols].mean().to_numpy(), 0.0, atol=1e-9)

    def test_raw_arm_no_nans(self, small_frames):
        train, test = small_frames
        train_f, test_f, feat_cols, _ = build_xy(train, test, use_rolling=False)
        assert not train_f[feat_cols].isna().any().any()
        assert not test_f[feat_cols].isna().any().any()

    def test_both_arms_row_aligned(self, small_frames):
        """Arms must be comparable: same rows in the same order, different columns."""
        train, test = small_frames
        raw_tr, raw_te, _, _ = build_xy(train, test, use_rolling=False)
        rol_tr, rol_te, _, _ = build_xy(train, test, use_rolling=True)
        for a, b in [(raw_tr, rol_tr), (raw_te, rol_te)]:
            np.testing.assert_array_equal(a["unit"].to_numpy(), b["unit"].to_numpy())
            np.testing.assert_array_equal(a["cycle"].to_numpy(), b["cycle"].to_numpy())
            np.testing.assert_array_equal(a["RUL"].to_numpy(), b["RUL"].to_numpy())

    def test_raw_arm_does_not_mutate_input(self, small_frames):
        train, test = small_frames
        before = train.copy()
        build_xy(train, test, use_rolling=False)
        assert train.equals(before)

    def test_window_changes_rolling_features(self, small_frames):
        """A different window must actually produce different feature values."""
        train, test = small_frames
        tr3, _, cols, _ = build_xy(train, test, window=3)
        tr20, _, _, _ = build_xy(train, test, window=20)
        rstd = [c for c in cols if c.endswith("_rstd")][0]
        assert not np.allclose(tr3[rstd].to_numpy(), tr20[rstd].to_numpy())
