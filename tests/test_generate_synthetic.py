"""Tests for the synthetic fallback generator.

The property under test is not "the file parses" — ``test_data_loader`` covers that.
It is **how many sensors survive feature selection**, because that is where v1 of this
generator quietly diverged from the real dataset.

Real FD001 has six sensors that never move, so ``select_informative_sensors`` drops
6 of 21 on it. v1 gave those six sigma=0.5 noise, so it dropped **0 of 21** — the
fallback ran a code path the real data never takes, and every number produced on it
came from 21 sensors and 63 features instead of 15 and 45. These tests pin the drop
count so that cannot regress silently.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import ALL_COLS, SENSOR_COLS, load_cmapss
from src.features import build_xy, select_informative_sensors
from src.generate_synthetic import CONSTANT, N_SENSORS, TRENDING, _one_engine

# What the real FD001 training file is flat in, and therefore what the fallback must
# also be flat in. Spelled out rather than imported from CONSTANT: a test that reads
# the value it is checking would pass no matter what that value became.
FD001_FLAT_SENSORS = ["s1", "s5", "s10", "s16", "s18", "s19"]
N_INFORMATIVE = 15


def _frame(units=(1, 2, 3), life=60):
    """A train-shaped frame straight from the generator's per-engine rows."""
    rows = []
    for unit in units:
        rows += _one_engine(unit, life)
    df = pd.DataFrame(rows, columns=ALL_COLS)
    df["RUL"] = (df.groupby("unit")["cycle"].transform("max") - df["cycle"]).clip(upper=125)
    return df


class TestSensorPartition:
    def test_constant_and_trending_cover_all_21(self):
        assert sorted(CONSTANT + TRENDING) == list(range(1, N_SENSORS + 1))
        assert not set(CONSTANT) & set(TRENDING)

    def test_six_constant_fifteen_trending(self):
        """The 21-sensor shape is preserved; only the split changes."""
        assert len(CONSTANT) == 6
        assert len(TRENDING) == N_INFORMATIVE

    def test_constant_set_matches_real_fd001(self):
        assert [f"s{j}" for j in CONSTANT] == FD001_FLAT_SENSORS


class TestDropCount:
    def test_drops_exactly_six_of_twenty_one(self):
        train = _frame()
        informative = select_informative_sensors(train)
        assert len(informative) == N_INFORMATIVE
        assert len(SENSOR_COLS) - len(informative) == 6

    def test_drops_the_sensors_real_fd001_drops(self):
        train = _frame()
        dropped = [c for c in SENSOR_COLS if c not in select_informative_sensors(train)]
        assert dropped == FD001_FLAT_SENSORS

    def test_constant_sensors_have_exactly_zero_variance(self):
        """Not "small" — zero. A near-constant sensor's drop would depend on the threshold."""
        train = _frame()
        for col in FD001_FLAT_SENSORS:
            assert train[col].var() == 0.0
            assert train[col].nunique() == 1

    def test_trending_sensors_clear_the_threshold_by_orders_of_magnitude(self):
        """The surviving 15 are not borderline: the drop is structural, not a tuning artefact."""
        train = _frame()
        for col in select_informative_sensors(train):
            assert train[col].var() > 1.0

    def test_v1_noise_on_flat_sensors_would_have_dropped_nothing(self):
        """Pin the exact defect this replaced.

        v1 emitted ``base + N(0, 0.5)`` for the flat sensors. That variance (~0.25) is
        250x the 1e-3 threshold, so all 21 sensors were kept. Re-injecting that noise
        must reproduce the old behaviour — proving the drop comes from removing the
        noise, not from anything else that changed in the generator.
        """
        train = _frame()
        rng = np.random.default_rng(0)
        v1 = train.copy()
        for col in FD001_FLAT_SENSORS:
            v1[col] += rng.normal(0, 0.5, len(v1))
        assert len(select_informative_sensors(v1)) == 21


class TestFeatureCount:
    def test_rolling_features_scale_with_the_smaller_sensor_set(self):
        """15 sensors x (raw + rmean + rstd) = 45 features, down from 63."""
        train = _frame()
        test = _frame(units=(4, 5), life=40)
        _, _, feat_cols, _ = build_xy(train, test)
        assert len(feat_cols) == 3 * N_INFORMATIVE == 45


@pytest.mark.parametrize("threshold", [1e-6, 1e-3, 1e-1])
def test_drop_count_is_insensitive_to_the_threshold(threshold):
    """Zero-variance vs variance>1 leaves a gap no plausible threshold falls in."""
    train = _frame()
    assert len(select_informative_sensors(train, threshold=threshold)) == N_INFORMATIVE


def test_end_to_end_written_files_drop_six(tmp_path, monkeypatch):
    """The generator's own self-check and the loader agree on the written files."""
    from src import generate_synthetic as gen

    monkeypatch.setattr(gen, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gen, "N_TRAIN", 6)
    monkeypatch.setattr(gen, "N_TEST", 3)
    monkeypatch.setattr("src.data_loader.DATA_DIR", tmp_path)

    gen.main()  # raises SystemExit if the written data is not flat in exactly CONSTANT

    train, test = load_cmapss("FD001")
    assert train.shape[1] == len(ALL_COLS) + 1  # 21 sensors still on disk, plus RUL
    assert len(select_informative_sensors(train)) == N_INFORMATIVE
