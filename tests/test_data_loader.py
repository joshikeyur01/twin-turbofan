"""Tests for C-MAPSS loading and RUL labelling.

RUL labelling is the single easiest thing to get subtly wrong in this project, and a
silent error here invalidates every metric downstream. Train and test engines are
labelled by *different* rules:

- **train** engines run to failure, so RUL = (last cycle − current cycle).
- **test** engines are truncated early, so the true RUL at the final recorded cycle
  comes from ``RUL_*.txt`` and earlier cycles are back-filled from that anchor.

Both are then clipped by the piecewise-linear cap (125 by default).
"""

import numpy as np
import pytest

from src.data_loader import ALL_COLS, SENSOR_COLS, last_cycle_rows, load_cmapss


class TestSchema:
    def test_columns_parsed(self, cmapss_dir):
        train, test = load_cmapss("FD001")
        for df in (train, test):
            assert list(df.columns) == ALL_COLS + ["RUL"]

    def test_no_nans(self, cmapss_dir):
        train, test = load_cmapss("FD001")
        assert not train.isna().any().any()
        assert not test.isna().any().any()

    def test_missing_data_raises_actionable_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data_loader.DATA_DIR", tmp_path / "nope")
        with pytest.raises(FileNotFoundError, match="No C-MAPSS data found"):
            load_cmapss("FD001")


class TestTrainLabelling:
    def test_final_cycle_has_zero_rul(self, cmapss_dir):
        """A run-to-failure engine has zero remaining life at its last cycle."""
        train, _ = load_cmapss("FD001")
        finals = train.loc[train.groupby("unit")["cycle"].idxmax()]
        assert (finals["RUL"] == 0).all()

    def test_rul_counts_down_by_one_per_cycle(self, cmapss_dir):
        """Below the cap, RUL must decrease exactly 1 per cycle."""
        train, _ = load_cmapss("FD001")
        g = train[train["unit"] == 1].sort_values("cycle")
        diffs = np.diff(g["RUL"].to_numpy())
        assert set(np.unique(diffs)) <= {-1, 0}
        # engine 1 has life 40 < cap, so it is a pure countdown with no plateau
        assert (diffs == -1).all()

    def test_rul_equals_cycles_remaining(self, cmapss_dir):
        train, _ = load_cmapss("FD001")
        life = cmapss_dir["train_lives"][2]
        g = train[train["unit"] == 2].sort_values("cycle")
        expected = np.minimum(life - g["cycle"].to_numpy(), 125)
        np.testing.assert_array_equal(g["RUL"].to_numpy(), expected)

    def test_cap_applied_to_long_lived_engine(self, cmapss_dir):
        """Engine 3 lives 200 cycles, so its early life must plateau at the cap."""
        train, _ = load_cmapss("FD001")
        g = train[train["unit"] == 3].sort_values("cycle")
        assert g["RUL"].max() == 125
        assert (g["RUL"].to_numpy()[:70] == 125).all(), "early life should be a plateau"

    def test_custom_cap_respected(self, cmapss_dir):
        train, _ = load_cmapss("FD001", rul_cap=30)
        assert train["RUL"].max() == 30

    def test_rul_never_negative(self, cmapss_dir):
        train, test = load_cmapss("FD001")
        assert (train["RUL"] >= 0).all()
        assert (test["RUL"] >= 0).all()


class TestTestLabelling:
    def test_last_cycle_matches_rul_file(self, cmapss_dir):
        """The anchor: RUL at each test engine's final cycle comes from RUL_*.txt."""
        _, test = load_cmapss("FD001")
        finals = last_cycle_rows(test).set_index("unit")["RUL"]
        for unit, (_, true_rul) in cmapss_dir["test_truth"].items():
            assert finals[unit] == min(true_rul, 125)

    def test_backfill_counts_down_toward_the_anchor(self, cmapss_dir):
        """Earlier cycles must have MORE remaining life than later ones."""
        _, test = load_cmapss("FD001")
        g = test[test["unit"] == 1].sort_values("cycle")
        diffs = np.diff(g["RUL"].to_numpy())
        assert (diffs == -1).all()
        # engine 1: truncated at 30 with true RUL 15 -> first cycle has 15 + 29
        assert g["RUL"].iloc[0] == 15 + 29

    def test_cap_applied_to_test_anchor(self, cmapss_dir):
        """Engine 3's true RUL is 140, above the cap, so it must clip to 125."""
        _, test = load_cmapss("FD001")
        finals = last_cycle_rows(test).set_index("unit")["RUL"]
        assert finals[3] == 125

    def test_engines_are_truncated_before_failure(self, cmapss_dir):
        """Sanity: no test engine reaches RUL 0, by construction of the dataset."""
        _, test = load_cmapss("FD001")
        assert (last_cycle_rows(test)["RUL"] > 0).all()


class TestLastCycleRows:
    def test_one_row_per_engine(self, cmapss_dir):
        _, test = load_cmapss("FD001")
        last = last_cycle_rows(test)
        assert len(last) == test["unit"].nunique()
        assert last["unit"].is_unique

    def test_selects_the_maximum_cycle(self, cmapss_dir):
        _, test = load_cmapss("FD001")
        last = last_cycle_rows(test).set_index("unit")["cycle"]
        expected = test.groupby("unit")["cycle"].max()
        assert last.to_dict() == expected.to_dict()

    def test_preserves_all_columns(self, cmapss_dir):
        _, test = load_cmapss("FD001")
        assert list(last_cycle_rows(test).columns) == list(test.columns)

    def test_returns_actual_sensor_values_not_aggregates(self, cmapss_dir):
        """The scoring row must be a real observation, not a groupby summary."""
        _, test = load_cmapss("FD001")
        last = last_cycle_rows(test)
        row = last[last["unit"] == 2].iloc[0]
        source = test[(test["unit"] == 2) & (test["cycle"] == row["cycle"])].iloc[0]
        np.testing.assert_array_equal(row[SENSOR_COLS].to_numpy(), source[SENSOR_COLS].to_numpy())
