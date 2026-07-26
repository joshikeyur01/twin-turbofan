"""Shared pytest fixtures.

Living at the repo root so pytest puts the root on ``sys.path`` and ``import src``
resolves without an installed package or a PYTHONPATH dance.

The fixtures here write **tiny** C-MAPSS-format files (a handful of engines, a few
dozen cycles) instead of touching ``data/``. Tests stay fast, hermetic, and
independent of whether the real dataset has been downloaded.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import ALL_COLS

# Sensors that actually move in the fixtures; the rest are held constant so tests
# can assert that constant sensors get dropped by feature selection.
FIXTURE_TRENDING = [2, 7, 11, 15]


def _engine_rows(unit, life, truncate=None, rng=None):
    """One engine's rows in C-MAPSS layout: unit, cycle, os1..os3, s1..s21."""
    rng = rng or np.random.default_rng(0)
    n = truncate or life
    rows = []
    for t in range(1, n + 1):
        health = t / life
        sensors = []
        for j in range(1, 22):
            base = 500.0 + j * 10.0
            if j in FIXTURE_TRENDING:
                sensors.append(base + 20.0 * health + rng.normal(0, 0.5))
            else:
                sensors.append(base)  # constant on purpose
        rows.append([unit, t, 0.0, 0.0, 100.0] + sensors)
    return rows


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(" ".join(str(v) for v in r) + "\n")


@pytest.fixture
def cmapss_dir(tmp_path, monkeypatch):
    """Write a miniature FD001 dataset and point the loader at it.

    Returns a dict describing the ground truth the tests assert against:
    ``train_lives`` (unit -> full life) and ``test_truth`` (unit -> (truncate, true RUL)).
    """
    rng = np.random.default_rng(42)
    data_dir = tmp_path / "data"
    subdir = data_dir / "CMAPSSData"
    subdir.mkdir(parents=True)

    train_lives = {1: 40, 2: 60, 3: 200}  # engine 3 exceeds the RUL cap of 125
    train_rows = []
    for unit, life in train_lives.items():
        train_rows += _engine_rows(unit, life, rng=rng)
    _write(subdir / "train_FD001.txt", train_rows)

    # (truncate_at, true_RUL_at_last_cycle)
    test_truth = {1: (30, 15), 2: (50, 90), 3: (20, 140)}  # 140 exceeds the cap
    test_rows = []
    for unit, (trunc, _) in test_truth.items():
        test_rows += _engine_rows(unit, trunc + 50, truncate=trunc, rng=rng)
    _write(subdir / "test_FD001.txt", test_rows)
    _write(subdir / "RUL_FD001.txt", [[r] for _, r in test_truth.values()])

    monkeypatch.setattr("src.data_loader.DATA_DIR", data_dir)
    return {"train_lives": train_lives, "test_truth": test_truth}


@pytest.fixture
def small_frames():
    """A minimal (train, test) pair already in loader output shape, with RUL."""
    rng = np.random.default_rng(7)
    rows = []
    for unit, life in {1: 30, 2: 45}.items():
        rows += _engine_rows(unit, life, rng=rng)
    train = pd.DataFrame(rows, columns=ALL_COLS)
    train["RUL"] = (train.groupby("unit")["cycle"].transform("max") - train["cycle"]).clip(
        upper=125
    )

    rows = []
    for unit, trunc in {1: 20, 2: 25}.items():
        rows += _engine_rows(unit, trunc + 30, truncate=trunc, rng=rng)
    test = pd.DataFrame(rows, columns=ALL_COLS)
    test["RUL"] = 30
    return train, test
