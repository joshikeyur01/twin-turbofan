"""Load the NASA C-MAPSS turbofan dataset and attach a Remaining-Useful-Life
(RUL) target.

The loader looks for the real dataset in ``data/CMAPSSData/`` first, and falls
back to synthetic data in ``data/synthetic/`` (see ``generate_synthetic.py``) so
the whole pipeline is runnable before you download anything.

File format (space separated, 26 columns):
    unit, cycle, op_setting_1..3, sensor_1..21
"""

import pandas as pd

from .paths import DATA_DIR

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["os1", "os2", "os3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def _resolve_dir(subset: str):
    for cand in [DATA_DIR / "CMAPSSData", DATA_DIR / "synthetic"]:
        if (cand / f"train_{subset}.txt").exists():
            return cand
    raise FileNotFoundError(
        "No C-MAPSS data found.\n"
        "  - Real data: place train_/test_/RUL_ *.txt in data/CMAPSSData/\n"
        "  - Or generate synthetic data: python -m src.generate_synthetic"
    )


def available_subsets():
    """Which of FD001–FD004 are actually present, in order.

    Lets the comparison and sweep scripts adapt instead of hard-coding four
    datasets and crashing on a checkout that only has the synthetic FD001.
    """
    found = []
    for subset in ("FD001", "FD002", "FD003", "FD004"):
        for cand in (DATA_DIR / "CMAPSSData", DATA_DIR / "synthetic"):
            if (cand / f"train_{subset}.txt").exists():
                found.append(subset)
                break
    return found


def using_real_data() -> bool:
    """True when the real NASA files are in place.

    Metrics computed on the synthetic fallback are plumbing checks, not benchmarks,
    so reports carry this flag rather than implying the numbers are comparable to
    published C-MAPSS results.
    """
    return (DATA_DIR / "CMAPSSData" / "train_FD001.txt").exists()


def _read_txt(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")  # strip trailing-space phantom columns
    df.columns = ALL_COLS[: df.shape[1]]
    return df


def load_cmapss(subset: str = "FD001", rul_cap: int = 125):
    """Return (train, test) DataFrames with a clipped ``RUL`` column.

    ``rul_cap`` implements the standard *piecewise-linear* RUL target: early in
    life, remaining life is not meaningfully predictable, so it is capped.
    """
    d = _resolve_dir(subset)
    train = _read_txt(d / f"train_{subset}.txt")
    test = _read_txt(d / f"test_{subset}.txt")
    rul_true = pd.read_csv(d / f"RUL_{subset}.txt", sep=r"\s+", header=None).iloc[:, 0].to_numpy()

    # Training engines run to failure -> RUL = cycles remaining.
    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train["RUL"] = (max_cycle - train["cycle"]).clip(upper=rul_cap)

    # Test engines are truncated; RUL_*.txt gives true RUL at each engine's
    # LAST recorded cycle. Back-fill earlier cycles from that anchor.
    test_max = test.groupby("unit")["cycle"].transform("max")
    rul_map = {u: rul_true[i] for i, u in enumerate(sorted(test["unit"].unique()))}
    test["RUL"] = (test["unit"].map(rul_map) + (test_max - test["cycle"])).clip(upper=rul_cap)
    return train, test


def last_cycle_rows(df):
    """One row per engine: its final recorded cycle (the scoring point)."""
    idx = df.groupby("unit")["cycle"].idxmax()
    return df.loc[idx].reset_index(drop=True)
