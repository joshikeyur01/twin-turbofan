"""Generate synthetic data in C-MAPSS FD001 format.

This exists ONLY so the pipeline is runnable before you download the real
dataset. The degradation here is a simple monotonic drift plus noise -- it is
NOT a substitute for the real NASA data and results on it are meaningless as a
benchmark. Use it to validate plumbing, then swap in data/CMAPSSData/.

**Sensor structure (v2).** Real FD001 splits its 21 sensors in two: six record the
same value on every cycle of every engine (s1, s5, s10, s16, s18, s19), and the
rest move with wear. v1 of this generator gave its "constant" sensors sigma=0.5
noise, which put their variance far above ``features.variance_threshold`` -- so
``select_informative_sensors`` dropped 0 of 21 here while dropping 6 of 21 on the
real data, and the drop path was exercised by unit tests only, never by a run.
The six are now emitted as literal constants, so the fallback takes the same code
path as the real data. See ``outputs/synthetic_fidelity.md`` for what that did to
the baseline numbers.

Run:
    python -m src.generate_synthetic --config config.yaml
"""

import numpy as np

from .config import load_config
from .paths import DATA_DIR

RNG = np.random.default_rng(42)
N_TRAIN = 100
N_TEST = 50
N_SENSORS = 21
# 1-indexed, matching s1..s21. The constant set is FD001's: those six sensors have
# zero variance in the real training file. Everything else drifts with degradation.
CONSTANT = (1, 5, 10, 16, 18, 19)
TRENDING = tuple(j for j in range(1, N_SENSORS + 1) if j not in CONSTANT)  # the other 15

# Column offset of s1 in the C-MAPSS row layout: unit, cycle, os1..os3, s1..s21.
_SENSOR_OFFSET = 5


def _one_engine(unit, life, truncate=None):
    rows = []
    for t in range(1, life + 1):
        health = (t / life) ** 1.5  # 0 (healthy) -> 1 (failed)
        os = [float(RNG.normal(0, 0.002)), float(RNG.normal(0, 0.0003)), 100.0]
        sensors = []
        for j in range(1, N_SENSORS + 1):
            base = 500.0 + j * 10.0
            if j in CONSTANT:
                # No noise at all: variance is exactly 0, as it is in real FD001.
                val = base
            else:
                sign = 1.0 if j % 2 == 0 else -1.0
                val = base + sign * 25.0 * health + RNG.normal(0, 1.5)
            sensors.append(float(val))
        rows.append([unit, t] + os + sensors)
        if truncate is not None and t == truncate:
            break
    return rows


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(" ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + "\n")


def _flat_sensors(rows, threshold):
    """Which sensors the feature selector will drop, computed from the rows just written.

    Recomputed from the data rather than asserted from :data:`CONSTANT` on purpose: the
    point of the constant set is what it does to ``select_informative_sensors``, and only
    the written values decide that.
    """
    arr = np.asarray(rows, dtype=float)[:, _SENSOR_OFFSET:]
    variances = arr.var(axis=0, ddof=1)
    return [i + 1 for i, v in enumerate(variances) if v <= threshold]


def main(config: str | None = None):
    cfg = load_config(config)
    out = DATA_DIR / "synthetic"
    out.mkdir(parents=True, exist_ok=True)

    # Training engines: run to failure.
    train_rows = []
    for u in range(1, N_TRAIN + 1):
        train_rows += _one_engine(u, int(RNG.integers(130, 250)))
    _write(out / "train_FD001.txt", train_rows)

    # Test engines: truncated before failure; record the true RUL.
    test_rows, ruls = [], []
    for u in range(1, N_TEST + 1):
        life = int(RNG.integers(130, 250))
        trunc = int(RNG.integers(30, life - 10))
        test_rows += _one_engine(u, life, truncate=trunc)
        ruls.append(life - trunc)
    _write(out / "test_FD001.txt", test_rows)
    _write(out / "RUL_FD001.txt", [[r] for r in ruls])

    # Fail loudly rather than write data that silently stops exercising the drop path.
    flat = _flat_sensors(train_rows, cfg.variance_threshold)
    if flat != list(CONSTANT):
        raise SystemExit(
            f"generator/selector disagreement: at variance_threshold="
            f"{cfg.variance_threshold} the written data is flat in sensors {flat}, "
            f"but CONSTANT declares {list(CONSTANT)}"
        )

    print(f"Wrote synthetic FD001 -> {out}  (train={N_TRAIN} engines, test={N_TEST} engines)")
    print(
        f"Flat sensors {[f's{j}' for j in flat]} — select_informative_sensors will keep "
        f"{N_SENSORS - len(flat)} of {N_SENSORS} at threshold {cfg.variance_threshold}"
    )


if __name__ == "__main__":
    import argparse

    from .config import add_config_args, setup_logging

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_config_args(p)
    a = p.parse_args()

    _cfg = load_config(a.config, logging_level=a.log_level)
    setup_logging(_cfg.logging_level, _cfg.logging_timestamps)
    main(config=a.config)
