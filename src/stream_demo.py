"""Replay one test engine cycle-by-cycle and print the twin's live RUL estimate.

This is the 'digital twin' loop in miniature: as new telemetry arrives, the twin
updates its prediction of remaining life. Later this becomes an MQTT subscriber
feeding a dashboard.

Run (after train_baseline):
    python -m src.stream_demo --unit 1
"""

import argparse
import pickle
import time

import numpy as np

from .data_loader import load_cmapss
from .features import build_xy
from .models import RidgeFallback  # noqa: F401  (ensures pickled fallback reloads)
from .paths import OUTPUTS_DIR


def main(unit: int = 1, delay: float = 0.0):
    train, test = load_cmapss()
    _, test_f, feat_cols, _ = build_xy(train, test)
    with open(OUTPUTS_DIR / "baseline.pkl", "rb") as f:
        model = pickle.load(f)

    eng = test_f[test_f["unit"] == unit].sort_values("cycle")
    print(f"[twin-turbofan] streaming engine {unit} ({len(eng)} cycles)...")
    for _, row in eng.iterrows():
        x = row[feat_cols].to_numpy().reshape(1, -1)
        pred = float(np.clip(model.predict(x)[0], 0, None))
        bar = "#" * int(min(pred, 125) / 5)
        print(f"  cycle {int(row['cycle']):3d} | RUL ~ {pred:6.1f}  {bar}")
        if delay:
            time.sleep(delay)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--unit", type=int, default=1)
    p.add_argument("--delay", type=float, default=0.0, help="seconds between cycles")
    main(**vars(p.parse_args()))
