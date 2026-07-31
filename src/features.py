"""Feature engineering for the RUL baseline.

Keeps only sensors that actually vary, then adds short rolling-window mean and
std per engine (a cheap proxy for degradation trend), and standardises using
statistics computed on the training set only.

Dependency-light on purpose: numpy + pandas only, no scikit-learn required.
"""

import pandas as pd

from .data_loader import SENSOR_COLS


def select_informative_sensors(train, threshold: float = 1e-3):
    variances = train[SENSOR_COLS].var()
    return [c for c in SENSOR_COLS if variances[c] > threshold]


def add_rolling_features(df, cols, window: int = 5):
    df = df.sort_values(["unit", "cycle"]).copy()
    g = df.groupby("unit")
    feats = {}
    for c in cols:
        feats[f"{c}_rmean"] = g[c].transform(lambda s: s.rolling(window, min_periods=1).mean())
        feats[f"{c}_rstd"] = g[c].transform(
            lambda s: s.rolling(window, min_periods=1).std().fillna(0.0)
        )
    return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)


def build_xy(train, test, window: int = 5, use_rolling: bool = True):
    """Return (train_f, test_f, feature_cols, (mean, std)).

    Standardisation stats are fit on train and applied to both splits.

    ``use_rolling=False`` gives raw selected sensors only — the ablation arm that
    isolates how much the rolling trend features actually contribute (see
    ``src/ablation.py``).
    """
    sensors = select_informative_sensors(train)
    if use_rolling:
        train_f = add_rolling_features(train, sensors, window)
        test_f = add_rolling_features(test, sensors, window)
        feature_cols = sensors + [f"{c}_rmean" for c in sensors] + [f"{c}_rstd" for c in sensors]
    else:
        # Sort to match the rolling path, so both arms are row-aligned.
        train_f = train.sort_values(["unit", "cycle"]).copy()
        test_f = test.sort_values(["unit", "cycle"]).copy()
        feature_cols = list(sensors)

    mean = train_f[feature_cols].mean()
    std = train_f[feature_cols].std().replace(0, 1.0)
    train_f[feature_cols] = (train_f[feature_cols] - mean) / std
    test_f[feature_cols] = (test_f[feature_cols] - mean) / std
    return train_f, test_f, feature_cols, (mean, std)
