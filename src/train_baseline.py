"""Train the Week-1 baseline RUL regressor.

Default model is scikit-learn's RandomForest. If scikit-learn isn't installed,
it transparently falls back to a numpy-only ridge regression so the pipeline
still runs anywhere (the fallback is weaker — install scikit-learn for the real
baseline).

Run:
    python -m src.generate_synthetic   # or drop the real data in data/CMAPSSData/
    python -m src.train_baseline

Writes outputs/baseline.pkl, outputs/metrics.json, outputs/pred_vs_true.png
"""

import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import load_config  # noqa: E402
from .data_loader import last_cycle_rows, load_cmapss  # noqa: E402
from .evaluate import phm_score, rmse  # noqa: E402
from .features import build_xy, select_informative_sensors  # noqa: E402
from .models import make_model  # noqa: E402
from .online import FeatureSpec  # noqa: E402
from .paths import OUTPUTS_DIR  # noqa: E402

# Kept as a module constant because tests and the feature spec both reference it; the
# value now comes from config.yaml, with this as the fallback when PyYAML is absent.
ROLLING_WINDOW = load_config().rolling_window


def main(subset: str | None = None, config: str | None = None):
    cfg = load_config(config, subset=subset)
    train, test = load_cmapss(cfg.subset, rul_cap=cfg.rul_cap)
    train_f, test_f, feat_cols, (mean, std) = build_xy(train, test, window=cfg.rolling_window)
    subset = cfg.subset

    model, model_name = make_model()
    print(f"model: {model_name}")
    model.fit(train_f[feat_cols].to_numpy(), train_f["RUL"].to_numpy())

    # Score on each engine's final cycle (the operational decision point).
    test_last = last_cycle_rows(test_f)
    y_true = test_last["RUL"].to_numpy()
    y_pred = np.clip(model.predict(test_last[feat_cols].to_numpy()), 0, None)

    metrics = {
        "model": model_name,
        "subset": subset,
        "rmse": round(rmse(y_true, y_pred), 3),
        "phm_score": round(phm_score(y_true, y_pred), 1),
        "n_test_engines": int(len(y_true)),
    }
    print(json.dumps(metrics, indent=2))

    with open(OUTPUTS_DIR / "baseline.pkl", "wb") as f:
        pickle.dump(model, f)
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Persist the feature contract so the live twin standardises with TRAINING
    # statistics instead of recomputing them from streaming data (train/serve skew).
    spec = FeatureSpec.from_build_xy(
        sensors=select_informative_sensors(train),
        feature_cols=feat_cols,
        window=cfg.rolling_window,
        mean=mean,
        std=std,
    )
    # Explicit path off this module's OUTPUTS_DIR so the artifact always lands beside the
    # model it describes, rather than wherever the module-level default points.
    print(f"Saved feature spec -> {spec.save(OUTPUTS_DIR / 'feature_spec.json')}")

    lim = float(max(y_true.max(), y_pred.max())) + 5
    plt.figure(figsize=(6, 6))
    plt.plot([0, lim], [0, lim], "k--", lw=1, label="perfect prediction")
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    plt.xlabel("True RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title(
        f"twin-turbofan baseline - {subset}\n"
        f"RMSE={metrics['rmse']:.1f}   PHM score={metrics['phm_score']:.0f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "pred_vs_true.png", dpi=120)
    print(f"Saved plot -> {OUTPUTS_DIR / 'pred_vs_true.png'}")


if __name__ == "__main__":
    import argparse

    from .config import add_config_args, setup_logging

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--subset", default=None)
    add_config_args(p)
    a = p.parse_args()

    _cfg = load_config(a.config, subset=a.subset, logging_level=a.log_level)
    setup_logging(_cfg.logging_level, _cfg.logging_timestamps)
    main(subset=a.subset, config=a.config)
