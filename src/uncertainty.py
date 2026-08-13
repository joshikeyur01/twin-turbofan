"""Prediction intervals and conservative RUL estimates via split conformal prediction.

**Motivation, from the Tier 1 error analysis.** The RandomForest's mean residual is
essentially zero, yet 71% of its PHM score comes from late predictions and 28% from a
single engine. The PHM penalty is exponential and asymmetric, so the aggregate is driven
by worst-case lateness, not average error. Chasing RMSE cannot fix that — but
deliberately predicting *lower* than the conditional mean can, because it trades a
little RMSE for a large reduction in the exponentially-punished tail.

This module quantifies that trade-off honestly rather than asserting it.

**Method — split conformal.** Training engines are split by unit into a fit set and a
calibration set. The model is fit on the fit set only; residuals ``pred − true`` are
collected on the held-out calibration engines; the empirical quantiles of those
residuals then (a) shift point predictions to a chosen quantile and (b) form two-sided
prediction intervals. Because calibration engines are never trained on, the resulting
intervals carry a finite-sample coverage guarantee that needs no distributional
assumption.

One caveat stated plainly: calibration uses *all* cycles of the calibration engines,
whereas the test set observes each engine at a single truncated point. The two
distributions over life stage are similar but not identical, so coverage is approximate
rather than exact. Sampling one random cycle per calibration engine would match the
test distribution more closely at the cost of much higher variance.

Run:
    python -m src.uncertainty

Writes outputs/uncertainty.md, outputs/uncertainty.png
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data_loader import last_cycle_rows, load_cmapss, using_real_data  # noqa: E402
from .evaluate import phm_score, rmse  # noqa: E402
from .features import build_xy  # noqa: E402
from .models import make_model  # noqa: E402
from .paths import OUTPUTS_DIR  # noqa: E402

# Residual quantiles to subtract from the point prediction.
#
# Sign convention, since it is easy to get backwards: residual = pred − true, and the
# adjusted prediction is ``point − quantile(residuals, q)``. A HIGH q is therefore the
# conservative direction — subtracting a large positive residual pushes the estimate
# EARLIER. q ≈ 0.5 reproduces the point prediction, and low q makes the twin optimistic
# (later), which is the unsafe direction. The range below deliberately spans both so the
# trade-off curve is mapped rather than assumed.
QUANTILES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
# Nominal coverage levels for two-sided intervals.
COVERAGES = [0.80, 0.90, 0.95]
CAL_FRAC = 0.3


def _verdict(phm_gain: float, best, base) -> str:
    """State whether the conservative-shift hypothesis actually held.

    Written as a function so the report cannot drift into claiming a win the numbers
    do not support — the text is derived from the measured gain.
    """
    rmse_delta = abs(float(best["rmse"]) - float(base["rmse"]))
    why_small = (
        "A *uniform* offset can only weakly exploit the tail: the PHM asymmetry "
        "(exp(d/10) late vs exp(-d/13) early) is mildly lopsided, so shifting every "
        "engine earlier pays a small early penalty on all of them to shave a few large "
        "late errors, and the many small costs largely cancel the few large gains. "
        "Attacking the tail properly needs *per-engine* conservatism scaled to that "
        "engine's own predictive uncertainty, not one global constant."
    )

    if phm_gain < 2.0:
        return (
            "**Hypothesis not supported.** The point prediction is already essentially "
            f"PHM-optimal — the best shift recovers only {phm_gain:+.1f}%. " + why_small
        )
    if phm_gain < 15.0:
        return (
            f"**Hypothesis only weakly supported.** The best shift (q={best['quantile']:.2f}, "
            f"{best['offset']:+.2f} cycles) improves PHM by {phm_gain:.1f}% for a "
            f"{rmse_delta:.2f}-cycle RMSE cost. The direction predicted by the error "
            "analysis is real and the trade is favourable, but the effect is far smaller "
            f"than the tail concentration suggested. {why_small}"
        )
    return (
        f"**Hypothesis supported.** Shifting to q={best['quantile']:.2f} cuts PHM by "
        f"{phm_gain:.1f}% for a {rmse_delta:.2f}-cycle RMSE cost, confirming the "
        "asymmetric metric rewards deliberate earliness."
    )


def split_by_engine(df, frac: float, seed: int = 42):
    """Hold out ``frac`` of engines (whole engines only — no cycle leakage)."""
    units = np.sort(df["unit"].unique())
    perm = np.random.default_rng(seed).permutation(units)
    n = max(1, int(round(len(units) * frac)))
    held = set(perm[:n].tolist())
    mask = df["unit"].isin(held)
    return df[~mask].copy(), df[mask].copy(), sorted(held)


def main(subset: str = "FD001", seed: int = 42):
    train, test = load_cmapss(subset)
    train_f, test_f, feat_cols, _ = build_xy(train, test)

    fit_df, cal_df, cal_units = split_by_engine(train_f, CAL_FRAC, seed)
    model, model_name = make_model()
    model.fit(fit_df[feat_cols].to_numpy(), fit_df["RUL"].to_numpy())
    print(f"model: {model_name}")
    print(
        f"engines: fit={fit_df['unit'].nunique()} calibration={len(cal_units)} "
        f"| calibration rows={len(cal_df):,}"
    )

    # Calibration residuals, pred - true (positive = late), on unseen engines.
    cal_pred = np.clip(model.predict(cal_df[feat_cols].to_numpy()), 0, None)
    cal_resid = cal_pred - cal_df["RUL"].to_numpy()

    test_last = last_cycle_rows(test_f)
    y_true = test_last["RUL"].to_numpy()
    y_point = np.clip(model.predict(test_last[feat_cols].to_numpy()), 0, None)

    base = {
        "label": "point (conditional mean)",
        "quantile": 0.5,
        "offset": 0.0,
        "rmse": round(rmse(y_true, y_point), 3),
        "phm": round(phm_score(y_true, y_point), 1),
        "pct_late": round(100.0 * float((y_point - y_true > 0).mean()), 1),
    }
    print(f"\nbaseline point prediction: RMSE {base['rmse']}  PHM {base['phm']}")

    # --- Quantile shift. High q subtracts a large residual -> earlier, safer call. ---
    rows = []
    for q in QUANTILES:
        offset = float(np.quantile(cal_resid, q))
        y_q = np.clip(y_point - offset, 0, None)
        resid = y_q - y_true
        rows.append(
            {
                "quantile": q,
                "offset": round(offset, 2),
                "rmse": round(rmse(y_true, y_q), 3),
                "phm": round(phm_score(y_true, y_q), 1),
                "pct_late": round(100.0 * float((resid > 0).mean()), 1),
                "mean_resid": round(float(resid.mean()), 2),
            }
        )
        print(
            f"  q={q:.2f}  offset={offset:+7.2f}  RMSE={rows[-1]['rmse']:7.3f}  "
            f"PHM={rows[-1]['phm']:8.1f}  late={rows[-1]['pct_late']:5.1f}%"
        )

    qdf = pd.DataFrame(rows)
    best_phm = qdf.loc[qdf["phm"].idxmin()]

    # --- Two-sided conformal intervals and their empirical coverage. ---
    cov_rows = []
    for cov in COVERAGES:
        alpha = 1.0 - cov
        lo_off = float(np.quantile(cal_resid, alpha / 2))
        hi_off = float(np.quantile(cal_resid, 1 - alpha / 2))
        lo = np.clip(y_point - hi_off, 0, None)
        hi = np.clip(y_point - lo_off, 0, None)
        inside = float(((y_true >= lo) & (y_true <= hi)).mean())
        cov_rows.append(
            {
                "nominal": cov,
                "empirical": round(100 * inside, 1),
                "mean_width": round(float((hi - lo).mean()), 1),
            }
        )
        print(
            f"  {cov:.0%} interval -> empirical {100 * inside:5.1f}%  "
            f"mean width {cov_rows[-1]['mean_width']:.1f} cycles"
        )

    cdf = pd.DataFrame(cov_rows)
    fig_path = _plot(qdf, base, cdf, y_true, y_point, cal_resid, subset)

    real = using_real_data()
    phm_gain = 100 * (base["phm"] - best_phm["phm"]) / base["phm"]
    rmse_cost = 100 * (best_phm["rmse"] - base["rmse"]) / base["rmse"]

    lines = [
        f"# Uncertainty & conservative prediction — {subset}",
        "",
        f"**Data: {'real NASA C-MAPSS' if real else 'SYNTHETIC fallback (plumbing only)'}**",
        "",
        f"Model: {model_name}. Split conformal: {fit_df['unit'].nunique()} fit engines, "
        f"{len(cal_units)} calibration engines, scored on each test engine's last cycle.",
        "",
        "## The hypothesis being tested",
        "",
        "The error analysis found the PHM score is driven by worst-case *lateness*, not",
        "average error — 71% of it came from late predictions and 28% from one engine.",
        "PHM punishes a late call exponentially harder than an early one, so the natural",
        "hypothesis was that shifting every prediction earlier would buy a large PHM",
        "reduction for a small RMSE cost. The table below tests that claim.",
        "",
        "Sign convention: adjusted = point − quantile(residuals, q), so **high q is the",
        "conservative (earlier) direction** and q≈0.5 reproduces the point prediction.",
        "",
        "## Quantile shift",
        "",
        f"Baseline point prediction: RMSE **{base['rmse']}**, PHM **{base['phm']}**, "
        f"{base['pct_late']}% late.",
        "",
        qdf.to_markdown(index=False),
        "",
        "### Read-out",
        "",
        f"- Best PHM at **q={best_phm['quantile']:.2f}** "
        f"(offset {best_phm['offset']:+.2f} cycles): "
        f"PHM **{best_phm['phm']}** vs {base['phm']} baseline "
        f"(**{phm_gain:+.1f}%**), RMSE {best_phm['rmse']} vs {base['rmse']} "
        f"({rmse_cost:+.1f}%).",
        f"- Late predictions at that setting: {best_phm['pct_late']}% "
        f"(baseline {base['pct_late']}%).",
        "",
        _verdict(phm_gain, best_phm, base),
        "",
        "## Conformal prediction intervals",
        "",
        cdf.to_markdown(index=False),
        "",
        "Empirical coverage close to nominal indicates the calibration set is a fair",
        "proxy for the test distribution. A systematic shortfall would point at the",
        "life-stage mismatch noted in the module docstring: calibration sees every cycle",
        "of its engines, while the test set observes one truncated point per engine.",
        "",
        f"Figure: `{fig_path.name}`",
        "",
    ]
    (OUTPUTS_DIR / "uncertainty.md").write_text("\n".join(lines))
    (OUTPUTS_DIR / "uncertainty.json").write_text(
        json.dumps({"baseline": base, "quantiles": rows, "coverage": cov_rows}, indent=2)
    )
    print(f"\nSaved -> outputs/uncertainty.md, {fig_path.name}")


def _plot(qdf, base, cdf, y_true, y_point, cal_resid, subset):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0][0]
    ax.plot(qdf["quantile"], qdf["phm"], "o-", color="tab:red", label="PHM score")
    ax.axhline(base["phm"], color="k", ls="--", lw=1, label="point prediction")
    ax.set_xlabel("residual quantile used")
    ax.set_ylabel("PHM score")
    ax.set_title("PHM vs quantile (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(qdf["quantile"], qdf["rmse"], "o-", color="tab:blue", label="RMSE")
    ax.axhline(base["rmse"], color="k", ls="--", lw=1, label="point prediction")
    ax.set_xlabel("residual quantile used")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs quantile — the cost side of the trade")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.hist(cal_resid, bins=40, color="tab:orange", edgecolor="k", alpha=0.85)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("calibration residual (pred − true)")
    ax.set_ylabel("count")
    ax.set_title("Calibration residuals on held-out engines")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][1]
    order = np.argsort(y_true)
    best = qdf.loc[qdf["phm"].idxmin()]
    ax.plot(y_true[order], y_true[order], "k--", lw=1, label="perfect")
    ax.scatter(y_true, y_point, s=22, alpha=0.7, label="point", edgecolor="k")
    ax.scatter(
        y_true,
        np.clip(y_point - best["offset"], 0, None),
        s=22,
        alpha=0.7,
        color="tab:green",
        label=f"q={best['quantile']:.2f} shifted",
        edgecolor="k",
    )
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("Points below the diagonal are early (safe)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"twin-turbofan — uncertainty & conservative prediction ({subset})")
    fig.tight_layout()
    path = OUTPUTS_DIR / "uncertainty.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
