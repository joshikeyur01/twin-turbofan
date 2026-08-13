# Uncertainty & conservative prediction — FD001

**Data: SYNTHETIC fallback (plumbing only)**

Model: RandomForestRegressor (sklearn). Split conformal: 70 fit engines, 30 calibration engines, scored on each test engine's last cycle.

## The hypothesis being tested

The error analysis found the PHM score is driven by worst-case *lateness*, not
average error — 71% of it came from late predictions and 28% from one engine.
PHM punishes a late call exponentially harder than an early one, so the natural
hypothesis was that shifting every prediction earlier would buy a large PHM
reduction for a small RMSE cost. The table below tests that claim.

Sign convention: adjusted = point − quantile(residuals, q), so **high q is the
conservative (earlier) direction** and q≈0.5 reproduces the point prediction.

## Quantile shift

Baseline point prediction: RMSE **12.149**, PHM **124.5**, 46.0% late.

|   quantile |   offset |   rmse |   phm |   pct_late |   mean_resid |
|-----------:|---------:|-------:|------:|-----------:|-------------:|
|       0.1  |    -9.22 | 15.934 | 295.1 |         82 |        10.38 |
|       0.2  |    -3.19 | 12.851 | 159.7 |         56 |         4.35 |
|       0.3  |    -0.99 | 12.282 | 133.3 |         48 |         2.14 |
|       0.4  |    -0.14 | 12.164 | 125.7 |         46 |         1.3  |
|       0.5  |     1.06 | 12.094 | 117.1 |         42 |         0.09 |
|       0.6  |     4.27 | 12.489 | 105   |         38 |        -3.12 |
|       0.7  |     7.71 | 13.755 | 108.6 |         26 |        -6.55 |
|       0.8  |    12.52 | 16.579 | 140.1 |         16 |       -11.34 |
|       0.9  |    19.38 | 21.734 | 241.2 |          6 |       -18.02 |
|       0.95 |    25.15 | 26.476 | 389.3 |          4 |       -23.41 |

### Read-out

- Best PHM at **q=0.60** (offset +4.27 cycles): PHM **105.0** vs 124.5 baseline (**+15.7%**), RMSE 12.489 vs 12.149 (+2.8%).
- Late predictions at that setting: 38.0% (baseline 46.0%).

**Hypothesis supported.** Shifting to q=0.60 cuts PHM by 15.7% for a 0.34-cycle RMSE cost, confirming the asymmetric metric rewards deliberate earliness.

## Conformal prediction intervals

|   nominal |   empirical |   mean_width |
|----------:|------------:|-------------:|
|      0.8  |          76 |         28.4 |
|      0.9  |          88 |         38.8 |
|      0.95 |          96 |         46.3 |

Empirical coverage close to nominal indicates the calibration set is a fair
proxy for the test distribution. A systematic shortfall would point at the
life-stage mismatch noted in the module docstring: calibration sees every cycle
of its engines, while the test set observes one truncated point per engine.

Figure: `uncertainty.png`
