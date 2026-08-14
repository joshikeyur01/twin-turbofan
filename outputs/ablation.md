# Feature ablation — FD001

Model: RandomForestRegressor (sklearn). Scored on each engine's last cycle.

| arm              |   n_features |   rmse |   phm |   mean_resid |   pct_late |   fit_s |
|:-----------------|-------------:|-------:|------:|-------------:|-----------:|--------:|
| raw sensors only |           15 | 11.985 | 108.9 |        -0.24 |         46 |     5   |
| + rolling w=3    |           45 | 12.205 | 115.2 |        -0.49 |         44 |    11   |
| + rolling w=5    |           45 | 12.513 | 115.4 |        -0.46 |         44 |    11   |
| + rolling w=10   |           45 | 12.628 | 103.9 |        -1.65 |         44 |    10.7 |
| + rolling w=20   |           45 | 10.115 |  74   |        -0.92 |         44 |     9.9 |

## Read-out

- Raw-sensor baseline: RMSE **11.985**, PHM **108.9** (15 features).
- Best RMSE: **+ rolling w=20** at **10.115** (+15.6% vs raw).
- Best PHM: **+ rolling w=20** at **74.0** (+32.0% vs raw).

Rolling features triple the feature count (15 → 45), so a marginal gain may not justify the cost.

Figure: `ablation.png`
