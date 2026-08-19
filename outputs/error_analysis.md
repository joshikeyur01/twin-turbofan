# Error analysis — FD001

Model: loaded outputs/baseline.pkl
Scored on each engine's last cycle (50 engines).

- RMSE: **12.513**
- PHM score: **115.4**
- Mean residual (pred − true): **-0.46** cycles
- Predicted late (residual > 0): **22/50** engines (44%)

## Bias by life stage

Residual = predicted − true. Positive means the twin over-estimates remaining
life, i.e. predicts failure later than it happens — the direction PHM punishes.

| bin            |   count |   mean |   std |
|:---------------|--------:|-------:|------:|
| (-0.001, 25.0] |       7 |   5.81 |  4.21 |
| (25.0, 50.0]   |       9 |   1.74 |  8.51 |
| (50.0, 75.0]   |      10 |   5.09 | 13.71 |
| (75.0, 100.0]  |      11 |  -1.83 | 18.43 |
| (100.0, 125.0] |      13 |  -8.47 |  6.84 |

## Figures

- `trajectories.png` — predicted vs true RUL trajectories (engines [12, 47, 21, 30, 35, 45])
- `residuals.png` — residual vs true RUL + bias by life stage
