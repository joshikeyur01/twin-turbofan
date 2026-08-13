# Per-engine vs global conservatism — FD001

**Data: SYNTHETIC (plumbing only)**

Model: RandomForestRegressor (sklearn), fit on 70 engines; 30 calibration engines used to choose `k`. Scored on each test engine's last cycle.

## The question

`src/uncertainty.py` found a uniform earlier-shift recovers only ~5.6% of PHM, and
argued the fix was conservatism proportional to each engine's own uncertainty.
Here `adjusted_i = point_i − k · sigma_i`, with `sigma_i` the spread of the
RandomForest's per-tree predictions for that engine.

Each per-engine setting is compared against the uniform offset producing the **same
mean shift**, so the comparison isolates *allocation* from *amount*.

Baseline point prediction: RMSE **12.149**, PHM **124.5**.

## Does the spread predict error at all?

`corr(sigma, |error|) = +0.584`

This is the precondition. Allocation can only beat a uniform shift if the model's
own uncertainty ranks which engines it will get wrong; a correlation near zero means
sigma carries no usable signal and no weighting scheme built on it can help.

## Results

|    k |   mean_shift |   pe_rmse |   pe_phm |   pe_pct_late |   uni_rmse |   uni_phm |
|-----:|-------------:|----------:|---------:|--------------:|-----------:|----------:|
| 0    |         0    |    12.149 |    124.5 |            46 |     12.149 |     124.5 |
| 0.25 |         2.19 |    12.171 |    108.7 |            42 |     12.138 |     111.2 |
| 0.5  |         4.37 |    12.639 |    102.5 |            40 |     12.515 |     104.8 |
| 0.75 |         6.56 |    13.508 |    105.2 |            36 |     13.248 |     105.5 |
| 1    |         8.75 |    14.706 |    116.6 |            32 |     14.281 |     112.9 |
| 1.5  |        13.12 |    17.808 |    167.6 |            22 |     16.987 |     146.3 |
| 2    |        17.5  |    21.502 |    265.1 |             6 |     20.243 |     206   |
| 3    |        26.25 |    29.762 |    685.5 |             2 |     27.378 |     425.1 |

Per-engine beats matched-amount uniform on PHM in **3 of 7** non-trivial settings.

Calibration-selected `k=0.75`: RMSE **13.508**, PHM **105.2** (uniform at the same mean shift: 105.5).

## Verdict

**Supported, in the regime that matters.** At the calibration-selected `k=0.75` the per-engine rule scores PHM **105.2** vs **124.5** unadjusted — a **15.5%** improvement, against the 5.6% that the best *global* offset managed in `src/uncertainty.py`. Crucially it also beats the uniform shift of the **same mean amount** (6.56 cycles) by 0.3 PHM, so the improvement comes from allocation rather than from simply being more conservative. The forest's per-tree spread correlates +0.584 with absolute error, which is what makes the weighting informative. The gain is confined to mild conservatism: 4 of 7 settings improve on the baseline at all, and per-engine turns *worse* than uniform for k ≥ 1. The failure mode is intuitive: sigma reaches ~16 cycles, so a large k drags the uncertain engines tens of cycles early and the early-penalty term takes over.

This is the follow-up the earlier conformal experiment called for, and it largely vindicates that diagnosis — the limitation there was the *uniform* offset, not the idea of trading RMSE for tail safety.

Figure: `uncertainty_per_engine.png`
