# Sweep finalists re-ranked across seeds — ATTENTION (FD001)

**Data: SYNTHETIC (plumbing only)**

Top 3 configurations from `sweep_attention.json`, each re-run at 5 seeds [42, 43, 44, 45, 46]. Ranked by **mean** validation RMSE, with a Student-t 95% confidence interval on that mean (t=2.776 at df=4).

| original rank | seq_len | hidden | lr | val (1 seed) | val mean ±95% CI | val 95% CI | val range | test mean ±95% CI | PHM mean | vs. winner |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 50 | 128 | 0.0003 | 5.857 | **4.114 ±1.272** | [2.842, 5.386] | 2.393 | 4.118 ±1.650 | 19.7 | — (winner) |
| 2 | 50 | 64 | 0.001 | 8.074 | **4.913 ±1.531** | [3.381, 6.444] | 3.322 | 4.994 ±1.782 | 24.7 | **tied** (CI overlaps) |
| 3 | 50 | 64 | 0.0003 | 9.254 | **4.327 ±1.009** | [3.318, 5.336] | 2.055 | 4.153 ±1.732 | 21.3 | **tied** (CI overlaps) |

## Verdict

**The winner survives, but 2 of 2 rivals are indistinguishable from it at 95% CI.** The same configuration (seq=50, hidden=128, lr=0.0003) still ranks first on the mean over 5 seeds, but its interval [2.842, 5.386] overlaps: seq=50, hidden=64, lr=0.0003; seq=50, hidden=64, lr=0.001. Finalists are separated by 0.799 validation RMSE against interval half-widths of up to 1.531 (raw across-seed range up to 3.322). Read this as 'the evidence does not separate these configurations', not as 'the sweep picked the best one'.

### Reading the overlap column

Overlap is a **conservative** test. Non-overlapping 95% intervals do imply a significant difference; overlapping ones do **not** prove the configurations are equivalent — two means can overlap and still differ at p<0.05. `separated` is therefore a stronger claim than `tied` is, and `tied` should be read as *unresolved by 5 seeds*, not as *proven equal*.

Why this check exists: `src/sweep.py` ranks on one seed, and `src/variance.py` measured across-seed RMSE spreads of 1.7–6.1 — larger than the gaps between these finalists. Selecting a configuration on a single seed is the same mistake as reporting a metric on one, one level up. The interval, unlike that raw spread, tightens as seeds are added, which is what makes a `separated` verdict reachable at all.
