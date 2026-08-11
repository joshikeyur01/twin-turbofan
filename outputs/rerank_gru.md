# Sweep finalists re-ranked across seeds — GRU (FD001)

**Data: SYNTHETIC (plumbing only)**

Top 3 configurations from `sweep_gru.json`, each re-run at 5 seeds [42, 43, 44, 45, 46]. Ranked by **mean** validation RMSE, with a Student-t 95% confidence interval on that mean (t=2.776 at df=4).

| original rank | seq_len | hidden | lr | val (1 seed) | val mean ±95% CI | val 95% CI | val range | test mean ±95% CI | PHM mean | vs. winner |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 50 | 128 | 0.0003 | 6.865 | **3.980 ±1.328** | [2.652, 5.307] | 2.647 | 3.341 ±2.209 | 16.5 | **tied** (CI overlaps) |
| 2 | 50 | 128 | 0.001 | 7.735 | **3.356 ±0.467** | [2.889, 3.823] | 0.929 | 2.985 ±0.941 | 10.9 | — (winner) |
| 3 | 30 | 128 | 0.003 | 7.966 | **4.808 ±0.666** | [4.142, 5.474] | 1.391 | 4.605 ±0.986 | 20.4 | separated |

## Verdict

**The selection does not survive re-seeding.** The single-seed sweep chose seq=50, hidden=128, lr=0.0003; averaged over 5 seeds the best configuration is instead seq=50, hidden=128, lr=0.001, though 1 finalist(s) remain indistinguishable from it at 95% CI (seq=50, hidden=128, lr=0.0003). Finalists differ by 1.452 validation RMSE against interval half-widths of up to 1.328, so the original single-seed ranking was reading noise. Any downstream result that used the single-seed winner inherits that arbitrariness.

### Reading the overlap column

Overlap is a **conservative** test. Non-overlapping 95% intervals do imply a significant difference; overlapping ones do **not** prove the configurations are equivalent — two means can overlap and still differ at p<0.05. `separated` is therefore a stronger claim than `tied` is, and `tied` should be read as *unresolved by 5 seeds*, not as *proven equal*.

Why this check exists: `src/sweep.py` ranks on one seed, and `src/variance.py` measured across-seed RMSE spreads of 1.7–6.1 — larger than the gaps between these finalists. Selecting a configuration on a single seed is the same mistake as reporting a metric on one, one level up. The interval, unlike that raw spread, tightens as seeds are added, which is what makes a `separated` verdict reachable at all.
