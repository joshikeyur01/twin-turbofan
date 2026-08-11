# Sweep finalists re-ranked across seeds — CNN (FD001)

**Data: SYNTHETIC (plumbing only)**

Top 3 configurations from `sweep_cnn.json`, each re-run at 5 seeds [42, 43, 44, 45, 46]. Ranked by **mean** validation RMSE, with a Student-t 95% confidence interval on that mean (t=2.776 at df=4).

| original rank | seq_len | hidden | lr | val (1 seed) | val mean ±95% CI | val 95% CI | val range | test mean ±95% CI | PHM mean | vs. winner |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20 | 32 | 0.0003 | 10.713 | **8.370 ±0.295** | [8.075, 8.664] | 0.615 | 9.611 ±0.648 | 66.6 | **tied** (CI overlaps) |
| 2 | 30 | 32 | 0.001 | 10.756 | **8.084 ±0.675** | [7.409, 8.758] | 1.316 | 9.315 ±0.888 | 63.7 | — (winner) |
| 3 | 20 | 64 | 0.0003 | 10.907 | **8.491 ±0.609** | [7.882, 9.100] | 1.21 | 9.594 ±0.971 | 65.6 | **tied** (CI overlaps) |

## Verdict

**The selection does not survive re-seeding.** The single-seed sweep chose seq=20, hidden=32, lr=0.0003; averaged over 5 seeds the best configuration is instead seq=30, hidden=32, lr=0.001, though 2 finalist(s) remain indistinguishable from it at 95% CI (seq=20, hidden=32, lr=0.0003; seq=20, hidden=64, lr=0.0003). Finalists differ by 0.407 validation RMSE against interval half-widths of up to 0.675, so the original single-seed ranking was reading noise. Any downstream result that used the single-seed winner inherits that arbitrariness.

### Reading the overlap column

Overlap is a **conservative** test. Non-overlapping 95% intervals do imply a significant difference; overlapping ones do **not** prove the configurations are equivalent — two means can overlap and still differ at p<0.05. `separated` is therefore a stronger claim than `tied` is, and `tied` should be read as *unresolved by 5 seeds*, not as *proven equal*.

Why this check exists: `src/sweep.py` ranks on one seed, and `src/variance.py` measured across-seed RMSE spreads of 1.7–6.1 — larger than the gaps between these finalists. Selecting a configuration on a single seed is the same mistake as reporting a metric on one, one level up. The interval, unlike that raw spread, tightens as seeds are added, which is what makes a `separated` verdict reachable at all.
