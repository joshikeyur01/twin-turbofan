# Sweep finalists re-ranked across seeds — LSTM (FD001)

**Data: SYNTHETIC (plumbing only)**

Top 3 configurations from `sweep_lstm.json`, each re-run at 5 seeds [42, 43, 44, 45, 46]. Ranked by **mean** validation RMSE, with a Student-t 95% confidence interval on that mean (t=2.776 at df=4).

| original rank | seq_len | hidden | lr | val (1 seed) | val mean ±95% CI | val 95% CI | val range | test mean ±95% CI | PHM mean | vs. winner |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 50 | 128 | 0.0003 | 7.841 | **4.574 ±0.476** | [4.098, 5.050] | 0.98 | 4.639 ±1.041 | 21.0 | **tied** (CI overlaps) |
| 2 | 50 | 128 | 0.003 | 8.785 | **4.059 ±0.942** | [3.117, 5.000] | 1.943 | 4.027 ±0.307 | 16.2 | — (winner) |
| 3 | 50 | 128 | 0.001 | 9.253 | **4.139 ±0.417** | [3.722, 4.555] | 0.774 | 4.352 ±0.538 | 18.5 | **tied** (CI overlaps) |

## Verdict

**The selection does not survive re-seeding.** The single-seed sweep chose seq=50, hidden=128, lr=0.0003; averaged over 5 seeds the best configuration is instead seq=50, hidden=128, lr=0.003, though 2 finalist(s) remain indistinguishable from it at 95% CI (seq=50, hidden=128, lr=0.001; seq=50, hidden=128, lr=0.0003). Finalists differ by 0.515 validation RMSE against interval half-widths of up to 0.942, so the original single-seed ranking was reading noise. Any downstream result that used the single-seed winner inherits that arbitrariness.

### Reading the overlap column

Overlap is a **conservative** test. Non-overlapping 95% intervals do imply a significant difference; overlapping ones do **not** prove the configurations are equivalent — two means can overlap and still differ at p<0.05. `separated` is therefore a stronger claim than `tied` is, and `tied` should be read as *unresolved by 5 seeds*, not as *proven equal*.

Why this check exists: `src/sweep.py` ranks on one seed, and `src/variance.py` measured across-seed RMSE spreads of 1.7–6.1 — larger than the gaps between these finalists. Selecting a configuration on a single seed is the same mistake as reporting a metric on one, one level up. The interval, unlike that raw spread, tightens as seeds are added, which is what makes a `separated` verdict reachable at all.
