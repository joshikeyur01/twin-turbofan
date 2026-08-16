# Interpretability — attention model (FD001)

**Data: SYNTHETIC (plumbing only)**

`AttentionRegressor` (GRU encoder + additive attention pooling), seq_len=50, hidden=128, lr=0.0003. Test RMSE **4.087**, PHM **14.8** (best epoch 71 of 86).

## Which cycles does the estimate rest on?

The prediction is the attention-weighted sum of encoder states, so these weights are
the readout itself, not a proxy for it. Each row of weights sums to 1.

- First cycle in the window: **0.0000**
- Last (most recent) cycle: **0.0676**
- Most recent quarter of the window holds **53.8%** of total attention (uniform attention would give 26.0%)

So attention is **concentrated on recent cycles**, which is what monotonic degradation should produce: the newest readings carry the most information about current health, and older ones mostly repeat it.

## Which sensors does it rely on?

Permutation importance: RMSE increase when a feature column is shuffled across the
scoring windows. Model-agnostic, and measures reliance rather than attention.

| feature | ΔRMSE when shuffled |
|---|---|
| `s8` | +6.736 |
| `s13` | +5.518 |
| `s7_rmean` | +4.501 |
| `s14_rmean` | +3.554 |
| `s2` | +2.445 |
| `s15_rmean` | +2.161 |
| `s6` | +2.112 |
| `s21_rmean` | +2.008 |
| `s7` | +1.934 |
| `s20` | +1.926 |
| `s4` | +1.692 |
| `s15` | +1.676 |

Top feature: **`s8`** (+6.736 RMSE).

8 of 45 features change RMSE by ≤0.001 when destroyed — the model does not use them. On synthetic data that is unsurprising: the generator only lets a subset of sensors drift, and `select_informative_sensors` cannot drop the rest because they still carry noise (see the ablation caveat).

**Read the ranking as signal *groups*, not clean per-sensor attribution.** Correlated
inputs share responsibility under permutation, so shuffling one of two near-duplicate
sensors understates both — and C-MAPSS sensors move together during HPC degradation.

Figure: `interpretability.png`
