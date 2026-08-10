# Run-to-run variance

**Data: SYNTHETIC (plumbing only)**

Config held fixed at `seq_len=50, hidden=128, lr=0.0003`; 5 runs per condition.

| arch | condition | RMSE mean | RMSE std | RMSE spread | PHM mean | PHM std |
|---|---|---|---|---|---|---|
| ATTENTION | same seed | 4.087 | 0.0 | 0.0 | 14.8 | 0.0 |
| ATTENTION | different seeds | 4.435 | 1.85 | 5.083 | 24.02 | 20.58 |
| CNN | same seed | 12.223 | 0.0 | 0.0 | 94.5 | 0.0 |
| CNN | different seeds | 12.574 | 0.716 | 2.095 | 103.64 | 11.202 |

## Verdict

The ATTENTION's RMSE advantage (8.139) **exceeds** the across-seed spread (5.083), so the ranking survives re-running and is worth reporting as a real difference.

**ATTENTION is markedly less stable than CNN** — across-seed spread 5.083 versus 2.095, a factor of 2.4. Its single-seed number is a correspondingly weaker guide to what a fresh training run will produce, which is a liability in its own right: a best case you cannot reliably reproduce is worth less than a slightly worse one you can.

Same-seed repeats came out **identical** (spread 0.0), so seeding Python, NumPy and torch is sufficient for this workload on this device — measured, not assumed. MPS and cuDNN give no general bitwise guarantee, so the condition is kept in the study to catch that changing.

Figure: `variance.png`
