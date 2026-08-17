# Ensemble — RandomForest + GRU (FD001)

**Data: SYNTHETIC (plumbing only)**

Blend: `w * gru + (1 - w) * forest`. Both models fit on the same 80 engines; weight chosen on 20 held-out validation engines by RMSE, **never on test**.

|   w_seq |   val_rmse |   rmse |   phm |   pct_late |
|--------:|-----------:|-------:|------:|-----------:|
|     0   |     12.804 | 12.506 | 131.3 |         46 |
|     0.1 |     11.633 | 11.335 | 103.6 |         46 |
|     0.2 |     10.48  | 10.175 |  81.5 |         48 |
|     0.3 |      9.35  |  9.027 |  64   |         46 |
|     0.4 |      8.255 |  7.9   |  49.8 |         46 |
|     0.5 |      7.211 |  6.802 |  38.4 |         46 |
|     0.6 |      6.241 |  5.75  |  29.1 |         46 |
|     0.7 |      5.387 |  4.776 |  21.7 |         48 |
|     0.8 |      4.713 |  3.936 |  16.3 |         46 |
|     0.9 |      4.304 |  3.335 |  13.3 |         42 |
|     1   |      4.236 |  3.114 |  11.7 |         38 |

## Read-out

- Forest alone (w=0.0): RMSE **12.506**, PHM **131.3**
- GRU alone (w=1.0): RMSE **3.114**, PHM **11.7**
- Validation-selected blend (w=1.0): RMSE **3.114**, PHM **11.7**

**No, blending does not help.** Validation picked w=1.0 — i.e. the sequence model alone. The blend curve offered nothing the better single model did not already provide, which is the honest outcome when one model dominates the other on every engine rather than making complementary errors.

Figure: `ensemble.png`
