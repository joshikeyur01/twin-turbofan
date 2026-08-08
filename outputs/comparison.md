# Model × dataset comparison

**Data: SYNTHETIC fallback**

> These numbers exercise the pipeline; they are **not** benchmark results.
> The synthetic generator produces smooth monotonic drift with no real fault
> modes, so absolute values and model ranking may both differ on real data.

Protocol: split by engine, score each engine's last cycle, RUL capped at 125.
Sequence models: up to 100 epochs, early stopping patience 10, best-val checkpoint restored. Each architecture uses **its own** validation-selected hyperparameters where a sweep exists (see the `config_source` column) — a single shared config penalises architectures whose inductive bias wants a different window or capacity.

Averaged over **5 seeds** ([42, 43, 44, 45, 46]); cells are `mean ±95% CI` — a Student-t interval on the mean, not the older half-range. Single-seed numbers are not reported as headline results: the across-seed spread (see `outputs/variance.md`) is comparable to or larger than the gaps between architectures, so one seed ranks noise — and seed 42 happens to be a favourable draw. Half-range was retired as the headline ± because it describes the runs rather than the mean, and widens rather than tightens as seeds are added; it survives as the `across-seed range` column, which answers a different and still-useful question.

## RMSE (lower is better)

| model        | FD001         |
|:-------------|:--------------|
| ATTENTION    | 4.435 ±2.568  |
| CNN          | 9.304 ±0.916  |
| GRU          | 2.985 ±0.941  |
| LSTM         | 4.122 ±0.431  |
| RandomForest | 12.645 ±0.105 |

## PHM score (lower is better; asymmetric, punishes late predictions)

| model        | FD001      |
|:-------------|:-----------|
| ATTENTION    | 24.0 ±28.6 |
| CNN          | 63.1 ±18.7 |
| GRU          | 10.9 ±3.4  |
| LSTM         | 16.7 ±2.5  |
| RandomForest | 118.4 ±2.5 |

## Statistical separation (95% CI on the mean RMSE)

With **5 seeds** the interval is `mean ± t·s/√n` with t=2.776 (df=4). Models whose intervals overlap are marked **indistinguishable at 95% CI**.

| rank | model | RMSE mean ±95% CI | 95% CI | across-seed range | verdict |
|---|---|---|---|---|---|
| 1 | GRU | 2.985 ±0.941 | [2.044, 3.926] | 1.920 | **indistinguishable at 95% CI** from LSTM, ATTENTION |
| 2 | LSTM | 4.122 ±0.431 | [3.692, 4.553] | 0.815 | **indistinguishable at 95% CI** from GRU, ATTENTION |
| 3 | ATTENTION | 4.435 ±2.568 | [1.867, 7.003] | 5.083 | **indistinguishable at 95% CI** from GRU, LSTM |
| 4 | CNN | 9.304 ±0.916 | [8.388, 10.220] | 2.000 | separated from every other model |
| 5 | RandomForest | 12.645 ±0.105 | [12.540, 12.749] | 0.231 | separated from every other model |

**GRU leads on the mean but is not separated from LSTM, ATTENTION at 95% CI** on FD001. On this evidence the top of the table is a tie, not a ranking.

Fully separated from everything else: CNN, RandomForest.

Overlap is a conservative test: non-overlapping intervals do imply a significant difference, but overlapping ones do **not** prove equivalence — two means can overlap and still differ at p<0.05. Read `indistinguishable` as *unresolved by 5 seeds*, not as *proven equal*.

## Per-seed detail

| model        | subset   |   rmse_mean |   rmse_min |   rmse_max |   phm_mean |   phm_min |   phm_max |   n_seeds | rmse_str      | phm_str    | rmse_ci95        |   rmse_range |
|:-------------|:---------|------------:|-----------:|-----------:|-----------:|----------:|----------:|----------:|:--------------|:-----------|:-----------------|-------------:|
| ATTENTION    | FD001    |       4.435 |      2.986 |      8.069 |      24.02 |      11.3 |      65.1 |         5 | 4.435 ±2.568  | 24.0 ±28.6 | [1.867, 7.003]   |        5.083 |
| CNN          | FD001    |       9.304 |      8.097 |     10.097 |      63.14 |      45.5 |      85.5 |         5 | 9.304 ±0.916  | 63.1 ±18.7 | [8.388, 10.220]  |        2     |
| GRU          | FD001    |       2.985 |      2.022 |      3.942 |      10.92 |       7.3 |      14.2 |         5 | 2.985 ±0.941  | 10.9 ±3.4  | [2.044, 3.926]   |        1.92  |
| LSTM         | FD001    |       4.122 |      3.746 |      4.561 |      16.66 |      14.9 |      19.2 |         5 | 4.122 ±0.431  | 16.7 ±2.5  | [3.692, 4.553]   |        0.815 |
| RandomForest | FD001    |      12.645 |     12.513 |     12.744 |     118.42 |     115.4 |     120.3 |         5 | 12.645 ±0.105 | 118.4 ±2.5 | [12.540, 12.749] |        0.231 |

## Every run

| model        | subset   |   seed |   rmse |   phm |   pct_late |   n_params |   train_s |   seq_len |   hidden |       lr | config_source                         |
|:-------------|:---------|-------:|-------:|------:|-----------:|-----------:|----------:|----------:|---------:|---------:|:--------------------------------------|
| RandomForest | FD001    |     42 | 12.513 | 115.4 |         44 |        nan |      12.8 |       nan |      nan | nan      | nan                                   |
| LSTM         | FD001    |     42 |  4.407 |  18.5 |         50 |     225857 |      40.5 |        50 |      128 |   0.003  | rerank_lstm.json (seed-averaged)      |
| GRU          | FD001    |     42 |  2.823 |  10.7 |         40 |     170433 |     179.2 |        50 |      128 |   0.001  | rerank_gru.json (seed-averaged)       |
| CNN          | FD001    |     42 |  9.291 |  53.9 |         52 |       8673 |      23   |        30 |       32 |   0.001  | rerank_cnn.json (seed-averaged)       |
| ATTENTION    | FD001    |     42 |  4.087 |  14.8 |         36 |     187074 |     294.1 |        50 |      128 |   0.0003 | rerank_attention.json (seed-averaged) |
| RandomForest | FD001    |     43 | 12.633 | 119.6 |         42 |        nan |      12.3 |       nan |      nan | nan      | nan                                   |
| LSTM         | FD001    |     43 |  4.005 |  14.9 |         48 |     225857 |      34.1 |        50 |      128 |   0.003  | rerank_lstm.json (seed-averaged)      |
| GRU          | FD001    |     43 |  3.524 |  13   |         44 |     170433 |     122.1 |        50 |      128 |   0.001  | rerank_gru.json (seed-averaged)       |
| CNN          | FD001    |     43 |  9.497 |  65.1 |         52 |       8673 |      14.3 |        30 |       32 |   0.001  | rerank_cnn.json (seed-averaged)       |
| ATTENTION    | FD001    |     43 |  3.543 |  14.2 |         40 |     187074 |     137.2 |        50 |      128 |   0.0003 | rerank_attention.json (seed-averaged) |
| RandomForest | FD001    |     44 | 12.674 | 119.4 |         44 |        nan |      11.1 |       nan |      nan | nan      | nan                                   |
| LSTM         | FD001    |     44 |  3.893 |  15.6 |         44 |     225857 |      52.6 |        50 |      128 |   0.003  | rerank_lstm.json (seed-averaged)      |
| GRU          | FD001    |     44 |  3.942 |  14.2 |         48 |     170433 |      98.9 |        50 |      128 |   0.001  | rerank_gru.json (seed-averaged)       |
| CNN          | FD001    |     44 | 10.097 |  85.5 |         48 |       8673 |      17.7 |        30 |       32 |   0.001  | rerank_cnn.json (seed-averaged)       |
| ATTENTION    | FD001    |     44 |  8.069 |  65.1 |         36 |     187074 |     224.1 |        50 |      128 |   0.0003 | rerank_attention.json (seed-averaged) |
| RandomForest | FD001    |     45 | 12.66  | 117.4 |         44 |        nan |      13.5 |       nan |      nan | nan      | nan                                   |
| LSTM         | FD001    |     45 |  3.746 |  15.1 |         56 |     225857 |      38.1 |        50 |      128 |   0.003  | rerank_lstm.json (seed-averaged)      |
| GRU          | FD001    |     45 |  2.615 |   9.4 |         52 |     170433 |      80.7 |        50 |      128 |   0.001  | rerank_gru.json (seed-averaged)       |
| CNN          | FD001    |     45 |  9.538 |  65.7 |         52 |       8673 |      21.6 |        30 |       32 |   0.001  | rerank_cnn.json (seed-averaged)       |
| ATTENTION    | FD001    |     45 |  2.986 |  11.3 |         48 |     187074 |     118.3 |        50 |      128 |   0.0003 | rerank_attention.json (seed-averaged) |
| RandomForest | FD001    |     46 | 12.744 | 120.3 |         46 |        nan |      10.8 |       nan |      nan | nan      | nan                                   |
| LSTM         | FD001    |     46 |  4.561 |  19.2 |         60 |     225857 |      38.1 |        50 |      128 |   0.003  | rerank_lstm.json (seed-averaged)      |
| GRU          | FD001    |     46 |  2.022 |   7.3 |         50 |     170433 |     109.1 |        50 |      128 |   0.001  | rerank_gru.json (seed-averaged)       |
| CNN          | FD001    |     46 |  8.097 |  45.5 |         44 |       8673 |      16.7 |        30 |       32 |   0.001  | rerank_cnn.json (seed-averaged)       |
| ATTENTION    | FD001    |     46 |  3.491 |  14.7 |         56 |     187074 |     284   |        50 |      128 |   0.0003 | rerank_attention.json (seed-averaged) |
