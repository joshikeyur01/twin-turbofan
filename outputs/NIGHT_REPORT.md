# Overnight run report

Elapsed **1.89 h** of a 8 h budget. 9 ran, 0 skipped, 0 failed.

Judged against the requirements in [`NIGHT_RUN.md`](../NIGHT_RUN.md).

## Tasks

| task | requirement | result | minutes | detail |
|---|---|---|---|---|
| `variance_lstm_gru` | R2, R3 | ok | 49.7 | — |
| `variance_attention_cnn` | R2, R3 | ok | 52.2 | — |
| `ablation` | R2 | ok | 0.9 | — |
| `error_analysis` | R2 | ok | 0.0 | — |
| `uncertainty` | R2 | ok | 0.2 | — |
| `uncertainty_per_engine` | R2 | ok | 0.2 | — |
| `interpret` | R2 | ok | 5.3 | — |
| `ensemble` | R2 | ok | 4.7 | — |
| `demo_gif` | R2 | ok | 0.1 | — |

## Acceptance gate (R1)

`python -m src.validate_docs` → **FAIL**

> FAIL — 337 numbers across 92 specs, 179 drift, 3 warnings

## Requirements

| id | requirement | status |
|---|---|---|
| R1 | docs derivable from artifacts | **NOT MET** |
| R2 | no pre-v2 results | met |
| R3 | claims separated from noise | enforced in the rewritten sections; see gate |
| R4 | artifacts before prose | structural (each script persists JSON first) |
| R5 | 8h budget | met — stopped at 1.89 h |
| R6 | resumable | met — 0 task(s) skipped as already fresh |
| R7 | fixed queue, no agent loop | met — queue is a literal in `src/overnight.py` |
| R8 | fail soft, never silent | met — 0 failure(s) listed above |
| R9 | commit per task | met — one commit per completed task |
| R10 | repo scope only | met — no pushes, no writes outside the repo |
| R11 | data not regenerated | met — `data/synthetic/` untouched |
