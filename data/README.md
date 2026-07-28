# Data

## Real dataset — NASA C-MAPSS Turbofan Degradation

The canonical aerospace predictive-maintenance benchmark. Simulated run-to-failure
data for a fleet of turbofan engines.

**Where to get it:** search for *"NASA C-MAPSS Turbofan Engine Degradation
Simulation Data Set"*. It is published by the **NASA Prognostics Center of
Excellence (PCoE)** data repository and is mirrored on Kaggle (e.g. the
*"NASA Turbofan Jet Engine Data Set"*). Confirm the current link before downloading.

**Install:** unzip and place the text files here:

```
data/CMAPSSData/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── ... (FD002 / FD003 / FD004)
```

Start with **FD001** (100 train engines, 100 test engines, one operating
condition, one fault mode — HPC degradation).

### File schema (26 space-separated columns)

| col | name | meaning |
|-----|------|---------|
| 1 | `unit` | engine id |
| 2 | `cycle` | time in operating cycles |
| 3–5 | `os1..os3` | operational settings |
| 6–26 | `s1..s21` | sensor measurements |

- **train_*** — each engine runs until it fails.
- **test_*** — each engine stops some time *before* failure.
- **RUL_*** — the true Remaining Useful Life at each test engine's last cycle.

## Synthetic fallback

Don't want to download yet? Generate C-MAPSS-shaped synthetic data so the pipeline
runs end-to-end:

```
python -m src.generate_synthetic   # writes data/synthetic/
```

The loader prefers `data/CMAPSSData/` and falls back to `data/synthetic/`.
Synthetic results are for **plumbing only** — not a benchmark.
