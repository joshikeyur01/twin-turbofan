# Benchmark report

Consolidated results with the exact commands that produce them, so any number here can be
regenerated and checked.

> **These are synthetic-data results.** The real NASA C-MAPSS dataset was not present for
> this run, so the pipeline fell back to `data/synthetic/`. That generator emits smooth
> monotonic drift with no real fault modes. Treat every figure as a correctness check on
> the pipeline, **not** as a benchmark, and do not compare it to published C-MAPSS
> numbers. `make all` regenerates everything once `data/CMAPSSData/` is populated.

## Environment

| | |
|---|---|
| Python | 3.11.1, native **arm64** (not the x86_64 conda build — see worklog) |
| numpy / pandas | 2.4.6 / 3.0.5 |
| scikit-learn | 1.9.0 |
| torch | 2.13.0, MPS available |
| Device (sequence models) | `mps` |

Reproducibility note: the RandomForest baseline produces byte-identical metrics on
scikit-learn 1.0.2 and 1.9.0 (`random_state=42`), and the full test suite passes on both
numpy 1.21/pandas 1.4 and numpy 2.4/pandas 3.0.

## Protocol

Identical for every model in every table:

- Piecewise-linear RUL target, capped at **125** cycles.
- Split **by engine**; no engine's cycles appear on both sides of any split.
- Scored on **each test engine's last recorded cycle** (50 engines on FD001).
- Both **RMSE** and the asymmetric **PHM'08 score** reported; predictions clipped at 0.
- Sequence models: validation over all windows of held-out engines, early stopping with
  best-validation checkpoint restored, hyperparameters selected on validation only.

---

## 1. Baseline

```bash
python -m src.train_baseline
```

| model | RMSE | PHM | n engines |
|---|---|---|---|
| RidgeFallback (numpy, no sklearn) | 12.759 | 101.2 | 50 |
| RandomForestRegressor | **12.513** | 115.4 | 50 |

The forest improves RMSE but *worsens* PHM — the metrics disagree because PHM is
asymmetric and tail-dominated. This is the single best argument for always reporting both.
The same disagreement held on generator v1 (11.067 / 79.2 vs 10.309 / 85.0), but those
numbers are **not** comparable to this table: v2 changed the data. See
[`../outputs/synthetic_fidelity.md`](../outputs/synthetic_fidelity.md).

## 2. Error analysis

```bash
python -m src.error_analysis
```

Residual = predicted − true; positive means predicted late.

| statistic | value |
|---|---|
| mean residual | −0.46 cycles |
| median residual | −1.07 cycles |
| range | −19.5 … +29.9 |
| predicted late | 22/50 (44%) |

Bias by life stage:

| true RUL bin | n | mean residual | std |
|---|---|---|---|
| 0–25 (near failure) | 7 | +5.81 | 4.21 |
| 25–50 | 9 | +1.74 | 8.51 |
| 50–75 | 10 | +5.09 | 13.71 |
| 75–100 | 11 | −1.83 | 18.43 |
| 100–125 (cap plateau) | 13 | −8.47 | 6.84 |

The near-failure bin is **late** here (+5.81), the direction PHM punishes. On generator v1
it was −1.53 — early, the safe direction — and the docs said so. The sign flipped with the
regenerated data, not with any model change.

PHM concentration:

| | share of total PHM |
|---|---|
| worst engine (true 86 → pred 116) | 16.4% |
| worst 6 engines | 51% |
| all late predictions | 68% |

Artifacts: `outputs/residuals.png`, `outputs/trajectories.png`, `outputs/error_analysis.md`

## 3. Feature ablation

```bash
python -m src.ablation
```

| arm | features | RMSE | PHM | % late |
|---|---|---|---|---|
| raw sensors only | 21 | 10.574 | 84.4 | 44.0 |
| **+ rolling w=3** | 63 | **10.249** | **79.0** | 46.0 |
| + rolling w=5 *(shipped default)* | 63 | 10.309 | 85.0 | 44.0 |
| + rolling w=10 | 63 | 10.555 | 100.3 | 52.0 |
| + rolling w=20 | 63 | 11.489 | 128.3 | 46.0 |

`w=3` beats the shipped default on both metrics. The default is intentionally unchanged —
tuning it here would fit the generator, not the problem.

Caveat specific to this table: it is a **generator v1** artifact and has not been re-run.
It was produced when the selector kept every sensor, which is why its raw arm has 21
features and its rolling arms 63. On v2 `select_informative_sensors` drops **6 of 21**,
so the arms would be 15 and 45, and the RMSE column is not comparable to §1 above.
Re-run with `make ablation`; see
[`../outputs/synthetic_fidelity.md`](../outputs/synthetic_fidelity.md).

Artifacts: `outputs/ablation.png`, `outputs/ablation.md`

## 4. Hyperparameter sweep — LSTM

```bash
python -m src.sweep --arch lstm
```

27 configurations (seq_len × hidden × lr), ranked by **validation** RMSE. Selected:
`seq_len=50, hidden=128, lr=3e-4`.

| seq_len | hidden | lr | val RMSE | test RMSE | test PHM |
|---|---|---|---|---|---|
| **50** | **128** | **3e-4** | **7.841** | **7.804** | **48.3** |
| 50 | 128 | 3e-3 | 8.785 | 7.830 | 42.1 |
| 50 | 128 | 1e-3 | 9.253 | 9.316 | 68.6 |
| 20 | 128 | 1e-3 | 10.631 | 11.125 | 100.0 |
| 30 | 64 | 1e-3 *(scaffold defaults)* | 11.844 | 11.037 | 92.6 |
| 50 | 32 | 1e-3 *(worst)* | 12.729 | 15.131 | 169.8 |

The `seq_len=50, hidden=128` corner dominates: all three learning rates there land at
7.8–9.3 RMSE while every other cell sits at 10.9–15.1. That corner is on the **grid
boundary**, so capacity looks unsaturated and the sweep should be extended rather than
treated as converged.

Artifacts: `outputs/sweep_lstm.md`, `outputs/sweep_lstm.json`

### 4b. All three architectures, each on its own grid

```bash
python -m src.sweep --arch gru
python -m src.sweep --arch cnn
```

| arch | selected config | val RMSE | test RMSE | test PHM | best epoch |
|---|---|---|---|---|---|
| LSTM | seq=50 hidden=128 lr=3e-4 | 7.841 | 7.804 | 48.3 | 21 / 29 |
| GRU | seq=50 hidden=128 lr=3e-4 | 6.865 | 7.489 | 46.3 | **60 / 60** |
| CNN | **seq=20 hidden=32** lr=3e-4 | 10.713 | 10.758 | 79.6 | 50 / 58 |

Two findings here, both of which change how the comparison table should be read.

**The CNN wants the opposite corner of the grid.** Both recurrent models pick the longest
window and the largest hidden size (50/128); the CNN picks the shortest and smallest
(20/32). Run at the LSTM's config it scored 14.285 RMSE / 135.8 PHM; on its own it scores
**10.758 / 79.6** — a 25% RMSE and 41% PHM improvement from hyperparameters alone. That
moves it from "much worse than the RandomForest" to level on RMSE and clearly better on PHM.
The earlier table's "untuned, not worse" caveat was correct and understated, and
`src/compare.py` now reads each architecture's own sweep rather than sharing one config.

**The GRU hit the epoch cap** (best epoch 60 of 60), so it had not converged. This also
explains a discrepancy I first misdiagnosed as nondeterminism — see §9.

Artifacts: `outputs/sweep_gru.md`, `outputs/sweep_cnn.md`

## 5. Model comparison — FD001

```bash
python -m src.compare --epochs 80 --patience 10
```

All sequence models at the sweep-selected configuration.

Each architecture uses its own configuration selected on **seed-averaged** validation
(`rerank_<arch>.json`, see §4c), with results averaged over **5 seeds** (42–46) at a
100-epoch budget. Cells are `mean ±95% CI` — a Student-t interval on the mean (t=2.776 at
df=4), which replaced `±half-range` when the seed count went 3 → 5: half-range describes the
runs rather than the mean and widens as seeds are added, so it could never separate anything.

| model | config | source | RMSE ↓ | PHM ↓ | across-seed RMSE range |
|---|---|---|---|---|---|
| **GRU** | 50/128/1e-3 | seed-averaged | **2.985** ±0.941 | **10.9** ±3.4 | 1.920 |
| LSTM | 50/128/3e-3 | seed-averaged | 4.122 ±0.431 | 16.7 ±2.5 | 0.815 |
| ATTENTION | 50/128/3e-4 | seed-averaged | 4.435 ±2.568 | 24.0 ±28.6 | 5.083 |
| CNN | 30/32/1e-3 | seed-averaged | 9.304 ±0.916 | 63.1 ±18.7 | 2.000 |
| RandomForest | — | — | 12.645 ±0.105 | 118.4 ±2.5 | **0.231** |

**Separation at 95% CI:** GRU, LSTM and ATTENTION all overlap each other — the top three are
tied on this evidence. CNN and RandomForest are each separated from every other model. The
prose in the rest of this section still describes the 3-seed / v1-synthetic ranking and has
**not** been re-derived; see `outputs/comparison.md` for the current verdicts.

**The GRU leads on the mean of both metrics** — 76% better RMSE and 91% better PHM than the
forest. It got there only after §4c: its single-seed sweep had picked `lr=3e-4`, and the
seed-averaged ranking prefers `lr=1e-3`, worth 0.36 RMSE (3.341 → 2.985).

**Read the margins against the intervals, though.** The GRU's lead over the LSTM is 1.137 RMSE
against across-seed ranges of 1.92 and 0.815, and their 95% intervals overlap — as do
ATTENTION's with both. Five seeds separate the *bottom* of this table cleanly (CNN and the
forest) and leave the *top* three tied. "Ahead on the mean, unresolved on the evidence" is the
honest phrasing; overlap being a conservative test, it is an open question rather than a
demonstrated tie.

**Two caveats that are properties of this table, not of the models:**

- **`attention` was swept last, and its optimum turned out to be the default it already had**
  (50/128/3e-4). For a while it was the only architecture running on shared defaults while the
  others used their own grids; since correct selection had been worth 0.36 RMSE to the GRU, I
  expected its number to be understated. It is not — the row was fair by luck rather than design.
  What the sweep *did* reveal is that attention varies by **2.393** validation RMSE at its own
  best configuration, and its comparison interval (±2.568) is the widest of any model here —
  wide enough that it cannot be separated from the GRU despite a mean 1.45 RMSE worse.
- **The CNN's configuration changed and, at 5 seeds, so did its standing.** It is now one of only
  two models separated from every other at 95% CI — not because it improved, but because its
  interval (±0.916) is narrow relative to its distance from the leaders.

**The RandomForest still owns reliability.** Its across-seed range is 0.231 against 0.8–5.1 for
every neural model — 4× to 22× tighter. For a periodically retrained maintenance model that
predictability is worth real money, and it is invisible in any single-seed table.

Only FD001 exists synthetically, so this is one column of an intended four.

Artifacts: `outputs/comparison.md`, `outputs/comparison.json`

## 6. Uncertainty — split conformal

```bash
python -m src.uncertainty
```

70 fit engines / 30 calibration engines. Baseline differs from §1 because it is fit on 70
engines, not 100. Convention: `adjusted = point − quantile(residuals, q)`, so **high q is
the conservative (earlier) direction**.

| q | offset | RMSE | PHM | % late |
|---|---|---|---|---|
| — *(point)* | 0.00 | 10.517 | 82.1 | 46 |
| 0.30 | −4.48 | 11.491 | 115.8 | 68 |
| 0.50 | −0.55 | 10.540 | 84.4 | 48 |
| 0.60 | −0.02 | 10.518 | 82.1 | 46 |
| **0.70** | **+1.86** | 10.654 | **77.5** | 38 |
| 0.80 | +6.78 | 12.428 | 87.2 | 22 |
| 0.90 | +16.11 | 19.041 | 182.4 | 6 |
| 0.95 | +22.22 | 23.834 | 302.4 | 4 |

Best shift recovers **5.6% of PHM for a 1.3% RMSE cost** — the predicted direction, far
smaller than the predicted magnitude. See `docs/writeup.md` §5 for why tail concentration
does not imply a uniform offset helps.

Interval coverage:

| nominal | empirical | mean width |
|---|---|---|
| 80% | 84.0% | 30.0 cycles |
| 90% | 92.0% | 39.4 cycles |
| 95% | 98.0% | 45.7 cycles |

Over-covering, as expected: calibration observes every cycle of its engines while the test
set observes one truncated point per engine, so calibration residuals are more dispersed
than test-time residuals.

Artifacts: `outputs/uncertainty.png`, `outputs/uncertainty.md`

## 6b. Per-engine vs global conservatism

```bash
python -m src.uncertainty_per_engine
```

The follow-up §6 called for: `adjusted_i = point_i − k · sigma_i`, with `sigma_i` the
spread of the RandomForest's per-tree predictions for that engine. Each setting is compared
against the uniform offset producing the **same mean shift**, so the comparison isolates
*allocation* from *amount*.

Precondition, measured rather than assumed: `corr(sigma, |error|) = **+0.546**`. If the
spread did not rank error, no weighting built on it could help.

| k | mean shift | per-engine PHM | uniform PHM (same shift) | delta |
|---|---|---|---|---|
| 0.00 | 0.00 | 82.1 | 82.1 | — |
| **0.25** | **2.20** | **74.4** | 77.2 | **−2.8** |
| 0.50 | 4.40 | 75.4 | 78.9 | −3.5 |
| 0.75 | 6.60 | 84.8 | 86.3 | −1.5 |
| 1.00 | 8.80 | 103.1 | 99.1 | +4.0 |
| 1.50 | 13.20 | 168.1 | 141.4 | +26.7 |
| 2.00 | 17.60 | 283.7 | 207.4 | +76.3 |
| 3.00 | 26.40 | 795.5 | 423.7 | +371.8 |

Calibration engines independently selected **k=0.25**, which is also the best test setting.
There it scores PHM **74.4** vs 82.1 unadjusted — a **9.4%** gain, against the **5.6%** the
best *global* offset achieved in §6 — while also beating the uniform shift of the same mean
amount by 2.8 PHM. So the earlier diagnosis was right and the earlier *implementation* was
the limitation.

The gain only exists for mild `k`. Sigma reaches ~16 cycles, so `k >= 1` drags the
uncertain engines tens of cycles early and the exponential early-penalty term takes over —
which is why per-engine is *worse* than uniform in the bottom half of the table.

Artifacts: `outputs/uncertainty_per_engine.png`, `outputs/uncertainty_per_engine.md`

## 7. Live twin

```bash
python -m src.telemetry simulate --unit 1
python -m src.make_demo_gif --unit 1
streamlit run app.py
```

Engine 1, 222 cycles, alert threshold 25:

| | |
|---|---|
| first alert | cycle **213** |
| final estimate | 16.8 (actual 22) |
| divergence | consistently negative (conservative) |

All three surfaces — CLI simulate, Streamlit dashboard, GIF renderer — produce identical
numbers, because all three consume the same `OnlineFeatureBuilder`. The fleet view's
last-known RUL for engine 1 (16.8) agrees with the replay.

Most of the mid-life divergence (down to ≈ −30 cycles around cycle 135) is an artefact of
the piecewise-linear cap: the target is pinned at 125 until cycle ~120 while the model
reports continuous degradation.

Artifacts: `outputs/live_twin_engine1.png`, `docs/demo.gif`

## 8b. Ensemble — does blending help?

```bash
python -m src.ensemble --arch gru
```

Blend `w * GRU + (1 - w) * forest`, both fit on the same 80 engines, weight chosen on 20
held-out validation engines.

| w on GRU | val RMSE | test RMSE | test PHM |
|---|---|---|---|
| 0.0 *(forest only)* | 11.566 | 10.307 | 80.2 |
| 0.5 | 8.151 | 7.714 | 48.2 |
| 0.7 | 7.241 | 7.220 | 42.9 |
| 0.8 | 6.943 | **7.132** | **42.3** |
| 0.9 | 6.770 | 7.156 | 43.0 |
| **1.0 *(GRU only — selected)*** | **6.730** | 7.292 | 44.9 |

**No — blending does not help.** Validation monotonically prefers more GRU and selects
`w=1.0`, the sequence model alone.

This is worth dwelling on, because the *test* column has a different opinion: `w=0.8` scores
7.132 / 42.3, beating the GRU alone on both metrics. Had the weight been chosen on test, this
section would report a 2% RMSE and 6% PHM "improvement" from ensembling. It would not be a
real finding — with 50 test engines and eleven weights to choose from, something will beat
both endpoints. Validation and test disagreeing about the optimum *is* the evidence that the
apparent gain is selection noise. The weight is chosen on validation for exactly this reason.

## 9. Reproducibility: a discrepancy that was not noise

The GRU sweep reported test RMSE **7.489** for `seq=50 hidden=128 lr=3e-4`, while an earlier
comparison run reported **7.292** for what looked like the same config and seed. My first
explanation was MPS run-to-run variance. It was wrong.

The two runs had different epoch budgets. Re-running each reproduces its number exactly:

| budget | best epoch | epochs run | RMSE | PHM |
|---|---|---|---|---|
| ≤60, patience 8 *(sweep)* | 60 — hit the cap | 60 | 7.489 | 46.3 |
| ≤80, patience 10 *(comparison)* | 73 | 80 | 7.292 | 44.9 |

So two things are true that were not before: training here is **exactly reproducible at a
fixed seed on MPS**, and the GRU is **epoch-limited rather than converged** — 7.292 is a
floor, not its best. `src/variance.py` measures same-seed repeats separately from
different-seed runs so a between-model claim can be checked against the actual spread
instead of quoted from a single run.

## 8c. Run-to-run variance

```bash
python -m src.variance --archs lstm gru --repeats 5
```

**This section deliberately quotes no numbers.** It used to reproduce the variance table by
hand, and those figures were computed on the **pre-v2 synthetic data** — before six sensors
were made genuinely constant to match real FD001 (§3). The data changed underneath them, so
every one of them became false while continuing to look authoritative.

`src/variance.py` already writes a full report to **`outputs/variance.md`**, regenerated from
the artifact each run. Duplicating its numbers here bought nothing and created a second place
for them to go stale, so this section points at it instead. That is the general fix for
hand-copied figures, not a special case.

What does not change with the data, and is therefore safe to state here:

- **The design.** Two conditions are measured separately — repeats at a *fixed* seed (kernel
  determinism) and repeats across *different* seeds (sensitivity to weight init, batch order,
  and the by-engine train/val split). They answer different questions and are reported apart.
- **The decision rule.** A difference between two models counts only if it exceeds the
  across-seed spread. Where intervals overlap, the report says the models are not separated
  rather than ranking them.
- **Why it exists.** A single seed was reused throughout this project's early results, and it
  flattered every neural model. That is what motivated seed-averaging both the metrics (§5) and
  the configuration selection (§4c).

## 8d. Attention model — best single-seed result, and why that is not the headline

```bash
python -m src.interpret --epochs 120     # trains AttentionRegressor + interpretability
python -m src.variance --archs attention gru --repeats 3
```

At seed 42 the attention model is the best result in the project: **RMSE 6.248, PHM 31.0**,
against the GRU's 7.292 / 44.9. Across seeds that advantage disappears, and something more
useful takes its place:

| arch | seed 42 | seed-mean | RMSE std | RMSE spread | PHM std |
|---|---|---|---|---|---|
| attention | **6.248** | 8.210 | 2.667 | **5.733** | 32.5 |
| GRU | 7.292 | 8.312 | 0.747 | 1.767 | 10.6 |

**The means are indistinguishable** (8.210 vs 8.312, a 0.102 gap against a 5.733 spread), so
the 14% single-seed win is not a property of the architecture.

**But attention is 3.2× less stable**, and that is the finding worth keeping. Its across-seed
RMSE spread is 5.733 against the GRU's 1.767, and its PHM std is 32.5 against 10.6. Seed 42
flatters it by 1.962 RMSE versus its own seed-average; the GRU's equivalent penalty is 1.020.
For a maintenance model that is a liability rather than a curiosity: a best case you cannot
reliably reproduce is worth less than a slightly worse one you can. On this evidence the GRU
is the better *engineering* choice despite the attention model owning the best number in the
report.

The attention model still earns its place — the pooling weights are what make §10 possible,
and interpretability was the reason it was added rather than accuracy.

## 10. Interpretability — what the twin looks at

Same run as §8d (`outputs/interpretability.md`).

**Attention over the window.** The most recent 13 of 50 cycles hold **62.1%** of the total
weight, against **26.0%** for uniform attention; the oldest cycle receives ≈0, and the profile
rises monotonically toward the scoring point, crossing the uniform line about 16 cycles back.

The expectation was written down before looking: monotonic degradation means the newest
readings carry the most information and older ones largely repeat it. A flat profile would have
meant the model was averaging and the sequence structure was doing nothing.

**Permutation importance** (ΔRMSE when a feature is shuffled across scoring windows):

| feature | ΔRMSE |
|---|---|
| `s7` | +2.454 |
| `s20` | +2.383 |
| `s4` | +1.757 |
| `s8` | +1.404 |
| `s12` | +1.321 |
| `s4_rmean` | +1.031 |

Raw sensors take the top five places and the first engineered feature appears sixth —
consistent with §3, where rolling statistics bought only ~3% RMSE. The model is reading the raw
sensor trajectory, and the smoothed versions are largely redundant to it.

Read as signal *groups*, not clean per-sensor attribution: correlated inputs share
responsibility under permutation, and C-MAPSS sensors move together during HPC degradation.

## 8. Test suite

```bash
make check    # ruff + mypy + doc-number check + pytest
```

| | |
|---|---|
| tests | **247** — 242 in the default run, 5 marked `slow` (they train models end-to-end) |
| coverage | **75%** overall; core library modules 91-100% |
| ruff / black / mypy | clean |
| mutation tests | 13 deliberate source breaks, 13 caught |
| dependency-light guarantee | verified by `make check-minimal`, not just asserted |
| doc numbers | **337** re-derived from `outputs/` by `make check-docs`, not trusted |

Every number in this report is quoted from an artifact in `outputs/`, which means every one
of them can go stale when a sweep is re-run. `src/validate_docs.py` recomputes them and
fails the build on any disagreement, reporting `file:line` with expected vs actual. It also
flags any number-bearing artifact in `outputs/` that no check reads, so a new experiment
cannot add unguarded figures to these pages.

It earned its place immediately: §5 listed the CNN's parameter count from the configuration
the *single-seed* sweep selected (`20/32`, 10,401 params) while the config column had already
been corrected to the seed-averaged `20/64` (26,881), and three across-seed ranges had been
truncated rather than rounded.

Coverage exceptions, stated rather than hidden: `dashboard.py` is 0% because it needs a
live streamlit runtime — it was verified manually in a browser instead, and its numbers
were cross-checked against the CLI. `telemetry.py` is 39% because the MQTT publish and
subscribe paths need a broker; the broker-free `simulate` path, which shares every line of
scoring logic with the subscriber, is covered.

Adding the entry-point smoke tests found a genuine bug: `FeatureSpec.save()` bound its
output path as a **default argument**, which Python evaluates once at import. Redirecting
`OUTPUTS_DIR` was therefore silently ignored and the feature spec was written to the real
`outputs/` directory regardless of configuration.

The mutation pass initially caught 12 of 13. The survivor — replacing the CNN's global
time-pooling with a single-timestep read — was shape-identical and so invisible to shape
assertions. Now covered by a test asserting the prediction responds to a mid-window
perturbation.
