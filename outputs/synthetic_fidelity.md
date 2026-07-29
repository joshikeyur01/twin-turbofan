# Synthetic generator v2 — fidelity fix and its cost

`src/generate_synthetic.py` v1 gave its non-trending sensors `sigma=0.5` noise. That
variance is ~250x `features.variance_threshold`, so `select_informative_sensors` kept
**0 of 21** sensors out of the drop path — while on real FD001 it drops **6 of 21**
(`s1, s5, s10, s16, s18, s19` are literally constant there). Every synthetic number this
project has ever produced was therefore computed on 21 sensors and 63 feature columns,
where the real data gives 15 and 45.

v2 emits those six as exact constants (no noise, variance 0). The fallback now takes the
same feature-selection path as the real data.

```bash
python -m src.generate_synthetic --config config.yaml   # regenerate data/synthetic/
python -m src.train_baseline                            # metrics.json, feature_spec.json
python -m src.error_analysis                            # trajectories.png, residuals.png
```

## Baseline: before vs after

RandomForest (`n_estimators=200, min_samples_leaf=5, random_state=42`), scored on each
test engine's last cycle. Identical protocol in both columns; only the data changed.

| quantity | v1 (sigma=0.5 on flat sensors) | v2 (flat sensors constant) |
|---|---|---|
| informative sensors | 21 of 21 | **15** of 21 |
| feature columns | 63 | **45** |
| RMSE | 10.309 | **12.513** |
| PHM score | 85.0 | **115.4** |
| mean residual | +0.09 cycles | **−0.46** cycles |
| median residual | −1.04 cycles | −1.07 cycles |
| residual range | −21.4 … +32.0 | −19.5 … +29.9 |
| predicted late | 22/50 (44%) | 22/50 (44%) |
| worst single engine's share of PHM | 27.8% | 16.4% |

Bias by life stage (residual = predicted − true; positive = predicted late):

| true RUL bin | v1 n | v1 mean | v2 n | v2 mean |
|---|---|---|---|---|
| 0–25 (near failure) | 5 | −1.53 | 7 | **+5.81** |
| 25–50 | 12 | +3.48 | 9 | +1.74 |
| 50–75 | 6 | +9.17 | 10 | +5.09 |
| 75–100 | 8 | −2.60 | 11 | −1.83 |
| 100–125 (cap plateau) | 19 | −3.37 | 13 | −8.47 |

**One qualitative finding reversed.** On v1 the twin was slightly *early* near failure
(−1.53), which the README called operationally favourable — the safe direction in the
regime where the maintenance call is made. On v2 it is **late** there (+5.81), the
direction PHM punishes. That claim has been corrected in the docs rather than kept.

## Where the 2.2 RMSE went

The baseline got worse, but almost none of that is the sensor change. Removing the noise
draws also shifts the generator's RNG stream, so v2 is a different sample of engine
lifetimes, truncation points and noise. Three runs separate the two effects — B is the
v2 data with `sigma=0.5` noise added back onto the six flat sensors, so it has v2's draw
and v1's sensor structure:

| run | data | sensors | RMSE | PHM |
|---|---|---|---|---|
| A | v1 | 21 | 10.309 | 85.0 |
| B | v2 draw, v1 flat-sensor noise | 21 | 12.324 | 114.7 |
| C | v2 as shipped | 15 | 12.513 | 115.4 |

- A → B = **+2.015 RMSE (91% of the shift)**: a different draw from the same generator.
- B → C = **+0.189 RMSE (9%)**: the actual fidelity fix. That is inside the forest's own
  run-to-run spread — across seeds 42/43/44 on v2 it scores 12.607 ±0.081 (range
  **0.161**); on v1 the recorded range was 0.190.

So dropping six pure-noise sensors cost the model nothing measurable, which is the
expected result: they carried no signal in v1 either. The headline move is a resample,
not a harder problem. Both are equally uninformative about real C-MAPSS.

## These numbers invalidate prior synthetic benchmarks

Every metric recorded before this change was produced on v1 data. They are not comparable
to anything produced after it, and averaging or ranking across the boundary is invalid.

Regenerated on v2 (safe to read together):

- `metrics.json`, `baseline.pkl`, `feature_spec.json`, `pred_vs_true.png`
- `error_analysis.md`, `trajectories.png`, `residuals.png`

Also regenerated on v2, because the fix invalidated them too: `docs/demo.gif` and
`live_twin_engine48.png`. The live-twin demo moved from engine 1 to engine 48 — on v2's
draw, test engine 1 stops at cycle 71 with 97 cycles of life left and never crosses the
alert threshold, so it no longer demonstrates anything. Engine 48 runs 179 cycles and
alerts at 163.

**Still on v1 — stale, and not comparable to the above:**

- `ablation.json` / `.md` / `.png` — its "raw sensors only" arm is 21 features; on v2 it
  would be 15
- `comparison.json` / `.md`, `variance.json` / `.md`, `ensemble.json` / `.md`
- `sweep_{lstm,gru,cnn,attention}.*`, `rerank_{lstm,gru,attention}.*`
- `uncertainty.json` / `.md`, `uncertainty_per_engine.json` / `.md`
- `interpretability.json` / `.md`

**Mixed, and worth a second look:** `rerank_cnn.*` and `published_comparison.*` were
written by a concurrent session *after* `data/synthetic/` was regenerated, so they trained
on v2 while the other three rerank artifacts are v1. Any table that reads them alongside
`rerank_{lstm,gru,attention}.json` is comparing across the data change.

The doc tables quoting those artifacts still match them, so `check-docs` reports no drift
against any of them — the inconsistency is between *artifacts*, not between an artifact
and its prose, and nothing in this repo checks for that. Cheap ones come back with
`make all`; the sweeps, reranks, variance study and ensemble need sequence-model training
and were out of budget for this change.

## Recommendation

**Re-run all comparisons on real C-MAPSS FD001 once available.** Do not spend more
sequence-model compute re-establishing a model ranking on v2 synthetic data: v2 is a
better *shape* match to the real dataset — same sensor count, same drop count, same
feature width — but its degradation is still a monotonic drift with no fault modes, so
its rankings remain plumbing checks. Drop the real files into `data/CMAPSSData/`, run
`make all` plus the sweeps, and treat everything above as superseded.
