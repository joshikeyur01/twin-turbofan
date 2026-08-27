# This project vs published C-MAPSS FD001 results

**Data: SYNTHETIC fallback** — our side of every table below comes from `outputs/comparison.json`, seeds [42, 43, 44].

> **This comparison does not establish that the models here are competitive.**
> The published rows are real FD001. Our rows are a synthetic generator emitting
> smooth monotonic drift with no fault modes, no operating-condition switching and
> no sensor pathology — a strictly easier regression problem. A favourable number
> here is the *expected* consequence of an easier test set, not evidence of a
> better model. This report exists to position the work and to mark exactly which
> comparisons will and will not become valid once `data/CMAPSSData/` is populated.

## 1. Published baselines

FD001, scored at each test engine's last cycle, RUL capped at 125–130 depending on the
paper. **Every figure is recalled from memory and none was fetched** — read §6 before
quoting any of them.

| source | year | family | RMSE ↓ | PHM ↓ | PHM/engine ↓ | params | confidence |
|---|---|---|---|---|---|---|---|
| Saxena et al. (C-MAPSS + score definition) [1] | 2008 | reference | — | — | — | not reported | high |
| SVR [3] | 2016 | classical | 20.96 | 1381.5 | 13.81 | not reported | medium |
| Random Forest [3] | 2016 | classical | 17.91 | 479.8 | 4.80 | not reported | medium |
| Deep CNN (Babu et al.) [2] | 2016 | early-deep | 18.45 | 1286.7 | 12.87 | not reported | high |
| MODBNE (deep belief net ensemble) [3] | 2016 | early-deep | 15.04 | 334.2 | 3.34 | not reported | high |
| Deep LSTM (Zheng et al.) [4] | 2017 | early-deep | 16.14 | 338.0 | 3.38 | not reported | high |
| DCNN, time-window input (Li et al.) [5] | 2018 | modern-deep | 12.61 | 273.7 | 2.74 | not reported | high |
| DAG network (CNN + LSTM) [6] | 2019 | modern-deep | 11.96 | 229.0 | 2.29 | not reported | medium |
| AGCNN (feature attention) [7] | 2020 | modern-deep | 12.42 | 225.5 | 2.25 | not reported | medium |

The `PHM/engine` column is the one to read. The PHM score is a **sum** over test
engines (`src/evaluate.py`), and real FD001 has 100 test engines
against this project's 50. Comparing raw sums across differently
sized test sets is a unit error, not a result.

## 2. Side by side

| model | RMSE ↓ | RMSE 95% CI | PHM ↓ | PHM/engine ↓ | test engines |
|---|---|---|---|---|---|
| SVR [3] | 20.96 | — | 1381.5 | 13.81 | 100 |
| Random Forest [3] | 17.91 | — | 479.8 | 4.80 | 100 |
| Deep CNN (Babu et al.) [2] | 18.45 | — | 1286.7 | 12.87 | 100 |
| MODBNE (deep belief net ensemble) [3] | 15.04 | — | 334.2 | 3.34 | 100 |
| Deep LSTM (Zheng et al.) [4] | 16.14 | — | 338.0 | 3.38 | 100 |
| DCNN, time-window input (Li et al.) [5] | 12.61 | — | 273.7 | 2.74 | 100 |
| DAG network (CNN + LSTM) [6] | 11.96 | — | 229.0 | 2.29 | 100 |
| AGCNN (feature attention) [7] | 12.42 | — | 225.5 | 2.25 | 100 |
| **this project — ATTENTION** (seed-avg) | 8.21 | [0.10, 16.33] | 54.6 | 1.09 | 50 |
| **this project — GRU** (seed-avg) | 6.82 | [3.15, 10.49] | 43.6 | 0.87 | 50 |

The interval is a two-sided 95% Student-t interval on the mean over 3 seeds (t=4.303), computed by `src/ci.py` — the same machinery `src/compare.py` uses, so this is the project's own headline number rather than a second opinion about it. It is wide because a handful of runs cannot pin a mean tightly; narrowing it is a matter of compute, not of method.

## 3. Deltas

Lead model: **ATTENTION**, seed-averaged over 3 seed(s).

| baseline | their RMSE | RMSE gap | PHM gap (raw) | PHM gap (per engine) | still beaten at our CI upper bound? |
|---|---|---|---|---|---|
| SVR [3] | 20.96 | +60.8% | +1326.9 | +92.1% | yes |
| Random Forest [3] | 17.91 | +54.2% | +425.1 | +77.2% | yes |
| Deep CNN (Babu et al.) [2] | 18.45 | +55.5% | +1232.1 | +91.5% | yes |
| MODBNE (deep belief net ensemble) [3] | 15.04 | +45.4% | +279.6 | +67.3% | **no** |
| Deep LSTM (Zheng et al.) [4] | 16.14 | +49.1% | +283.4 | +67.7% | **no** |
| DCNN, time-window input (Li et al.) [5] | 12.61 | +34.9% | +219.1 | +60.1% | **no** |
| DAG network (CNN + LSTM) [6] | 11.96 | +31.4% | +174.4 | +52.3% | **no** |
| AGCNN (feature attention) [7] | 12.42 | +33.9% | +170.9 | +51.5% | **no** |

`RMSE gap` is positive when this project's mean is lower. The last column is the
stricter test — does the baseline still lose to the **upper** end of our confidence
interval? — and it is the column that would flip first on real data.

### Parameter count

| model | params |
|---|---|
| this project — ATTENTION | 193,986 |
| this project — GRU | 177,345 |
| every published row above | **not reported** |

The brief asks for a parameter-count delta and it cannot be computed. RUL papers on
C-MAPSS report RMSE and score and almost never report model size, so there is no
published denominator. Worth stating plainly rather than estimating: an invented
baseline count would make the number here look either efficient or bloated purely by
choice of fiction.

## 4. Interpretation

**On the means, RMSE is ahead of everything published** — 8.21 against
a best published 11.96 (DAG network (CNN + LSTM) [6]), a +31.4% gap. It should not be believed, for two
independent reasons, and they fail in different directions.

**First, the interval swallows most of the table.** The 95% CI on this project's mean is [0.10, 16.33] over 3 seeds — wide enough that **5 of the 8 published baselines are not beaten at its upper end**, including every result from 2017 onward. The point estimate says 8.21; the evidence says somewhere between 0.10 and 16.33, and a good part of that span is ordinary territory for a 2017 paper. This is a compute problem, not a data problem, and it is fixable today: more seeds tighten the interval as ~1/√n.

Worth noting which model this bites. The brief names the attention model as the project's best, and it is the one whose interval fails — while GRU [3.15, 10.49] clears every published row in the table at its upper bound. That is the same pattern `src/variance.py` already found: attention owns the best single run and the worst stability, and picking a headline model on a single seed picks the wrong one.

**Second, and this one would survive any number of seeds: the test set is not the same
problem.** The generator's degradation is monotonic and low-noise, so late-life RUL is
nearly a deterministic function of the sensor trajectory. That is not true of FD001's
HPC-degradation fault modes, and no amount of compute on this data will tell us how
much of the margin survives contact with them.

**On PHM the raw numbers also look ahead (54.6 vs 229.0), and
most of that gap is a unit error.** Normalise for the engine count and the comparison
becomes 1.09 against 2.29 per engine —
still favourable, and still for the same reason. PHM's exponential late-prediction
penalty is dominated by the worst few engines, and the synthetic test set has no
genuinely hard engines to produce them. `outputs/error_analysis.md` already shows 57%
of this project's PHM concentrated in six engines on data with *no* fault modes; on
FD001 that tail is the thing the score is built to measure.

**So the honest summary is not 'beat on RMSE, lag on PHM'.** It is: the pipeline
produces numbers in a plausible range, under a protocol matching the literature's, on a
test set easier than the literature's by an unknown factor, against baselines recalled
rather than verified. Three of those four clauses have to be fixed before there is a
comparison claim at all — and the one that is already sound (the protocol) is the one
that took the most work.

## 5. What would change on real FD001

Written down now so it can be scored later rather than rationalised afterwards.

- **RMSE rises to roughly 12–16.** FD001 carries irreducible noise this generator does
  not: the true RUL of an engine at a fixed sensor state is not a point. Landing under
  12 would put this project at the 2019–2020 state of the art, which a ~200k-parameter
  recurrent model trained on a laptop over a few seeds should not be expected to do.
- **PHM rises superlinearly, to roughly 250–450 over 100 engines.** A factor of 2 comes
  from the engine count alone, and the exponential penalty means the extra hard engines
  cost disproportionately more than the easy ones. This is where the gap to published
  work will actually show, because the score punishes exactly the tail behaviour that
  synthetic data lacks.
- **The neural margin over the forest should widen — and if it does not, that is a
  finding.** On smooth monotonic drift the sequence models have little temporal
  structure to exploit that the rolling features do not already capture, which is the
  most likely reason the RandomForest stays within ~3 RMSE of them here. Real fault
  modes are where a sequence model earns its parameters.
- **The attention model's instability should get worse, not better.** Its across-seed
  RMSE range is already the widest in the project on easy data. Harder data with a
  longer tail gives the seed more to disagree about, so the gap between its best and
  typical run should grow.

## 6. Data sources and caveats

**Sources.** 8 comparable baselines drawn from 7 papers, all cited in the header comment of `src/compare_published.py`. All were written
from memory under the brief's no-scraping constraint. Each row carries a `confidence`
field; `medium` means the method and magnitude are right but the digits need re-reading
from the paper. **No row is marked verified, because none is.**

**Where the brief's supplied numbers and these disagree.**

| brief's estimate | RMSE | PHM | assessment |
|---|---|---|---|
| Saxena et al. (original 2008) | ~18.5 | ~200 | **no such row exists.** [1] defines the dataset and the score; it reports no FD001 last-cycle RMSE. ~18.5 matches Babu et al. 2016 (18.45) closely enough that the figure is probably that one, misattributed |
| Recent deep learning (e.g. LSTM) | ~8–12 | ~50–80 | RMSE optimistic — best published FD001 is ~11.9–12.6, so 8 is below anything I can cite. **PHM off by roughly 4x**: real values are ~225–340 |
| Random Forest / XGBoost typical | ~10–14 | ~60–100 | optimistic — the published RF row is 17.91 / 479.75 [3]. A tree ensemble at 10–14 RMSE on FD001 would be near state of the art |

The PHM divergence is the important one, and it is the same unit error the rest of this
report is built to avoid: 50–100 is the range of a PHM *sum over ~50 engines*, or of a
per-engine mean, but not of a published FD001 score. Had the brief's figures been
hardcoded as given, this report would have concluded "competitive on RMSE, roughly
level on PHM" — and been wrong on both counts.

**Remaining caveats.**

- Our numbers are 3 seed(s). The intervals are wide; treat any ordering inside them
  as unresolved rather than close.
- Published papers differ on RUL cap (125 vs 130) and input window length. Both move
  RMSE by a few tenths — negligible against the gaps here, not negligible against the
  intervals.
- Papers report their best run and rarely say over how many. A seed-averaged number is
  being compared against what is probably a favourable single draw, which biases every
  comparison in this report *against* this project. Of the biases present here, that is
  the one direction that is safe.
- Nothing in this file is checked by `src/validate_docs.py` against a source of truth,
  because the published rows have no artifact to be checked against — they are inputs,
  not outputs. The derived cells are recomputed from `outputs/comparison.json` on every
  run.
