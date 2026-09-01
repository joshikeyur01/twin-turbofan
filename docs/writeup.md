# Building a digital twin for jet-engine RUL prediction

A technical account of `twin-turbofan`: what the problem actually is, the decisions that
mattered, what worked, what didn't, and what I got wrong along the way.

**Standing caveat, stated once and meant throughout:** the real NASA C-MAPSS dataset was
not available for this run, so every number here comes from a synthetic fallback that
emits smooth monotonic drift with no real fault modes. The engineering is real and the
protocol is real; the *numbers* validate plumbing. Where a conclusion would likely change
on real data, I say so.

---

## 1. The problem

Given multivariate sensor telemetry from a fleet of turbofan engines, predict each
engine's **Remaining Useful Life** — the number of operating cycles before it needs
service. The C-MAPSS benchmark gives 21 sensors and 3 operational settings per cycle, with
training engines run to failure and test engines truncated partway through life.

Two things make this more than a generic regression task.

### The target is a modelling decision, not a given

RUL is not in the data; you construct it. For a training engine that failed at cycle
*N*, the RUL at cycle *t* is *N − t*. But that implies the model should distinguish "300
cycles left" from "290 cycles left" — which is not learnable, because a healthy engine
shows no signal about exactly how much life remains. The standard fix is a
**piecewise-linear cap**: clip RUL at some ceiling (125 here), so early life is a flat
plateau and only the final descent is a real countdown.

That cap has a consequence which shows up later in the live twin: the model reports
continuous degradation while the *target* sits flat, so the twin looks pessimistic
through mid-life. That is an artefact of the label, not an error in the model — a
distinction worth being able to make when reading a divergence plot.

### The error metric is asymmetric, and it matters

Predicting failure **late** means the part failed in service. Predicting **early** means
you serviced it sooner than strictly necessary. These are not equally bad, and RMSE
treats them identically. The PHM'08 challenge score doesn't:

```
penalty(d) = exp(-d/13) - 1   if d < 0   (early)
             exp( d/10) - 1   if d ≥ 0   (late)
```

where `d = predicted − true`. Both metrics are reported everywhere in this project,
because they genuinely disagree. Switching from a ridge fallback to a RandomForest
improved RMSE (11.07 → 10.31) while making the PHM score **worse** (79.2 → 85.0). A
project reporting only RMSE would have called that a clean win.

---

## 2. Protocol decisions

These are the choices that determine whether the numbers mean anything.

**Split by engine, never by cycle.** Consecutive cycles from one engine are
near-duplicates. A random row-level split puts cycle 100 in train and cycle 101 in
validation, which is close to putting the answer in both. Every split here partitions
*unit ids*.

**Score on each engine's last recorded cycle.** That is the operational decision point,
and it is what the benchmark's `RUL_*.txt` files define. Applying it identically to the
tree model and the sequence models is what makes them comparable at all.

**Validate sequence models over all windows, not last cycles.** This one is a trap.
Training engines run to failure, so a *held-out training* engine's final cycle always has
RUL = 0. Early-stopping on that single degenerate point would measure almost nothing and
mirror nothing about the test set, whose engines are truncated mid-life. So validation
uses RMSE across every window of the held-out engines. A closer mimic would randomly
truncate each validation engine to imitate the test distribution; that is noted in the
code as a refinement, not done.

**Select hyperparameters on validation, never test.** The sweep ranks by validation RMSE
and prints the best *test* number only as a sanity check. Picking the best test row is
selecting on the evaluation set, and it silently inflates every headline figure.

---

## 3. What the error analysis found

Aggregate metrics say how much error there is. They don't say where it lives, and that
turned out to be the more useful question.

The RandomForest's mean residual is **+0.09 cycles** — essentially unbiased. But the PHM
score is concentrated in a tail:

| | share of total PHM |
|---|---|
| worst single engine (true 72 → predicted 104) | 27.8% |
| worst 6 engines | 57% |
| all late predictions (44% of engines) | 71% |

Because the penalty is exponential, a handful of large late misses dominate the aggregate
while RMSE — averaging squared error over 50 engines — barely registers them. That is the
mechanism behind the ridge-vs-forest metric disagreement above.

Broken down by life stage, the picture is operationally reassuring:

| true RUL bin | mean residual |
|---|---|
| 0–25 (near failure) | **−1.53** (early — safe) |
| 25–50 | +3.48 |
| 50–75 | **+9.17** (worst) |
| 75–100 | −2.60 |
| 100–125 (cap plateau) | −3.37 |

The worst errors are mid-life. Near failure — where the maintenance decision is actually
made — the twin is accurate and slightly early, which is the safe direction.

---

## 4. What worked

### Sequence models, once given enough capacity

My first LSTM run used the scaffold's defaults (`seq_len=30, hidden=64`) and scored RMSE
11.04 / PHM 92.6 — **worse than the RandomForest**. I initially took that at face value.
It was wrong: a 27-point sweep showed the defaults were simply under-capacity.

Each architecture was then swept on its own grid and the comparison averaged over 3 seeds.
The seed-averaging is not a detail — see §5, where it removes two conclusions I had already
written down.

| model | config | RMSE (3 seeds) | PHM | across-seed range | params |
|---|---|---|---|---|---|
| **GRU** | 50/128/1e-3 | **6.820** ±1.336 | **43.6** ±19.5 | 2.672 | 177,345 |
| ATTENTION | 50/128/3e-4 | 8.210 ±2.866 | 54.6 ±34.8 | 5.733 | 193,986 |
| LSTM | 50/128/3e-4 | 8.302 ±0.861 | 53.7 ±8.5 | 1.722 | 235,073 |
| RandomForest | — | 10.191 ±0.095 | 81.4 ±3.0 | **0.190** | — |
| CNN | 20/64/3e-4 | 11.386 ±1.045 | 97.3 ±19.7 | 2.090 | 10,401 |

**The GRU is best on both metrics** — 33% better RMSE and 46% better PHM than the forest. It
reached that only after the *configuration selection* was itself corrected (§5): its single-seed
sweep picked `lr=3e-4`, the seed-averaged ranking prefers `lr=1e-3`, and that is worth 1.49 RMSE.
Before that fix this table had the GRU tied with the LSTM at 8.3.

**Read the margin against the range, though.** The GRU leads the LSTM by 1.482 RMSE while the two
vary by 2.672 and 1.722 across seeds. Best available, not decisively separated.

**The CNN wanted the opposite corner of the grid** — a short window and few channels, where the
recurrent models want 50/128. Run at the LSTM's configuration it scored 14.285 / 135.8; on its own
it reaches 11.386 / 97.3. The "untuned, not worse" caveat on that early row was correct. It is
still worse than the forest, but a **10,401-parameter** model landing within ~12% of it is a real
efficiency observation — 17× smaller than the GRU.

**The most useful column is the range.** The forest varies by 0.190 RMSE across seeds; every
neural model varies by 1.7 to 5.7, and attention by 6.1 at its own best configuration. For
something retrained periodically, that reliability gap matters more than the mean — and it is
invisible in any single-seed table.

### An online feature path that actually streams

The inherited streaming demo computed its features by calling the batch feature builder
over the *entire test set* up front, then replaying pre-computed rows. The printed numbers
were right, and it looked like a live twin. It wasn't one: it depended on data from the
future, and would have broken on the first real telemetry message.

The fix is a persisted feature contract. At training time the pipeline writes a
`FeatureSpec` — selected sensors, column order, rolling window, and the **training**
mean/std. At serving time `OnlineFeatureBuilder` keeps a per-engine ring buffer and emits
one standardised vector per reading, using those frozen statistics.

Reusing training statistics rather than recomputing from the stream is the whole point.
Recomputing is the classic train/serve skew bug, and it is nasty precisely because it
throws no error: the model just receives inputs on a different scale than it was fit on
and quietly degrades. A test asserts the streaming path reproduces the batch path to
1e-9, and another asserts that interleaved engines — which is what an MQTT bus actually
delivers — keep isolated state.

### Mutation-testing the test suite

Every test in this project passed the first time it was written, which is not evidence of
anything. So I broke the source deliberately, 13 ways, and checked each break failed a
test: dropping the RUL cap, swapping the PHM early/late constants, removing the rolling
NaN fill, standardising the test split on its own statistics, letting rolling windows span
engine boundaries, splitting by row instead of engine, taking the first cycle instead of
the last, padding right instead of left.

Twelve were caught. **One survived**, and it was the interesting one: replacing the CNN's
global time-pooling with "read only the first convolution output" is shape-identical, so
every shape assertion still passed — a model that ignored 98% of its input window would
have looked fine. The gap is now covered by a test asserting the prediction responds to a
mid-window perturbation.

---

## 5. What didn't work

### Conservative prediction, my main wrong call

The tail finding in §3 suggested an obvious lever: since PHM punishes lateness
exponentially and 71% of the score comes from late predictions, shifting every prediction
earlier should trade a little RMSE for a large PHM gain. I recorded that in the backlog as
the highest-value remaining work on the metric.

Split-conformal calibration on 30 held-out engines says the effect is small:

| residual quantile | offset | RMSE | PHM |
|---|---|---|---|
| 0.50 | −0.55 | 10.540 | 84.4 |
| 0.60 | −0.02 | 10.518 | 82.1 |
| **0.70** | **+1.86** | 10.654 | **77.5** |
| 0.80 | +6.78 | 12.428 | 87.2 |
| 0.90 | +16.11 | 19.041 | 182.4 |

The curve is U-shaped with a shallow minimum: the best shift is **+1.86 cycles** and
recovers **5.6% of PHM for a 1.3% RMSE cost**. Favourable and in the predicted direction,
but nothing like what "71% of the score is late predictions" implied.

The reasoning error is worth naming, because it generalises: **tail concentration does not
imply a uniform offset helps.** The PHM asymmetry (`exp(d/10)` vs `exp(-d/13)`) is only
mildly lopsided. Shifting all 50 engines earlier pays a small early penalty on every one
of them to shave a few large late errors, and the many small costs nearly cancel the few
large gains. Attacking that tail requires **per-engine** conservatism scaled to each
engine's own predictive uncertainty — a wide interval on the uncertain engines, not a
constant subtracted from everybody.

I also had the sign convention inverted on the first implementation. With
`adjusted = point − quantile(residuals, q)`, *high* q is the conservative direction; my
initial sweep ran 0.05–0.50 and therefore explored mostly the **unsafe** half of the
curve, producing a table that looked like conservatism catastrophically hurt. The
convention is now stated explicitly in the module, the report, and the sweep range. The
report's verdict sentence is generated from the measured gain rather than written by hand,
so it cannot drift into claiming a win the numbers don't support.

### Single-seed reporting, which quietly biased everything

This is the one I would most want a reviewer to see, because it invalidated conclusions I had
already written up as findings.

Every sequence-model number in this project came from `seed=42`, fixed early and reused. A
variance study (`src/variance.py`) re-ran fixed configurations across seeds:

| arch | seed 42 | 3-seed mean | across-seed range |
|---|---|---|---|
| attention | 6.248 | 8.210 | 5.733 |
| GRU | 7.292 | 8.312 | 1.767 |
| LSTM | 7.804 | 8.302 | 1.723 |
| CNN | 10.236 | 11.359 | 2.235 |
| RandomForest | 10.309 | 10.191 | **0.190** |

**Seed 42 flattered every neural model by 0.5–2.0 RMSE, and left the forest alone.** Two
conclusions did not survive:

1. *"The GRU beats the LSTM, so the LSTM's extra gate earns nothing."* Their means differ by
   0.010 against ranges of ~1.7. They are tied.
2. *"The CNN matches the forest on RMSE and beats it 17% on PHM — an efficiency win."* Averaged,
   it is 11% worse on RMSE and 8% worse on PHM.

Same-seed repeats, by contrast, are bit-identical, so this is seed *sensitivity*, not
nondeterminism — the seed also drives the by-engine train/val split, which is the most likely
source of the swing at 100 training engines.

The lesson is not subtle and I had to learn it three times in one session: a difference smaller
than the spread of your own re-runs is not a result. `compare.py` now averages over seeds by
default, and single-seed values appear only as detail rows.

### Rolling features earn less than expected

| arm | features | RMSE | PHM |
|---|---|---|---|
| raw sensors only | 21 | 10.574 | 84.4 |
| + rolling w=3 | 63 | **10.249** | **79.0** |
| + rolling w=5 (default) | 63 | 10.309 | 85.0 |
| + rolling w=20 | 63 | 11.489 | 128.3 |

Tripling the feature count buys ~3% RMSE. On data whose degradation is smooth monotonic
drift that is expected — there is little noise for a window to average away — and I expect
a larger gap on real C-MAPSS. Two other notes: the shipped default `w=5` is *not* optimal
(`w=3` beats it on both metrics), and I deliberately did not change it, because tuning a
hyperparameter on synthetic data fits an artefact of the generator rather than the
problem. Long windows are actively harmful: a 20-cycle mean lags a monotonic trend and
systematically understates current degradation.

---

## 6. Things found by looking rather than assuming

**The environment was emulated.** The available Python was an x86_64 build running under
Rosetta on arm64 hardware. That silently caps torch at 2.2.2 and removes MPS acceleration
entirely — every sequence model would have trained on emulated CPU. Switching to a native
arm64 interpreter got torch 2.13 with MPS. Worth checking `sysconfig.get_platform()`
against `uname -m` before concluding a machine is slow.

**Sensor selection was a no-op on synthetic data — until it wasn't.**
`select_informative_sensors` dropped 0 of 21 sensors on the fallback, because the
generator gave its "constant" sensors σ=0.5 noise — variance ≈0.25, about 250× the
threshold. Real FD001 has six genuinely flat sensors. So that code path was exercised by
unit tests (whose fixtures use truly constant sensors) but never by a pipeline run.
Generator v2 emits `s1, s5, s10, s16, s18, s19` as exact constants; the fallback now
drops 6 of 21 and builds 45 feature columns instead of 63. It cost the recorded baseline —
RMSE 10.309 → 12.513 — though only ~9% of that move is the fix itself and the rest is the
different random draw that removing the noise produces. Full accounting in
[`../outputs/synthetic_fidelity.md`](../outputs/synthetic_fidelity.md). **The tables in §4
and §6 below predate this and were not re-run**; they are v1 numbers.

**Reproducibility held across a major version jump.** The suite passes unchanged on
numpy 1.21/pandas 1.4/sklearn 1.0 and on numpy 2.4/pandas 3.0/sklearn 1.9, and the
baseline reproduces byte-identical metrics on both. That is what `random_state=42` is
supposed to buy and rarely gets verified.

**mypy found a design flaw, not a nit.** `train()` tracked its best checkpoint in one
dict — `{"val_rmse": float, "epoch": int, "state": dict | None}`. mypy inferred
`dict[str, float | None]` and flagged five downstream errors: comparing a float to `None`,
subtracting from `None`, passing a float to `load_state_dict`. All five were symptoms of
one cause — bundling three unrelated types in a dict defeats type checking at every
access. Three separate locals fixed all five.

---

## 7. Honest limitations

- **No real data.** This is the limitation that qualifies everything else. Model ranking,
  the ablation's window choice, and the conformal offsets could all change on real
  C-MAPSS.
- Only FD001 exists synthetically, so the model×dataset table is one column of four and
  cross-condition generalisation (train FD001 → test FD002/FD004) is untested. That
  experiment is arguably the most interesting one in the backlog, since it probes whether
  the twin transfers across operating conditions at all.
- **Three seeds do not cleanly separate these models.** The GRU leads the LSTM by 1.482
  RMSE against across-seed ranges of 2.672 and 1.722. The ordering is the best available
  but is not decisively established; more seeds would firm it up, at roughly linear cost.
- The winning sweep cell (`seq_len=50, hidden=128`) sits on the grid boundary, so capacity
  may be unsaturated. Extending the grid is **not** obviously worthwhile though: the
  across-seed spread (1.7–6.1) already exceeds the gaps a wider single-seed grid would be
  ranking, so a bigger grid would mostly generate differences below the noise floor.
- Sweeps themselves are ranked on a single seed. `src/rerank.py` re-checks only the
  *finalists* across seeds, which caught two bad selections — but a fully seed-averaged
  sweep (seeds × 27 configs × 4 architectures) was out of budget.
- Conformal intervals over-cover (80→84%, 90→92%, 95→98%). Expected direction: calibration
  sees every cycle of its engines while the test set observes one truncated point each, so
  the calibration residual spread is wider than the test-time spread.
- Standardisation statistics come from the whole training file, including engines later
  held out for validation. A mild optimism in the validation number, kept deliberately so
  the tree and sequence models receive identical inputs; test metrics are unaffected.
- No comparison to published C-MAPSS results, because comparing synthetic numbers to
  real-data benchmarks would be meaningless.
- ~~`select_informative_sensors` drops 0 of 21 sensors on synthetic data.~~ Fixed in
  generator v2, which drops 6 of 21 as real FD001 does. It did invalidate the recorded
  synthetic metrics: the baseline was re-run, but the comparison, ablation, sweep,
  variance, uncertainty and ensemble artifacts were not, so those tables remain v1.

---

## 8. What I would do next, in order

1. **Get the real data in** and re-run everything. Nothing else on this list is worth
   doing first, and several conclusions above are provisional until it happens.
2. **Cross-condition generalisation** (FD001 → FD002/FD004). The most informative
   experiment available, and the one that tests whether this is a twin or a curve-fit.
3. **Per-engine uncertainty** instead of a global conformal offset — the actual fix for
   the tail, per §5.
4. Sweep the CNN and GRU on their own terms; extend the LSTM grid past its boundary.
5. Ensemble the RandomForest with the best sequence model and report whether it helps —
   including if it doesn't.
