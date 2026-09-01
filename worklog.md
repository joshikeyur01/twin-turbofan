# twin-turbofan — Worklog

Timestamped log of actions, decisions, and measured results. Numbers pasted here are
the actual values produced by the run, not targets.

---

## 2026-07-30 — Session start

### Repo isolation (before any code work)

`twin-turbofan` was **not its own git repository**. `git rev-parse --show-toplevel`
resolved to `/Users/keyur` — the project sat inside a repo rooted at the home
directory, with zero commits on `main`.

This mattered because `CONTINUE.md` instructs the agent to branch and commit after
every task, and `run_continuous.sh` runs with `--dangerously-skip-permissions`. A
`git add`-and-commit loop at that root would have swept in home-directory dotfiles —
agent CLI auth data, `.config/`, `.docker/`, `.cisco/`, `.vpn/`. The project's
own `.gitignore` is written for a root at `twin-turbofan`, so none of its patterns
would have applied.

**Decision:** ran `git init` inside `twin-turbofan` so the repo is self-contained.
Committed the Week-1 scaffold as `33c62fa`, then branched `continuous-build`.
Repo-local git identity set to `Keyur Joshi <keyurjoshi2104@gmail.com>`.

The stray repo at `/Users/keyur` is left untouched — cleaning it up is the user's call.

### Environment

Active interpreter is Anaconda Python **3.9.13** at `~/opt/anaconda3/bin/python3`.
Installed versions sit *below* the floors in `requirements.txt`:

| package | required | installed | status |
|---|---|---|---|
| numpy | >=1.24 | 1.21.5 | below floor, works |
| pandas | >=2.0 | 1.4.4 | below floor, works |
| matplotlib | >=3.7 | 3.5.2 | below floor, works |
| scikit-learn | >=1.3 | 1.0.2 | below floor, works |
| pytest | — | 7.1.2 | available |
| torch | optional | missing | blocks Tier 2 |
| streamlit, paho-mqtt | optional | missing | blocks Tier 3 |

**Decision:** did *not* run `pip install -r requirements.txt`. Upgrading numpy/pandas
in the base conda environment is a system-wide change that could break the user's
other projects, and every core dependency is already present and functional. The
whole pipeline runs green on these versions, so the floors in `requirements.txt`
appear stricter than the code actually needs. Logged as a TODO to either relax the
floors or pin a tested environment — rather than silently mutating base conda.

### Data availability — BLOCKED

`data/CMAPSSData/` does not exist. Only `data/synthetic/` (FD001-shaped, from
`src/generate_synthetic.py`) is present. Per `CONTINUE.md` this is logged as blocked
and the run proceeds on FD001 synthetic.

**Consequence to be honest about:** `generate_synthetic.py`'s own docstring says its
degradation is monotonic drift plus noise and "results on it are meaningless as a
benchmark." So every metric below validates plumbing only. Tier 2's FD001–FD004
comparison and Tier 6's comparison against published C-MAPSS results cannot be done
at all without the real dataset — synthetic has only FD001. Downloading the NASA data
needs the user's go-ahead, so it is surfaced rather than assumed.

### Tier 1.1 — RandomForest baseline active

`outputs/metrics.json` previously read `RidgeFallback (numpy)`, meaning whichever
interpreter last ran the baseline had no scikit-learn. The active env does, so
`make_model()` now selects RandomForest as intended. Re-ran `python -m src.train_baseline`
(38.6s wall, 100 train / 50 test engines):

| model | RMSE | PHM score |
|---|---|---|
| RidgeFallback (previous, numpy) | 11.067 | 79.2 |
| RandomForestRegressor (now) | **10.309** | **85.0** |

**Observation worth keeping:** RF improves RMSE (11.07 → 10.31) but its PHM score gets
*worse* (79.2 → 85.0). The two metrics disagree because PHM is asymmetric — it
punishes late predictions (over-estimated RUL) far harder than early ones. RF is
landing slightly later than ridge on average, which is the more dangerous direction
for maintenance: a late RUL estimate means the part fails in service. This is exactly
why `CONTINUE.md` requires reporting both numbers, and it is the first thing the
error analysis in Tier 1.2 should confirm or refute.

Baseline to beat (synthetic FD001): **RMSE 10.309 / PHM 85.0**.

### Tier 1.2 — Error analysis (`src/error_analysis.py`)

New module writing `outputs/trajectories.png`, `outputs/residuals.png`, and a
`outputs/error_analysis.md` summary. Sign convention follows `phm_score`:
`residual = predicted − true`, so **positive = predicted late**.

Measured on the 50 last-cycle scoring rows:

| statistic | value |
|---|---|
| mean residual | **+0.09** cycles |
| median residual | −1.04 cycles |
| range | −21.4 … +32.0 |
| predicted late | 22/50 engines (44%) |

Bias by life stage (true-RUL bin → mean residual):

| true RUL bin | n | mean resid | std |
|---|---|---|---|
| 0–25 (near failure) | 5 | **−1.53** | 3.17 |
| 25–50 | 12 | +3.48 | 8.97 |
| 50–75 | 6 | **+9.17** | 14.97 |
| 75–100 | 8 | −2.60 | 14.29 |
| 100–125 (cap plateau) | 19 | −3.37 | 7.10 |

**This corrects the hypothesis from Tier 1.1.** I had read RF's worse PHM score as a
systematic late bias. It isn't — the mean residual is essentially zero (+0.09) and the
median is slightly *early*. The real story is the tail:

| | share of total PHM |
|---|---|
| worst single engine (true 72 → pred 104, +32) | **27.8%** |
| worst 6 engines | **57%** |
| all late engines (44% of the fleet) | **71%** |

Because the PHM penalty is exponential, a handful of large late misses dominate the
aggregate while RMSE — which averages squared error — barely registers them. That is
why the two metrics moved in opposite directions between ridge and RF.

Two consequences worth carrying forward:

1. **Where the error lives is operationally favourable.** The worst misses are all
   mid-life engines (true RUL 37–101). Near failure (0–25) the twin is both accurate
   and slightly *early* — the safe direction, and the regime where the maintenance
   decision actually gets made.
2. **The lever for PHM is worst-case lateness, not average error.** Chasing RMSE will
   not fix this; predicting a lower quantile of the RUL distribution would. That makes
   Tier 6's uncertainty item (quantile/conformal prediction) the highest-value work in
   the backlog for this metric, not a stretch goal — noting it now, keeping the
   backlog order.

The trajectory plots also confirm the twin tracks degradation continuously rather than
just landing the endpoint: the long-lived engine (222 cycles) follows the true
piecewise-linear curve closely through the whole descent, and short truncated engines
sit flat on the cap plateau as they should.

### Tier 1.3 — pytest suite

49 tests across three files, all green in 0.66s:

| file | tests | covers |
|---|---|---|
| `tests/test_data_loader.py` | 20 | RUL labelling for train (countdown) and test (RUL-file anchor + back-fill), cap clipping, `last_cycle_rows`, schema, missing-data error |
| `tests/test_evaluate.py` | 15 | `phm_score` asymmetry, monotonicity, exact PHM'08 formula, RMSE symmetry |
| `tests/test_features.py` | 14 | no NaNs, expected columns, engine-boundary and train-stats leakage guards |

Fixtures build miniature C-MAPSS files in `tmp_path` and monkeypatch `DATA_DIR`, so
tests never depend on whether the real dataset is present, and run in under a second.

**Decision:** a suite that passes on first write is not yet evidence of anything, so I
mutation-tested it — broke the source six ways and confirmed each break is caught:

| mutation | result |
|---|---|
| drop the RUL cap | 2 failed |
| swap PHM early/late constants (13↔10) | 6 failed |
| remove rolling-std NaN fill | 4 failed |
| standardise test split on its own stats (leakage) | 1 failed |
| let rolling windows span engine boundaries | 3 failed |
| `last_cycle_rows` picks the first cycle | 2 failed |

All six caught; source restored via `git checkout` after each. The asymmetry test is
parametrised over five error magnitudes, which is what makes the 13↔10 swap fail
loudly rather than subtly.

Added `pytest.ini` (testpaths, `--strict-markers`, DeprecationWarning-as-error for
`src.*`) and a root `conftest.py` so `import src` resolves without PYTHONPATH setup.
New dependencies logged: **pytest>=7.0** (test runner) and **tabulate>=0.8** (backs
`DataFrame.to_markdown()` in the error-analysis report); both added to
`requirements.txt` and both already present in the env.

---

## Environment rebuild — native arm64 venv

Investigating the torch install surfaced a bigger problem: the Anaconda interpreter is
an **x86_64 build running under Rosetta** on arm64 hardware.

```
uname -m            -> arm64
sysconfig.get_platform() -> macosx-10.9-x86_64
```

That combination caps torch at 2.2.2 (PyTorch stopped shipping macOS x86 wheels after
that) and rules out MPS acceleration entirely, so every sequence model in Tier 2 would
have trained on emulated CPU.

Found a native interpreter already on the system — `/Library/Frameworks/Python.framework/Versions/3.11`,
Python 3.11.1, `arm64`. Built `.venv` from it instead of installing torch into conda.

| package | conda (x86_64, py3.9) | venv (arm64, py3.11) |
|---|---|---|
| numpy | 1.21.5 | **2.4.6** |
| pandas | 1.4.4 | **3.0.5** |
| matplotlib | 3.5.2 | **3.11.1** |
| scikit-learn | 1.0.2 | **1.9.0** |
| torch | — | **2.13.0** |
| MPS available | no | **yes** |

Two things fall out of this:

1. **The `requirements.txt` floors are now satisfied and validated**, so that TODO is
   closed. The full 56-test suite passes on numpy 2.x / pandas 3.x / sklearn 1.9 with
   no source changes — the code turned out to be forward-compatible across the pandas
   1.4 → 3.0 major break, which I did not assume and did check.
2. **Results are reproducible across both environments.** The baseline gives byte-identical
   metrics (RMSE 10.309 / PHM 85.0) on sklearn 1.0.2 and 1.9.0, which is what
   `random_state=42` is supposed to buy and rarely gets verified.

`.venv` is 1.0G and gitignored. All subsequent numbers come from the venv
(`.venv/bin/python -m src.*`) so the environment is held constant.

### Tier 2 — Feature ablation (`src/ablation.py`)

Added a `use_rolling` flag to `build_xy` so the raw-sensor arm is a first-class code
path rather than a hand-edit, then swept the rolling window. Same protocol throughout:
split by engine, score on the last cycle, report both metrics.

| arm | features | RMSE | PHM | % late |
|---|---|---|---|---|
| raw sensors only | 21 | 10.574 | 84.4 | 44.0 |
| + rolling w=3 | 63 | **10.249** | **79.0** | 46.0 |
| + rolling w=5 *(current default)* | 63 | 10.309 | 85.0 | 44.0 |
| + rolling w=10 | 63 | 10.555 | 100.3 | 52.0 |
| + rolling w=20 | 63 | 11.489 | 128.3 | 46.0 |

**Findings, with the caveat that this is synthetic data:**

- Rolling features buy very little here — 3% RMSE for 3× the feature count. On data
  whose degradation is smooth monotonic drift, a snapshot is already nearly sufficient;
  there is no noise for the window to average away. I expect a larger gap on real
  C-MAPSS, and this is the first thing to re-run when the data lands.
- **The default `window=5` is not optimal — `w=3` beats it on both metrics.** Not
  changing the default yet: picking a hyperparameter on synthetic data would be
  fitting to an artefact of the generator.
- Long windows degrade badly (w=20 is worse than using no rolling features at all).
  A 20-cycle window lags a monotonic trend, so the smoothed value systematically
  under-states current degradation.

### Limitation found: sensor selection is a no-op on synthetic data

`select_informative_sensors` drops **0 of 21** sensors on the synthetic set. The
generator gives its non-trending sensors `normal(0, 0.5)` noise → variance ≈ 0.25,
which is ~250× the `1e-3` threshold, so everything passes. Real FD001 has roughly six
genuinely flat sensors (s1, s5, s10, s16, s18, s19) that should be dropped.

So `generate_synthetic.py` does not actually achieve what its own comment claims
("mimic that by only letting a subset drift"), and the ablation's raw arm would be
~14 features on real data, not 21.

**Decision:** logged, not fixed. Fixing the generator would invalidate every number
above and force a full re-benchmark, and the real dataset is about to replace it
anyway. Added to TODO as a fidelity fix for the fallback path. The unit tests do cover
the selection logic properly — their fixtures use truly constant sensors — so this is a
gap in the synthetic *data*, not in test coverage.

### Tier 2 — Sequence models (LSTM / GRU / 1D-CNN)

`src/seq_models.py` holds the three architectures behind one contract
(`(B, T, F) → (B,)`); `src/train_seq.py` is the single training and evaluation path.
`model_lstm.py` shrank to a thin entry point over the harness — it previously trained
without ever evaluating, and carried duplicate copies of the dataset and model classes.

Protocol decisions, both places this is easy to get wrong:

- **Split by engine, not by cycle.** Adjacent cycles of one engine are near-duplicates,
  so a mid-engine split leaks the target into validation.
- **Validation is scored over all windows, not last cycles.** Training engines run to
  failure, so a held-out engine's final cycle always has RUL = 0 — scoring only there
  would measure one degenerate point and mirror nothing about the truncated test set.

First run at the scaffold's defaults (seq_len=30, hidden=64) converged at epoch 6 and
then overfitted hard — train MSE fell 90 → 3.3 while val RMSE stalled near 12. Early
stopping with best-checkpoint restore caught it. Test: RMSE 11.037 / PHM 92.6, i.e.
**worse than the RandomForest**.

### Tier 2 — Hyperparameter sweep, and a correction

27 configurations (seq_len × hidden × lr), ranked by **validation** RMSE — never test,
which would be selecting on the evaluation set.

**This corrects the conclusion above.** The default LSTM losing to the RF was a
capacity problem, not a property of sequence models on this data:

| config | val RMSE | test RMSE | test PHM |
|---|---|---|---|
| RandomForest baseline | — | 10.309 | 85.0 |
| LSTM at scaffold defaults (30 / 64 / 1e-3) | 11.844 | 11.037 | 92.6 |
| **LSTM selected on val (50 / 128 / 3e-4)** | **7.841** | **7.804** | **48.3** |

The tuned LSTM beats the RandomForest by **24% on RMSE and 43% on PHM**. The
`seq_len=50, hidden=128` corner dominates: all three learning rates there land at
7.8–9.3 RMSE while every other cell sits at 10.9–15.1. Two honest caveats — the grid's
best cell is at its own boundary, so capacity looks unsaturated and the sweep should be
extended rather than treated as converged; and val selection happening to also pick the
grid's best test RMSE is a good sign but partly luck at n=50 test engines.

### Tier 6 — Conformal prediction, and a hypothesis that mostly failed

Motivated by the Tier 1.2 finding, I predicted that shifting predictions earlier would
buy a large PHM reduction, and recorded in TODO that this made the uncertainty work the
highest-value item for that metric. **Measured, the effect is much smaller than that
claim.**

Split conformal: 70 fit engines, 30 calibration engines (whole engines, no leakage),
residual quantiles from the held-out set. Note the baseline here (RMSE 10.517 / PHM
82.1) differs from the headline baseline because it is fit on 70 engines, not 100.

| q | offset (cycles) | RMSE | PHM | % late |
|---|---|---|---|---|
| 0.30 | −4.48 | 11.491 | 115.8 | 68 |
| 0.50 | −0.55 | 10.540 | 84.4 | 48 |
| 0.60 | −0.02 | 10.518 | 82.1 | 46 |
| **0.70** | **+1.86** | **10.654** | **77.5** | **38** |
| 0.80 | +6.78 | 12.428 | 87.2 | 22 |
| 0.90 | +16.11 | 19.041 | 182.4 | 6 |

The curve is U-shaped with a **shallow** minimum: the best shift is only +1.86 cycles
and recovers **5.6% of PHM for a 1.3% RMSE cost**. Favourable, and in the predicted
direction — but nothing like the gain implied by "71% of the score comes from late
predictions."

Why the reasoning failed: tail concentration does not imply a *uniform* offset helps.
The PHM asymmetry (exp(d/10) late vs exp(−d/13) early) is only mildly lopsided, so
moving all 50 engines earlier pays a small early penalty on every one of them to shave a
few large late errors, and the many small costs nearly cancel the few large gains.
Reducing that tail needs **per-engine** conservatism scaled to each engine's own
predictive uncertainty, not one global constant. Logged as the follow-up.

I also had the sign convention backwards on the first implementation — with
`adjusted = point − quantile(resid, q)`, **high** q is the conservative direction, not
low. The first run swept 0.05–0.5 and so explored mostly the *unsafe* half of the curve.
Fixed, and the convention is now stated explicitly in the module and the report.

Conformal interval coverage came out slightly conservative:

| nominal | empirical | mean width |
|---|---|---|
| 80% | 84.0% | 30.0 cycles |
| 90% | 92.0% | 39.4 cycles |
| 95% | 98.0% | 45.7 cycles |

Over-coverage is the expected direction given the known mismatch: calibration sees
every cycle of its engines while the test set observes one truncated point per engine,
so the calibration residual spread is wider than the test-time spread.

The report text is generated from the measured gain via a `_verdict()` helper rather
than written by hand, so it cannot drift into claiming a win the numbers do not support.

### Tier 2 — Model comparison

All models on FD001 with the sweep-selected sequence config (seq_len=50, hidden=128,
lr=3e-4), 80 epochs, patience 10:

| model | RMSE | PHM | % late | params | train time |
|---|---|---|---|---|---|
| RandomForest | 10.309 | 85.0 | 44 | — | 17s |
| LSTM | 7.804 | 48.3 | 48 | 235,073 | 44s |
| **GRU** | **7.292** | **44.9** | 54 | 177,345 | 297s |
| CNN | 14.285 | 135.8 | 32 | 78,273 | 37s |

**The GRU wins, and it was included precisely as the control.** It beats the LSTM on
both metrics (RMSE −6.6%, PHM −7.0%) with 25% fewer parameters, which says the LSTM's
extra gate and cell state buy nothing on this data. On smooth monotonic degradation
that is a believable result — there is little long-range structure for the richer gating
to exploit.

**The CNN number is not a fair verdict on the architecture.** Every sequence model here
ran with hyperparameters selected by the *LSTM* sweep. A 1D-CNN with hidden=128 over a
50-cycle window is a very different model from a recurrent one at those settings, and
the ablation already showed long windows hurt on this data. Sweeping the CNN separately
is on the TODO; until then its row should be read as "untuned", not "worse".

Two other notes. The GRU took 297s against the LSTM's 44s — it trained for many more
epochs before early stopping, not because a GRU step is slower. And `% late` rises for
the better models (44% → 54%), another reminder that RMSE/PHM improvements and the
late/early balance are separate axes.

### Tier 4 — Lint, types, tooling

`ruff`, `black` (line length 100) and `mypy` all clean on `src/` + `tests/`; suite green
throughout at 114 tests.

The mypy pass found one thing worth fixing rather than silencing: `train()` tracked its
best checkpoint in a single dict `{"val_rmse": float, "epoch": int, "state": dict|None}`.
mypy inferred `dict[str, float | None]` and then flagged five downstream errors —
comparing a float to `None`, subtracting from `None`, passing a float to
`load_state_dict`. All were symptoms of one design flaw: bundling three unrelated types
in one dict defeats type checking on every access. Split into three locals. The other
two errors were genuine `None`-initialised variables that later hold floats.

Also added: `pyproject.toml` (ruff/black/mypy config), `Makefile` (`make all`
regenerates every artifact), `Dockerfile` with torch behind a build arg — the core image
stays ~500MB because the RF baseline, ablation and tests need no torch — and GitHub
Actions with a deliberate **no-torch job**, which is what actually proves the README's
"core pipeline needs only numpy/pandas/matplotlib" claim rather than just asserting it.

### Tier 3 — Live twin, and two bugs found by running it

Built the online feature path first, because the inherited `stream_demo` was not actually
streaming: it called `build_xy` across the whole test set up front and replayed
pre-computed rows. Right numbers, but it depended on data from the future and would break
on real telemetry. `FeatureSpec` now persists the fitted contract at training time and
`OnlineFeatureBuilder` rebuilds features one reading at a time from *training* statistics —
recomputing them from the stream is the train/serve skew bug, which throws no error and
just quietly degrades predictions. Tested to reproduce the batch path to 1e-9, and tested
that interleaved engines keep isolated state (which is what MQTT actually delivers).

Then I ran the dashboard in a browser rather than assuming it worked, and found two things:

1. **`streamlit run src/dashboard.py` cannot work at all.** `streamlit run` executes its
   target as a top-level script, so a module inside a package has no parent package and
   every relative import raises `ImportError: attempted relative import with no known
   parent package`. I confirmed the failure mode explicitly (via `runpy.run_path`) before
   fixing it rather than guessing. Fix: a root-level `app.py` shim — streamlit then puts
   the repo root on `sys.path`, making `src` a real package.
2. **The fleet table recomputed ~7,500 single-row predictions on every interaction.**
   Streamlit re-runs the whole script per widget change, so moving the threshold slider
   paid the full cost again. Now `@st.cache_data`.

Verified end to end: replaying engine 1 in the dashboard alerts at cycle 213 and finishes
at 16.8 vs actual 22 — identical to the CLI `simulate` run, and the fleet view's
last-known RUL for engine 1 agrees at 16.8. Three surfaces, one online path, same numbers.

Demo GIF is rendered by `src/make_demo_gif.py` from the twin's real output rather than
screen-recorded, so it cannot drift from the model and needs no capture tooling. Verified
first, middle and last frames (the last shows the red MAINTENANCE banner).

### Tier 4 — Coverage, and a third bug

Coverage started at **25%**. The library code was well covered but every
report-generating script was at **0%**, meaning a crash in `make all` would only surface
when a human ran it. Added two smoke-test files — torch-free and torch-gated, so the
no-torch CI job stays complete — taking coverage to **78%** with core modules at 91–100%.

**The smoke tests immediately found a real bug.** `FeatureSpec.save()` bound its output
path as a *default argument*, which Python evaluates once at import. Redirecting
`OUTPUTS_DIR` was therefore silently ignored, and the feature spec was written to the real
`outputs/` directory regardless of configuration — I noticed because a test run wrote into
the actual repo. Now resolved at call time, and `train_baseline` passes an explicit path.

Two of my new tests failed on my own wrong assumptions rather than on defects, and both
were rewritten to assert invariants instead of behaviour I had guessed at:

- I asserted early stopping would trigger before the epoch cap. On three fixture engines
  with one validation engine, validation kept improving every epoch, so it legitimately
  ran to the cap. The test now asserts the *rule*: the run never exceeds the cap, and if
  it ended short then `patience` epochs had passed without improvement.
- I asserted the comparison report would carry the SYNTHETIC caveat. The fixture writes
  into `data/CMAPSSData/`, so `using_real_data()` correctly reported "real". The caveat
  branch now has its own test with the flag patched.

Deliberate remaining gaps, stated rather than hidden: `dashboard.py` at 0% (needs a
streamlit runtime — verified manually in a browser and cross-checked against the CLI) and
`telemetry.py` at 39% (MQTT publish/subscribe need a broker; the broker-free `simulate`
path, which shares every line of scoring logic with the subscriber, is covered).

### Tier 5 — Docs

`README.md` rewritten: results tables, figures, the GIF, the twin-loop diagram, a
design-decisions section explaining *why* each protocol choice exists, and an explicit
limitations list. The synthetic-data caveat is the first thing a reader sees, because
every number in the file depends on it.

`docs/writeup.md` is the technical article — problem framing, protocol, what worked, what
didn't (the conformal hypothesis and the inverted sign convention both get their own
section), and limitations. `docs/benchmarks.md` pairs every table with the command that
regenerates it.

---

## Session summary

Six tiers touched, 9 commits, all green. 146 tests at 78% coverage; ruff, black and mypy
clean.

**Three real bugs found and fixed**, all of which produced correct-looking output rather
than errors: the streaming demo depending on future data, `streamlit run` on a package
module, and the default-argument path binding in `FeatureSpec.save()`. Two of the three
surfaced only by actually running the thing rather than reading it.

**Two of my own claims corrected by measurement**: that sequence models lose to the
RandomForest here (they win decisively once given capacity), and that a conservative shift
would exploit the PHM tail (it recovers 5.6%, not the large gain I predicted).

**The headline caveat stands**: `data/CMAPSSData/` is still empty, so every metric in this
log is a plumbing check. The pipeline, protocol, tests and tooling are ready for the real
data; `make all` regenerates everything once it lands.

---

## 2026-07-30 — Repo relocated

Moved out of the session-scoped agent outputs directory to
`~/Documents/Side Projects/twin-turbofan`, joining the rest of the twin-* family. All 9
commits intact, tree clean, `main` and `continuous-build` both present.

**The move broke every `.venv/bin` console script** — 35 of them. `pip` bakes the
interpreter's *absolute* path into each shebang, so `pytest`, `black`, `mypy` and
`streamlit` all died with `bad interpreter: No such file or directory`. The venv itself
was fine: `.venv/bin/python` is a symlink to the unmoved base interpreter, so imports and
`python -m pytest` kept working while bare `pytest` did not. That asymmetry is the
confusing part — the error names a missing interpreter, not a moved project.

Two fixes, one reactive and one preventive:

- `scripts/fix_venv.py` rewrites the stale prefix in place, idempotently. Verified
  properly: running it clean reports "all already correct", and after deliberately
  breaking `.venv/bin/pytest` with a stale path *containing a space* it repointed exactly
  that one script and `pytest --version` worked again. The space matters — the old path
  lived under "Application Support", and a whitespace-delimited match would have failed.
- The Makefile now invokes every tool as `$(PY) -m <tool>` rather than `.venv/bin/<tool>`,
  so a future relocation cannot break the build in the first place. `make fix-venv` exists
  for when you want the bare commands back.

Worth noting a mistake I made mid-fix: my first attempt filtered candidate files with
`head -c 200 | grep -F "$OLD"`, which found **0 of 35** scripts. The old path was 212
characters, so the 200-byte window truncated it mid-path and the match never landed. The
loop reported success while doing nothing. Rewrote it in Python over whole file contents.

`make check` green at the new location: 146 tests, mypy clean, ruff clean.

### Tier 4 — Config layer

`config.yaml` + `src/config.py`. The settings it collects were literals spread across
`src/` — the RUL cap in `data_loader`, the rolling window in two places, the alert
threshold in `telemetry`. That makes a protocol hard to audit: a reviewer had to read five
modules to learn what a run actually used.

Precedence is defaults < `config.yaml` < `TWIN_CONFIG` < CLI. Defaults live in a dataclass
so the package still imports in a fresh checkout or a container with no config mounted, and
CLI flags win so a one-off run never requires editing a tracked file.

**Unknown keys raise.** This is the design decision worth defending: silently ignoring a
typo like `rul_capp: 100` would leave the real cap at 125 and the run would look completely
correct while measuring something else. Sections, keys and the nested `twin.mqtt` block are
all validated against an explicit schema.

PyYAML turned out not to be installed in the venv — I had checked for it in the *conda*
env earlier, which was the wrong environment. So the graceful-degradation path got
exercised for real: the whole config test file skipped via `importorskip`. Added PyYAML to
`requirements.txt` (small, pure-python) while keeping the fallback for minimal installs.

Verified by observation rather than assertion: with `rul_cap: 60, rolling_window: 3` the
baseline moves from RMSE 10.309 to 4.276 and the persisted feature spec records `window: 3`.
That override run overwrote `outputs/`, so the defaults were re-run afterwards to keep the
committed artifacts consistent with the documented numbers.

21 tests, including a drift guard asserting `config.yaml` cannot silently contradict the
dataclass defaults — otherwise a reader of the YAML and a reader of the code would disagree
about what the experiment used.

### Tier 6 — Per-engine conservatism: the hypothesis holds this time

`src/uncertainty_per_engine.py`. The global-offset experiment recovered only 5.6% of PHM and
I argued the fix was conservatism scaled to each engine's own uncertainty. This tests that,
using the spread of the RandomForest's per-tree predictions as `sigma_i`.

**The comparison is matched on amount, which is the whole design.** Subtracting `k · sigma_i`
shifts predictions earlier by `mean(k · sigma_i)` on average, so a per-engine rule could
"win" merely by being more conservative. Every setting is therefore compared against the
uniform offset with the *same mean shift*; any remaining difference is allocation alone.

| k | mean shift | per-engine PHM | uniform PHM | delta |
|---|---|---|---|---|
| 0.00 | 0.00 | 82.1 | 82.1 | — |
| **0.25** | **2.20** | **74.4** | 77.2 | **−2.8** |
| 0.50 | 4.40 | 75.4 | 78.9 | −3.5 |
| 0.75 | 6.60 | 84.8 | 86.3 | −1.5 |
| 1.00 | 8.80 | 103.1 | 99.1 | +4.0 |
| 1.50 | 13.20 | 168.1 | 141.4 | +26.7 |
| 3.00 | 26.40 | 795.5 | 423.7 | +371.8 |

Calibration engines independently chose **k=0.25**, which is also the best test setting —
so the headline is not selected on test. There: PHM **74.4** vs 82.1 unadjusted (**−9.4%**),
beating both the unadjusted baseline and the 5.6% from the best global offset, *and* beating
the same-amount uniform shift by 2.8 PHM.

Precondition measured rather than assumed: `corr(sigma, |error|) = +0.546`. Allocation can
only beat a uniform shift if the model's own uncertainty ranks which engines it will get
wrong; had this been ~0, no weighting scheme built on it could have helped, and I wrote the
verdict function to say so explicitly in that case.

Honest limits: the gain exists only for mild `k`. Sigma reaches ~16 cycles, so `k >= 1`
drags uncertain engines tens of cycles early and the exponential early-penalty term takes
over — per-engine is then *worse* than uniform. Also, tree-to-tree spread measures
disagreement between trees, not the full conditional distribution, so it understates true
predictive variance; it is a ranking signal, not a calibrated sigma.

I had to rewrite the verdict logic once. The first version counted wins uniformly across all
`k`, which let absurd over-corrections (k=2, k=3) outvote the setting calibration actually
selected, and reported "inconclusive" for what is a clear result in the only regime anyone
would deploy.

**Net:** the earlier conformal diagnosis was right; the *uniform offset* was the limitation,
not the idea of trading a little RMSE for tail safety.

### Tier 4 — run_continuous.sh hardening

The script that started this whole run would have spun for four hours and exited 0. Fixed
its three silent-failure modes rather than only documenting them:

1. **Missing the agent CLI on PATH.** `set -uo pipefail` without `-e` meant each iteration
   printed `command not found`, slept 3s, and looped — ~4,700 no-op iterations. Now a
   fail-fast precondition that also names authentication, since an unauthenticated CLI
   cannot complete a login flow from inside the loop.
2. **Committing from the wrong repo root.** Now asserts `git rev-parse --show-toplevel`
   equals `$PWD`, with the reason stated. Tested both ways using a stub CLI on PATH: a
   nested directory is blocked with exit 1, a proper root passes.
3. **Hot-looping on instant failures.** An iteration exiting non-zero, or returning in
   under 20s, did no real work. Those are now counted, backed off progressively, and abort
   after 3 consecutive — so a quota or auth error stops rather than spinning.

Added `--dry-run`. Testing it surfaced a small real bug: `git rev-parse --abbrev-ref HEAD`
prints a git fatal on a repo with no commits — exactly the state guard 2 tells you to
create. Switched to `git symbolic-ref` with a fallback.

### Verifying the dependency-light claim

The README and CI both asserted that the core pipeline runs without torch/streamlit/paho and
that the optional tests skip themselves. Nobody had checked. `make check-minimal` now does.

**My first attempt at the check was wrong**, and wrongly implicated the code. I put a stub
`torch.py` on `PYTHONPATH` that raised `ImportError`, and pytest reported two *collection
errors* rather than skips — which looked like the guarantee was broken. It wasn't: with a
stub, the module exists but fails internally, and `pytest.importorskip` deliberately
re-raises that case so a genuinely broken install can't hide behind a skip. An uninstalled
package raises `ModuleNotFoundError` instead. Replaced the stub with a `sys.meta_path`
finder that raises `ModuleNotFoundError`, which is a faithful simulation.

Measured, with the extras hidden:

| | full env | extras hidden |
|---|---|---|
| tests | 173 passed | 126 passed, 2 skipped |
| baseline RMSE | 10.309 | 10.309 |
| live twin first alert | cycle 213 | cycle 213 |

The 2 skips are module-level `importorskip` skips covering 45 tests in the two torch-gated
files. The baseline, error analysis and the broker-free live twin all run and produce
identical numbers, so the guarantee holds as written.

### Correction: the 7.489 vs 7.292 gap was NOT nondeterminism

The GRU sweep selected the same configuration as the LSTM sweep
(`seq_len=50, hidden=128, lr=3e-4`) but reported test RMSE **7.489**, where the earlier
comparison run had reported **7.292** at what looked like the same config and seed. I called
that MPS run-to-run variance. **That was wrong**, and it took two minutes to check.

The two runs had different epoch budgets — the sweep used `epochs=60, patience=8`, the
comparison `epochs=80, patience=10`. Re-running each reproduces its number exactly:

| budget | best epoch | epochs run | RMSE | PHM |
|---|---|---|---|---|
| ≤60, patience 8 | **60** (hit the cap) | 60 | 7.489 | 46.3 |
| ≤80, patience 10 | 73 | 80 | 7.292 | 44.9 |

Two things follow. First, training here is **exactly reproducible at a fixed seed on MPS** —
better than I claimed, and worth having verified rather than hedged. Second, the GRU is
**epoch-limited, not converged**: it was still improving when both budgets truncated it, so
7.292 is a floor rather than its best. The comparison was re-run with a 150-epoch budget for
that reason.

Lesson recorded because it generalises: "probably nondeterminism" is a cheap explanation that
stops you looking. The runs differed in a parameter I had passed myself.

### Tier 2 — CNN swept on its own grid: the caveat was right

The comparison table had flagged the CNN's 14.285 RMSE as "untuned, not worse", because all
three architectures ran at hyperparameters chosen by the *LSTM's* sweep. Sweeping the CNN
properly confirms that and then some:

| arch | selected config | test RMSE | test PHM |
|---|---|---|---|
| LSTM | seq=50 hidden=128 lr=3e-4 | 7.804 | 48.3 |
| GRU | seq=50 hidden=128 lr=3e-4 | 7.489 | 46.3 |
| CNN | **seq=20 hidden=32** lr=3e-4 | **10.758** | **79.6** |

The CNN prefers the **opposite corner of the grid** — the shortest window and the fewest
channels, where both recurrent models want the longest window and the most capacity. Given
its own configuration it improves from 14.285 → 10.758 RMSE (−25%) and 135.8 → 79.6 PHM
(−41%), which moves it from "much worse than the RandomForest" to roughly level on RMSE and
clearly better on PHM.

So the original shared-config table understated the CNN badly, and `src/compare.py` was
fixed rather than just annotated: it now reads each architecture's own
`outputs/sweep_<arch>.json` and records a `config_source` column, with `--shared-config` kept
only for reproducing the old table. A single config is not a fair comparison between
architectures whose inductive biases want different windows and capacities.

### Test suite cost: a fast/slow split

Adding end-to-end coverage for `compare.swept_config` and `src/variance.py` pushed the suite
from ~17s into minutes, because those cases drive entry points that train models. A suite
that slow stops being the inner feedback loop.

Split it: a registered `slow` marker, excluded by default via `addopts = -m "not slow"`, with
`make test-all` and `make test-slow` for the rest and CI running everything (`pytest -m ""`).
Only genuinely expensive cases are marked — the pure `swept_config` selection logic was
pulled into its own unmarked class so it still runs by default, since that is where the real
correctness risk lives (it must rank by *validation*, not test).

Counts: **180 total, 176 default, 4 slow.**

Also worth recording: several commands during this stretch hit their timeout, and the cause
was contention rather than hangs — a 150-epoch comparison was saturating the GPU while I ran
tests against it. Anything MPS-bound needs to be serialised on this machine, not overlapped.

### Lint coverage gap between the automation and my own commands

Validating the CI workflow (never previously parsed, since there is no remote to run it)
turned up an inconsistency rather than a syntax error: both the CI lint job and `make lint`
checked only `src tests conftest.py`, while I had been passing `app.py scripts` by hand on
every manual check. So the automation was *weaker* than what I was actually running — lint
errors in `app.py` or `scripts/` would have passed CI silently.

Fixed by defining `LINT_PATHS` once in the Makefile and using the same list in the CI job,
and by making `make lint` run `black --check` too rather than only `ruff` (previously
formatting was only enforced in CI, so `make check` could pass locally on unformatted code).
The wider list passes: 38 files unchanged.

### Tier 6 — Attention model and interpretability (written, not yet run)

`AttentionRegressor`: GRU encoder plus additive attention pooling, registered in
`ARCHITECTURES` so the existing harness, sweep and comparison pick it up for free.

Two deliberate design choices:

**Additive pooling rather than multi-head self-attention.** The reason for adding attention
here is interpretability, and self-attention maps are poor explanations — many heads, and
attention-to-token is not token-importance. Additive pooling gives exactly one weight per
cycle, summing to 1, and the prediction *is* that weighted sum, so the weights are the
readout rather than a proxy for it.

**A GRU encoder underneath.** The sweep showed recurrence is what works on this data, so
attention replaces only the final-timestep readout. Any gain is then attributable to the
pooling rather than to having swapped in a different sequence model.

`src/interpret.py` pairs the attention profile with **permutation importance** over features,
because the two answer different questions: attention says where the estimate rests,
permutation says what the model relies on, and attention can be high on a cycle whose sensors
carry no usable signal. The report states the correlated-inputs caveat rather than presenting
permutation importance as clean per-sensor attribution — C-MAPSS sensors move together during
HPC degradation, so it ranks signal *groups*.

Verified without training: output shape `(B,)`, attention rows sum to 1, varying `seq_len`
accepted. The permutation shuffle got its own check after I found a fragility — the first
version indexed a batch with a permutation of `len(dataset)`, which is only a valid
permutation while the whole dataset fits in one batch, and otherwise maps many rows onto few
so the "shuffle" is biased rather than random. Now permutes within the batch, and a unit check
confirms other columns are untouched, the column really is a row permutation, it actually
moves, and per-row time order is preserved.

Not yet run at full scale — the comparison job has the GPU.

### Tier 2 — The fair comparison table

Re-ran `compare.py` with each architecture on its own swept configuration and a 150-epoch
budget so nothing is truncated:

| model | config | RMSE | PHM | % late | params |
|---|---|---|---|---|---|
| **GRU** | seq=50 hid=128 lr=3e-4 | **7.292** | **44.9** | 54 | 177,345 |
| LSTM | seq=50 hid=128 lr=3e-4 | 7.804 | 48.3 | 48 | 235,073 |
| CNN | seq=20 hid=32 lr=3e-4 | 10.236 | 70.6 | 36 | **10,401** |
| RandomForest | — | 10.309 | 85.0 | 44 | — |

**The CNN result is the one that changed most, and it is an efficiency finding.** At
10,401 parameters — 17× fewer than the GRU, 23× fewer than the LSTM — it matches the
RandomForest on RMSE (10.236 vs 10.309) and beats it by 17% on PHM (70.6 vs 85.0). The
original shared-config table had it at 14.285 / 135.8 and flagged as "untuned, not worse";
that flag was right, and the gap was larger than I expected.

Both the CNN and the GRU improved again when the budget went from 60 to 150 epochs
(CNN 10.758 → 10.236), so the sweep's own runs were epoch-limited too — the sweep is useful
for *ranking* configurations but its absolute numbers are pessimistic. The GRU is now
genuinely converged: 150 epochs reproduces the 80-epoch result exactly (7.292).

An observation worth keeping: **`% late` runs opposite to quality here.** The CNN is the most
conservative model (36% late) and has the worse PHM; the GRU is late most often (54%) and has
the best. Being early only pays if the errors are small — which is the same lesson the
conformal experiment taught from the other direction.

### Tier 2 — Ensemble: no, and the near-miss is the interesting part

`w * GRU + (1 - w) * forest`, weight chosen on 20 held-out validation engines. Validation
prefers more GRU monotonically and selects **w=1.0** — the sequence model alone. So blending
does not help.

The instructive part: **test disagrees.** `w=0.8` scores 7.132 RMSE / 42.3 PHM, beating the
GRU alone (7.292 / 44.9) on both. Had I chosen the weight on test, this would read as a 2%
RMSE and 6% PHM win from ensembling — a completely fabricated finding, because with 50 test
engines and eleven weights on offer something will always beat both endpoints. The
validation/test disagreement about where the optimum sits *is* the evidence that the apparent
gain is selection noise. Choosing on validation is what stopped me writing it up as a win.

### Run-to-run variance — and it invalidates my own headline ranking

| arch | condition | RMSE mean | spread | PHM mean |
|---|---|---|---|---|
| LSTM | same seed ×3 | 7.804 | **0.000** | 48.3 |
| LSTM | different seeds ×3 | 8.302 | 1.723 | 53.7 |
| GRU | same seed ×3 | 7.292 | **0.000** | 44.9 |
| GRU | different seeds ×3 | 8.312 | 1.767 | 59.9 |

**Seeding fully pins this workload** — same-seed repeats are identical to the digit on MPS,
confirming the epoch-budget diagnosis independently.

**The GRU-beats-LSTM conclusion does not survive re-seeding.** Across-seed means are 8.302 vs
8.312: a gap of **0.010 RMSE against a spread of 1.767**. The single-seed comparison table
shows the GRU ahead by 0.512, but that ordering is an artefact of seed 42. I had written that
the GRU "won" as the capacity control and drawn an inference from it — that the LSTM's extra
gate earns nothing. The premise is not supported at this sample size; the honest statement is
that the two are tied and this data cannot separate them.

**Worse for the report as a whole: seed 42 is a lucky draw.** Both models beat their
seed-average by 0.5–1.0 RMSE. So every sequence-model number quoted anywhere in this project
is optimistic by roughly that much. Added to TODO as a real deficiency: results should be
seed-averaged, which costs ~3× the compute per row. The RandomForest rows are unaffected —
deterministic given `random_state`, and no train/val split.

This is the second time in this run that a difference I reported as meaningful turned out to
be an artefact of a knob I set myself (the first being the epoch budget). Both were caught by
measuring the thing rather than reasoning about it.

### A plotting bug that cost a training run

`src/interpret.py` crashed *after* 152s of training and after computing both the attention
profile and the permutation importances, in `_plot`: I had written `color="tab:teal"`, which
is not a matplotlib colour (the `tab10` palette has cyan, not teal). Everything was recomputed
on the re-run, so nothing was lost but time.

Two takeaways applied rather than just noted: the fix is `tab:cyan`, and I added a sweep over
every `color="..."` literal in `src/` validated against `matplotlib.colors.to_rgba` — clean.
The deeper issue is ordering: expensive computation should persist its results *before*
rendering, so a cosmetic failure cannot discard a training run. Logged in TODO.

### A registry that wasn't the single source of truth

The variance run for the attention model failed immediately:

```
variance.py: error: argument --archs: invalid choice: 'attention'
              (choose from 'lstm', 'gru', 'cnn')
```

`attention` was registered in `ARCHITECTURES` and worked correctly as a model — `interpret.py`
had just trained it to the best result in the project — but **four entry points hardcoded**
`choices=["lstm", "gru", "cnn"]`: `variance.py`, `sweep.py`, `ensemble.py`, `train_seq.py`.
Only `compare.py` derived its list from the registry, which is why the comparison picked the
new architecture up for free and nothing else did.

The point of a registry is that adding an entry reaches everything. Duplicating the list in
four argparse calls quietly defeated that, and the failure surfaced at the CLI layer *after*
a successful three-minute training run elsewhere — the most annoying place for it to appear.

Fixed by publishing `ARCH_NAMES = sorted(ARCHITECTURES)` next to the registry and using it in
all four. Verified every CLI now offers `{attention,cnn,gru,lstm}`.

Guarded against recurrence with three tests rather than trusting discipline: `ARCH_NAMES`
matches the registry, no module contains a `choices=["lstm"...]` literal, and each of the four
CLI modules actually references `ARCH_NAMES`. The middle one is a source-level check, which is
the right shape for a source-duplication bug.

Also corrected the `seq_models` docstring, which still said "All three" and "Why three" after
the fourth architecture was added — the same class of drift, in prose.

### Tier 6 — Interpretability results

Re-ran cleanly after the colour fix, and reproduced the earlier numbers exactly (RMSE 6.248,
PHM 31.0, best epoch 27, and identical permutation importances) — a third independent
confirmation that seeded runs are deterministic here.

**Attention concentrates on recent cycles, as predicted.** The most recent 13 of 50 cycles
hold **62.1%** of the total weight against **26.0%** for uniform attention, and the oldest
cycle in the window receives ≈0. The expectation was written down before looking: monotonic
degradation means the newest readings carry the most information and older ones largely repeat
it. Had attention come out flat, the model would have been averaging and the sequence
structure would have been doing nothing.

**Permutation importance ranks raw sensors above engineered ones:**

| feature | ΔRMSE when shuffled |
|---|---|
| `s7` | +2.454 |
| `s20` | +2.383 |
| `s4` | +1.757 |
| `s8` | +1.404 |
| `s12` | +1.321 |
| `s4_rmean` | +1.031 |

The first engineered feature appears sixth. That is consistent with the ablation's finding
that rolling statistics add only ~3% on this data — the model is reading the raw sensor
trajectory and the smoothed versions are largely redundant. Read as signal *groups* rather
than clean per-sensor attribution: correlated inputs share responsibility under permutation.

### Tier 6 — Attention: the best number is not the best model

At seed 42 `AttentionRegressor` produced the best result in the project: RMSE **6.248**, PHM
**31.0**, against the GRU's 7.292 / 44.9 — a 14% RMSE and 31% PHM improvement. Given what the
LSTM/GRU variance study had just taught me, I did not report it before checking.

| arch | seed 42 | seed-mean | RMSE std | RMSE spread | PHM std |
|---|---|---|---|---|---|
| attention | **6.248** | 8.210 | 2.667 | **5.733** | 32.5 |
| GRU | 7.292 | 8.312 | 0.747 | 1.767 | 10.6 |

**The means tie** — 8.210 vs 8.312, a 0.102 gap against a 5.733 spread — so the single-seed win
is not architectural. Third claim in this run that measurement removed.

**The stability difference is the real result.** Attention's across-seed spread is 3.2× the
GRU's, and its PHM std is 3.1× larger. Seed 42 flatters attention by 1.962 RMSE against its own
seed-average; the GRU's equivalent penalty is 1.020. So the attention model is not merely "not
better" — it is *riskier to train*, and its headline number is the least trustworthy in the
report.

Stated plainly in the docs: **the GRU is the better engineering choice**, even though attention
owns the best number. For a maintenance model, a best case you cannot reproduce is worth less
than a slightly worse one you can. The attention model keeps its place because the pooling
weights are what make the interpretability analysis possible — which is why it was added.

I extended `variance.py` to report this rather than leaving it for a reader to compute: when one
architecture's spread exceeds another's by more than 2×, the report now says so explicitly. The
original verdict logic only compared the *gap* against the spread, which would have called this
"indistinguishable" and stopped — hiding the more actionable half of the finding. Regenerated
the report from the saved JSON rather than retraining.

### Seed-averaged reporting

The variance study's uncomfortable implication was that every headline sequence number came
from a single seed that happened to be favourable. `compare.py` now takes `--seeds N`
(default **3**) and reports `mean ±half-range` per model × dataset, with per-seed detail
below.

Two decisions inside that:

**The RandomForest participates on the same footing.** `make_model(seed)` threads the seed to
`random_state`. The forest has no train/val split to move, so I expected a negligible spread —
but a two-seed check gave 10.309 and 10.146, a 0.16 RMSE range. Small, and not nothing.
Excluding it would have quietly held one row to a different standard than the rest.

**Half-range rather than standard deviation.** With three samples a std invites more precision
than the data supports; `±half-range` says plainly how far apart the runs actually landed. The
function also falls back to a bare number when `--seeds 1`, so the old single-seed table is
still reproducible.

### Persist before render — applied and verified

Reordered `interpret`, `variance`, `ensemble`, `ablation` and `uncertainty_per_engine` to write
their JSON *before* plotting. `variance.py` was the most exposed: twelve trainings behind a
figure that could fail on a typo.

Verified rather than assumed — I injected a `ValueError` into `uncertainty_per_engine`'s plot
and confirmed the run still left a complete JSON behind (all 8 rows, correct `k_selected` and
correlation), then restored and regenerated cleanly. Given that the original bug *was* a typo
in a plotting call, checking the fix by actually breaking a plot seemed like the only honest
test.

### Extending the LSTM sweep — deprioritised on evidence

The winning sweep cell sits on the grid boundary, which normally argues for extending the grid.
The variance study argues against it: across-seed spread is 1.767 RMSE for the GRU and 5.733
for attention, while the differences a wider single-seed grid would rank are well under 1.0. A
bigger grid would generate rankings below the noise floor and read as progress. Marked blocked
rather than open, with the condition stated — worth doing only together with seed-averaged
sweeping, which multiplies an already ~15-minute sweep by the seed count.

### Seed-averaged comparison — it removed two conclusions I had already published

Ran `compare.py --seeds 3 --epochs 100`. Fifteen runs: RandomForest + four architectures ×
seeds 42/43/44.

| model | RMSE (3 seeds) | range | seed 42 | seed 42 vs mean |
|---|---|---|---|---|
| ATTENTION | **8.210** ±2.866 | 5.733 | 6.248 | −1.962 |
| LSTM | 8.302 ±0.861 | 1.723 | 7.804 | −0.498 |
| GRU | 8.312 ±0.883 | 1.767 | 7.292 | −1.020 |
| RandomForest | 10.191 ±0.095 | **0.190** | 10.309 | **+0.118** |
| CNN | 11.359 ±1.117 | 2.235 | 10.236 | −1.123 |

PHM: LSTM 53.7 ±8.5, attention 54.6 ±34.8, GRU 59.9 ±11.9, forest 81.4 ±3.0, CNN 88.2 ±16.9.

**Two conclusions I had already written into the README and benchmarks did not survive:**

1. *"The GRU beats the LSTM, so the LSTM's extra gate earns nothing."* Means differ by 0.010
   against ranges of ~1.7. Tied. (Already corrected once; this confirms it with the full table.)
2. *"The CNN matches the RandomForest on RMSE and beats it 17% on PHM — an efficiency win."*
   Averaged it is **11% worse on RMSE** and **8% worse on PHM**. Seed 42 flattered it by 1.123
   RMSE. What survives is narrower: a 10,401-parameter model lands within ~11% of the forest,
   which is a real efficiency observation but not a win. I had stated the strong version
   confidently, twice.

**The systematic finding is the one I did not anticipate.** Seed 42 flattered *every* neural
model — by 0.498, 1.020, 1.123 and 1.962 RMSE — and was the only seed where the forest did
slightly *worse* than its mean. The seed was fixed once at the start and reused everywhere, so
it biased every sequence-model figure in the project in the same direction. Nothing about that
was visible from any single run.

**The most practically useful result is stability, which no single-seed table can show.** The
forest's across-seed range is 0.190 against 1.7–5.7 for every neural model — 9× to 30× tighter.
For a model that gets retrained periodically, landing in the same place each time is worth a
lot, and on that axis the forest wins outright despite being ~19% behind on mean RMSE.

Updated the headline tables in `README.md`, `docs/benchmarks.md` and `docs/writeup.md`, and
added a dedicated section to the writeup's "what didn't work" — this belongs there, not in a
footnote, because it retracts published findings rather than merely qualifying them.

Remaining gap, logged rather than hidden: **the sweeps are still single-seed.** Configuration
ranking may itself be a seed artefact, which would mean the "own best config" each architecture
was given is also a favourable draw. Fixing it costs seeds × ~15 minutes per architecture.

That is the fourth claim this run that measurement removed (epoch budget, GRU-vs-LSTM, attention
advantage, CNN efficiency). Every one was a difference smaller than the spread of my own
re-runs, and every one looked convincing until it was checked.

### Was the *configuration* selection also noise? Mostly yes.

The last finding left an obvious hole: `compare.py` gave each architecture its "own best
config", but those were ranked on a single seed while the across-seed spread is 1.7–5.7. Lining
the two up was uncomfortable:

| arch | top-3 val RMSE spread | across-seed spread |
|---|---|---|
| LSTM | 1.412 | 1.723 |
| GRU | 1.101 | 1.767 |
| CNN | **0.194** | 2.235 |

Every architecture's finalists sit inside its own re-run noise; the CNN's within a tenth of it.

`src/rerank.py` tests this affordably — re-running whole grids at every seed would be 4 × 27 × 3
trainings, so it re-runs only the **finalists** (top *N* from an existing sweep, each at *M*
seeds, ranked by mean validation RMSE). Result across the three swept architectures:

| arch | single-seed pick | seed-averaged pick | changed? | test RMSE (3 seeds) |
|---|---|---|---|---|
| CNN | 20/32/3e-4 | 20/**64**/3e-4 | yes | 11.359 → 11.386 |
| LSTM | 50/128/3e-4 | 50/128/3e-4 | no | 8.302 |
| GRU | 50/128/3e-4 | 50/128/**1e-3** | yes | 8.312 → **6.820** |

**Two of three selections did not survive.** The CNN's change is immaterial — 11.359 vs 11.386
test RMSE, i.e. those configurations are genuinely equivalent, which is the honest reading when
finalists differ by 0.194 val RMSE. **The GRU's change is not immaterial:** the seed-averaged
winner is a different learning rate with a 3-seed test RMSE of **6.820** against the single-seed
pick's 8.312. So the comparison table was not merely reporting a lucky seed — it was reporting a
*worse configuration*, chosen because one seed preferred it.

The LSTM's winner held, but its finalists differ by 1.178 val RMSE against an across-seed range
of 3.422, so `rerank`'s verdict says so explicitly rather than calling it a confirmation. I wrote
three verdict branches for exactly this: selection holds *and* is separable; holds but sits
inside the noise; or does not hold.

`compare.swept_config` now prefers `rerank_<arch>.json` over `sweep_<arch>.json` and records
which it used, so the headline table is selected on seed-averaged validation. Re-running the
comparison to make that artifact canonical.

**A gap this exposed:** `attention` was never swept at all. It ran with the shared CLI defaults
(50/128/3e-4) while the other three used their own grids — `config_source` said `shared`, which
I had not noticed. Its 8.210 is therefore an *unswept* number, and given the GRU gained 1.5 RMSE
from correct selection, attention may be similarly understated. Logged rather than quietly left
in the table, and the source column now says `shared (never swept)`.

This is a cascade worth naming: the seed-bias finding invalidated two claims, then invalidated
the configuration selection those claims rested on, and that in turn changed a model's numbers by
more than the effect I originally set out to measure. Each layer looked settled until checked.

### Sweeping attention — the caveat resolves, favourably and by luck

`attention` was the one architecture never given its own grid; it ran on the shared CLI defaults
(50/128/3e-4) while the other three used swept configurations. Since correct selection had been
worth 1.49 RMSE to the GRU, I expected attention to be understated.

It wasn't. Its 27-config sweep selects **50/128/3e-4** — exactly the shared default it was already
using — and the finalist re-rank confirms that config on the seed-averaged ranking too. So the
row was fair all along, by luck rather than design. Test mean stays **8.210**.

Two things still worth recording:

**The verdict is "equivalent", not "confirmed".** Attention's finalists differ by 2.190 validation
RMSE while a single configuration varies by up to **6.131** across seeds — the widest of any
architecture. `rerank`'s middle branch fires correctly here: the winner survives on the mean, but
the ordering is inside the noise.

**That 6.131 is itself the story.** It reinforces what the variance study found: attention is much
the least stable model here. Its own best configuration swings by more than the entire gap between
the best and worst *architectures* in the comparison.

Regenerated `outputs/comparison.md` so the `config_source` column reads
`rerank_attention.json (seed-averaged)` rather than the now-false `shared (never swept)`. The
numbers are provably unchanged — same config, fixed seeds, and same-seed runs are bit-identical —
so this re-run buys artifact truthfulness rather than new information. Worth 35 minutes of wall
clock given how much of this session has been about not letting stale claims sit in generated
files.

### Stale-claim audit across the docs

Regenerating the comparison confirmed the claim I had made about it: every number identical
(8.210 / 11.386 / 6.820 / 8.302 / 10.191 and the PHM row), with only `config_source` changing
from the false `shared (never swept)` to `rerank_attention.json (seed-averaged)`. Verified rather
than asserted, since "provably unchanged" was my claim to check.

That prompted a sweep for stale statements elsewhere, and it found several — all of them written
true and left behind by later work:

- README limitations still said *"The CNN and GRU were never swept on their own
  hyperparameters"* — false since the per-arch sweeps.
- README limitations still said the ensemble *"has not been run … so 'does blending help?' is not
  yet answered"* — it was answered (no) several commits earlier. This is the worst kind of stale
  claim: a reader would conclude an open question remains where one was settled.
- `docs/writeup.md` §4 still carried the pre-rerank table (GRU 8.312, CNN 11.359) after the
  configuration correction moved them to 6.820 and 11.386.
- Both README and writeup still recommended extending the sweep past its grid boundary without
  the caveat that the across-seed spread now exceeds the gaps such a grid would rank.

Added a mechanical check rather than relying on re-reading: assert every model's mean RMSE from
`outputs/comparison.json` appears verbatim in each document that tabulates it. It caught the two
writeup rows immediately. Docs and artifacts now agree — mismatches: none.

The general lesson, and the reason this is in the log: in a session where findings were repeatedly
overturned, the *documents* drift out of date faster than the code, and nothing fails when they
do. Tests catch stale code; only a deliberate check catches stale prose.

---

## 2026-07-30 — Synthetic generator v2: the fallback now drops sensors like real FD001

`select_informative_sensors` dropped **0 of 21** sensors on the synthetic fallback. Real
FD001 drops 6 — `s1, s5, s10, s16, s18, s19` are literally constant there. The cause was
in the generator: its non-trending sensors carried `sigma=0.5` noise, variance ~0.25,
about 250x `features.variance_threshold`. So every synthetic number this project has
produced came from 21 sensors and 63 feature columns, where the real data gives 15 and 45,
and the drop path ran only inside unit-test fixtures.

v2 emits those six as exact constants (variance 0) and promotes `s6` and `s14` to the
trending set, keeping the 21-sensor shape: 6 flat + 15 trending. The generator now
recomputes the written data's per-sensor variance against the configured threshold and
exits non-zero if the flat set is not exactly what `CONSTANT` declares — a generator that
silently stops exercising the drop path is the failure this replaced, so it fails loudly
instead. `--config` is now a real flag rather than one argparse silently swallowed.

### What it cost, and how much of that is the fix

| | v1 | v2 |
|---|---|---|
| informative sensors / features | 21 / 63 | 15 / 45 |
| RandomForest RMSE / PHM | 10.309 / 85.0 | **12.513 / 115.4** |
| mean residual | +0.09 | −0.46 |

Worth measuring rather than assuming: removing the noise draws also shifts the RNG
stream, so v2 is a *different sample* of engine lifetimes and truncation points, not just
a different sensor structure. Re-running the v1 generator from git into a scratch dir
reproduced 10.309 / 85.0 exactly, and adding `sigma=0.5` back onto v2's six flat sensors
scores 12.324 / 114.7. So:

- **+2.015 RMSE (91%) is the new draw** — the same generator, resampled.
- **+0.189 RMSE (9%) is the fidelity fix** — inside the forest's own across-seed range
  (0.161 on v2 over seeds 42/43/44; 0.190 recorded on v1).

Dropping six pure-noise sensors cost nothing measurable, which is the expected result:
they carried no signal in v1 either. The headline move is a resample, not a harder
problem, and neither number says anything about real C-MAPSS.

### One qualitative finding reversed

The near-failure bin (RUL 0–25) went from **−1.53** (early — the safe direction, and the
README said so approvingly) to **+5.81** (late — the direction PHM punishes). That is a
claim about the twin's operational behaviour that a data regeneration flipped, so it was
corrected in README and benchmarks rather than left standing.

The live-twin demo also broke: on v2's draw, test engine 1 stops at cycle 71 with 97
cycles of life left and never crosses the alert threshold. Moved the demo to engine 48
(179 cycles, alerts at 163, final estimate 11.3 vs actual 11) and regenerated `demo.gif`.

### What is now inconsistent, stated plainly

Only the baseline, error analysis and live-twin artifacts were re-run. `ablation`,
`comparison`, the four sweeps, `variance`, `uncertainty`, `uncertainty_per_engine`,
`interpretability` and `ensemble` are still v1 and are **not** comparable to the baseline
above. `make check-docs` stays green on them because each doc table still matches the
artifact it quotes — the inconsistency is between artifacts, which no checker here
catches. `outputs/synthetic_fidelity.md` records the split explicitly, and the README
caveat and every affected table now say which generator produced them.

Recommendation recorded there and in TODO: do not spend sequence-model compute
re-establishing a ranking on v2 synthetic. v2 is a better *shape* match — same sensor
count, same drop count, same feature width — but the degradation is still monotonic drift
with no fault modes. Re-run everything on real FD001 when `data/CMAPSSData/` lands.

### Left open, deliberately

Two claims this change touched but could not truthfully re-measure, because a concurrent
session is editing the same tree:

- README's `check-minimal` line still quotes **242 passed / 168 passed + 2 module-skips**.
  This change adds 13 tests and the concurrent session adds more, so both counts are
  stale. The measured parts of that claim *were* re-verified: with the optional extras
  hidden, the baseline still scores RMSE 12.513 and the live twin still alerts at cycle
  163, byte-identical to the full-dependency run.
- `make check-docs` reports 22 drift findings, all in the model-comparison tables. They
  are not from this change: the other session moved `src/compare.py` and the checker's
  table cells from `±half-range` to `±95% CI` and has not yet updated the two doc tables.
  Every spec touching the baseline, error analysis, sensor count and fidelity report is
  clean.

Also worth flagging to whoever runs the reruns: `outputs/rerank_cnn.*` and
`outputs/published_comparison.*` were written *after* `data/synthetic/` was regenerated,
so they are v2 while the sibling rerank artifacts are v1.

## Re-ranking at 5 seeds, and replacing half-range with a confidence interval

Task: push `src/rerank.py` from 3 seeds to N (default 5) so the LSTM-vs-GRU ordering stops
sitting inside the noise floor.

**More seeds alone would not have worked.** Both `rerank` and `compare` reported `mean ±half-
range`, and half-range is an extremum: it describes the runs that happened and *widens* as seeds
are added. No amount of compute spent on that statistic could ever have separated two models. So
the seed count and the statistic had to change together — `src/ci.py` now computes a Student-t
interval on the **mean**, which tightens as ~1/√n (t=2.776 at df=4, against 4.303 at df=2; using
the normal 1.96 at n=5 would understate the interval by ~30%). The raw range is kept as its own
column: it still answers "how badly can one run land?", which the interval does not.

`rerank_<arch>.json` now stores **per-seed samples** (`val_by_seed`, `test_by_seed`,
`phm_by_seed`), not just summaries. The 3→5 extension had to re-run all 36 existing trainings
from scratch precisely because the old files stored only mean and range, and samples cannot be
recovered from a range. Storing them makes the artifact re-analysable without a GPU.

### What 5 seeds settled, and what it did not

| | verdict at 95% CI |
|---|---|
| CNN (9.304 ±0.916), RandomForest (12.645 ±0.105) | separated from every other model |
| GRU (2.985 ±0.941), LSTM (4.122 ±0.431), ATTENTION (4.435 ±2.568) | mutually **indistinguishable** |

So the original question — does the GRU really beat the LSTM? — is still open, and now has a
stated error bar saying so. The GRU leads on the mean by 1.137 RMSE while the two intervals
overlap. ATTENTION is the reason the top is so stubborn: its ±2.568 is six times the LSTM's
±0.431, so it cannot be separated from anything despite a clearly worse mean.

Finalist re-ranking told the same story per architecture: all four winners changed or stayed only
within overlapping intervals, and 7 of 8 non-winning finalists are tied with their winner. The
three verdict branches in `rerank` were re-cut around CI overlap rather than range comparison,
and the "separated" branch — the one that can say a selection is *real* — is now reachable at all.

**Overlap is deliberately the conservative direction.** Non-overlapping intervals do imply a
significant difference; overlapping ones do not prove equivalence (two means can overlap and
still differ at p<0.05). Both reports say so where they use the word, because "indistinguishable"
is a claim about the evidence and "equivalent" would be a claim about the models.

### Notes for whoever picks this up

- All four `rerank_*.json` are now **v2 synthetic**, computed after `data/synthetic/` was
  regenerated; the v1/v2 artifact split flagged in the previous entry is resolved for these files
  (data SHA-256 verified identical before and after the runs). `outputs/variance.json` is still
  v1.
- `make check-docs` is down to **2 findings**, both in §9 of benchmarks. They subtract a v1
  `variance.json` GRU mean from a v2 `comparison.json` LSTM mean, so the claim spans two datasets
  and patching the digits would make it numerically consistent and scientifically false. It needs
  `src/variance.py` re-run on v2 — deliberately not done here, per the standing recommendation
  not to spend sequence-model compute re-establishing rankings on v2 synthetic.
- The comparison tables and the prose claims derived from them were updated in README and
  benchmarks (76%/91% vs the forest, the 0.36 RMSE GRU selection correction, attention's 2.393
  spread, the forest's 4×–22× reliability edge). The framing changed too, not just the digits:
  "best, not decisively" became "ahead on the mean, unresolved on the evidence".
- `--seeds` is a CLI arg on both scripts, defaulting to 5. `make rerank ARCH=<a>` picks it up.
