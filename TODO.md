# twin-turbofan — Backlog

Mirrored from `CONTINUE.md`. Tick items as they land; extend as new work appears.
Status keys: `[ ]` todo · `[x]` done · `[~]` partial · `[!]` blocked (reason in `worklog.md`).

## Setup
- [x] `pip install -r requirements.txt` — deps already present in the active env; see worklog for version deltas.
- [x] Confirm RandomForest baseline active (`outputs/metrics.json` says `RandomForestRegressor`).
- [!] NASA C-MAPSS data in `data/CMAPSSData/` — absent. Running on `data/synthetic/`.
      Needs a download decision; **all metrics below are plumbing-only until this lands.**
- [~] Record the FD001 baseline as the number to beat — recorded, but on synthetic data.

## Tier 1 — Finish Week 1
- [~] RandomForest baseline on FD001; record metrics. *(done on synthetic, not real data)*
- [x] `src/error_analysis.py`: per-engine predicted-vs-true RUL trajectories + residual-vs-true-RUL plot.
      → `outputs/trajectories.png`, `outputs/residuals.png`, `outputs/error_analysis.md`.
      Finding: error is tail-driven, not a systematic late bias. See worklog.
- [x] pytest suite: RUL labelling correctness; `phm_score` asymmetry; features have no NaNs and expected columns.
      → 49 tests, green, mutation-tested (6/6 breaks caught).

## Tier 2 — Modeling depth
- [x] Finish `src/model_lstm.py` into a trainable + evaluable model (by-engine split, last-cycle eval).
      → now a thin entry point over `src/train_seq.py`; evaluates instead of stopping
      after the final epoch.
- [x] 1D-CNN and GRU variants sharing one harness (`src/train_seq.py`).
- [x] Hyperparameter sweep (sequence length, hidden size, learning rate).
      → `outputs/sweep_lstm.md`. Tuned LSTM (50/128/3e-4) hits RMSE 7.804 / PHM 48.3,
      beating RF by 24% / 43%. The scaffold defaults were simply under-capacity — my
      earlier "sequence models lose to RF here" note was wrong and is corrected in the
      worklog.
- [~] All models × FD001–FD004 → `outputs/comparison.md`. Harness done
      (`src/compare.py`, auto-detects available subsets); **only FD001 exists until the
      real data lands**, so this is one column of four.
- [x] Feature ablation: raw sensors vs + rolling features. → `src/ablation.py`,
      `outputs/ablation.md`. Rolling buys only ~3% RMSE on synthetic; `w=3` beats the
      `w=5` default; long windows hurt badly. **Re-run on real data before concluding.**
- [x] Ensemble (RF + best sequence model); report whether it helps. → `src/ensemble.py`,
      `outputs/ensemble.md`. **Answer: no.** Validation prefers more GRU monotonically and
      selects w=1.0 (sequence model alone). Test would have picked w=0.8 (7.132/42.3, beating
      the GRU) — so choosing on test would have manufactured a 2%/6% "win". The val/test
      disagreement is itself the evidence that gain is selection noise.
- [x] Sweep the CNN and GRU separately. → `outputs/sweep_gru.md`, `outputs/sweep_cnn.md`.
      GRU picks the same config as the LSTM; the CNN picks the **opposite corner**
      (seq=20/hidden=32) and reaches 10.236/70.6 with only **10,401 params** — matching the
      RandomForest on RMSE and beating it 17% on PHM. `compare.py` now reads each arch's own
      sweep.
- [x] **Run-to-run variance study** (added during the run) → `src/variance.py`,
      `outputs/variance.md`. Same-seed repeats are bit-identical; across seeds the LSTM/GRU
      gap (0.010) is far below the spread (1.767), so **that ranking is noise**. Seed 42 is
      also a lucky draw — both models beat their seed-average by 0.5–1.0 RMSE.
- [x] **Report seed-averaged sequence-model numbers**, not single-seed. `compare.py` takes
      `--seeds N` (default 3), reports `mean ±half-range`, and `make_model(seed)` lets the
      RandomForest average on the same footing. Headline tables in README/benchmarks/writeup
      regenerated. **It removed two conclusions I had already written up:** the GRU-beats-LSTM
      ranking (means differ by 0.010 against ranges of ~1.7) and the CNN "efficiency win" (it is
      11% worse than the forest on RMSE averaged, not level). Seed 42 flattered every neural
      model by 0.5–2.0 RMSE and left the forest alone.
- [x] Seed-average the **sweeps** too. → `src/rerank.py` re-runs a sweep's *finalists* across
      seeds instead of whole grids (4×27×3 would be hours). **Two of three selections did not
      survive:** the CNN's changed but immaterially (11.359 → 11.386 test RMSE, i.e. equivalent
      configs), the GRU's changed materially (8.312 → **6.820**, a different learning rate).
      The LSTM's held but sits inside the noise. `compare.swept_config` now prefers
      `rerank_<arch>.json` and records which source it used.
- [x] **Sweep `attention`.** → `outputs/sweep_attention.md`, `outputs/rerank_attention.md`. Its
      swept optimum is **50/128/3e-4 — the same as the shared default it was already using**, so
      that row was fair by luck and its 8.210 stands. The re-rank confirms the config on the
      seed-averaged ranking, but reports "equivalent" not "confirmed": finalists differ by 2.190
      val RMSE while one config varies by **6.131** across seeds, the widest of any architecture —
      reinforcing that attention is the least stable model here.
- [x] Re-rank with more than 3 seeds / more than 3 finalists — with ranges of 1.2–6.1 on three
      seeds, even the seed-averaged ordering is not comfortably separated. Roughly linear cost.
      **Done (5 seeds, 3 finalists, all 4 archs).** The fix was not only more seeds but a better
      statistic: `±half-range` → `±95% CI` on the mean (`src/ci.py`), because a range cannot
      tighten with n. Outcome: the *bottom* of the comparison separates (CNN, RandomForest), the
      *top* three (GRU/LSTM/ATTENTION) remain indistinguishable at 95% CI. Still open below.
- [ ] Separate GRU vs LSTM vs ATTENTION — 5 seeds was not enough; all three intervals overlap.
      Since the CI tightens as ~1/√n, the next informative step is ~10–20 seeds, not 7. Worth
      doing only on real FD001: re-establishing this ranking on v2 synthetic is not a result.
- [ ] Automate the doc/artifact consistency check (currently a one-off script): assert every
      headline number in README/benchmarks/writeup still matches `outputs/*.json`. Stale prose is
      the one failure mode the test suite cannot catch, and this session produced four instances.

## Tier 3 — The live twin system
- [x] Online feature path (`src/online.py`) — **not in the original backlog, but the live
      twin was broken without it.** `stream_demo` computed features by running `build_xy`
      over the whole test set offline, so it depended on future data. `FeatureSpec` now
      persists the fitted contract at training time and `OnlineFeatureBuilder` builds
      features incrementally from training statistics. Tested to reproduce the batch path
      to 1e-9.
- [x] MQTT telemetry bus (`paho-mqtt`): publisher replays cycles, twin predicts live.
      → `src/telemetry.py`, with a broker-free `simulate` mode so the demo and its tests
      need no infrastructure. Verified end-to-end: engine 1 alerts at cycle 213.
- [x] Streamlit dashboard: live RUL, alert threshold, twin-vs-actual divergence.
      → `src/dashboard.py` + root `app.py`. **Verified running in a browser**: replayed
      engine 1 to cycle 222, alert raised at cycle 213 — identical to the CLI `simulate`
      run, confirming dashboard and MQTT twin share one online path. Fleet view sorted by
      urgency agrees with the replay (engine 1 → 16.8).
      Two bugs found and fixed by actually running it: (1) `streamlit run src/dashboard.py`
      cannot work — it executes a package module as a script, so relative imports raise
      `ImportError`; hence the root `app.py` shim. (2) the fleet table recomputed ~7,500
      single-row predictions on *every* widget interaction; now `@st.cache_data`.
- [x] Demo GIF into `docs/`. → `src/make_demo_gif.py` renders `docs/demo.gif` (112 frames,
      593 KB) from the twin's real output rather than screen-recording, so it stays in sync
      with the model and needs no capture tooling. Verified frame-by-frame.

## Tier 4 — Engineering hardening
- [x] Config via YAML + argparse; centralised logging; global seed control.
      → `config.yaml` + `src/config.py`. Layered defaults < config.yaml < `TWIN_CONFIG` <
      CLI, and **unknown keys raise** — silently ignoring `rul_capp: 100` would leave the
      real cap in place while the run looked correct. PyYAML is a soft dependency (falls
      back to dataclass defaults). Wired into `train_baseline` and `telemetry`; verified
      by observation, not assertion (`rul_cap: 60` moves RMSE 10.309 → 4.276 and the
      feature spec records the window actually used). 21 tests incl. a drift guard so
      `config.yaml` cannot silently contradict the code defaults.
- [x] `ruff` + `black` clean; type hints; `mypy` passing on `src/`. → all three green.
      mypy caught a real design flaw (heterogeneous `best` dict in `train()`), not just nits.
- [x] `Makefile`, `Dockerfile`, GitHub Actions CI (tests + lint).
      CI runs a deliberate no-torch job, which is what actually proves the
      dependency-light claim instead of asserting it.
- [x] Coverage report on data/feature/eval code. → **78%** overall (was 25%), via
      `tests/test_pipeline_smoke.py` and `tests/test_seq_pipeline_smoke.py`.
      Core library modules 91-100%; the report/entry-point scripts were all at 0%, which
      meant a crash in `make all` would only surface when a human ran it.
      Two deliberate gaps: `dashboard.py` (0% — needs a streamlit runtime; verified
      manually in a browser instead) and `telemetry.py` (39% — MQTT publish/subscribe
      needs a broker; the broker-free `simulate` path that shares all scoring logic *is*
      covered).
      **These smoke tests found a real bug:** `FeatureSpec.save()` bound its output path
      as a default argument, evaluated once at import, so redirecting `OUTPUTS_DIR` was
      silently ignored and the spec was written to the real `outputs/` regardless.

## Tier 5 — Docs & portfolio polish
- [x] README rewritten: results tables, figures, GIF, twin-loop diagram, a "design
      decisions" section, and an explicit limitations section. Synthetic-data caveat is
      the first thing a reader sees.
- [x] `docs/writeup.md`: technical article — problem framing, protocol, what worked,
      **what didn't** (the conformal hypothesis, the inverted sign convention), and
      honest limitations.
- [x] `docs/benchmarks.md`: every table with the command that regenerates it.
- [~] Docstrings across modules — every module and public function has one; a few small
      private helpers don't.

## Tier 6 — Research stretch
- [x] Uncertainty: quantile or conformal prediction → RUL intervals. → `src/uncertainty.py`,
      `outputs/uncertainty.md`. Split conformal, 30 calibration engines.
      **Correction to my earlier claim here:** I promoted this as the highest-value PHM
      work on the theory that a conservative shift would exploit the late-error tail.
      Measured, it recovers only 5.6% of PHM (best q=0.70, +1.86 cycles) for 1.3% RMSE —
      real and favourable, but small. Tail concentration does not imply a uniform offset
      helps. Intervals over-cover slightly (80→84%, 90→92%, 95→98%).
- [x] Follow-up from the above: **per-engine** conservatism scaled to each engine's own
      predictive uncertainty. → `src/uncertainty_per_engine.py`,
      `outputs/uncertainty_per_engine.md`. **This one worked.** At the
      calibration-selected `k=0.25`: PHM **74.4** vs 82.1 unadjusted (**−9.4%**), against
      the 5.6% the best global offset managed — and 2.8 PHM better than a uniform shift of
      the *same* 2.2-cycle mean amount, so the gain is from allocation, not from being
      more conservative. Precondition measured, not assumed: `corr(sigma, |error|) = +0.546`.
      Gain confined to mild `k`; at `k >= 1` it is worse than uniform because sigma reaches
      ~16 cycles and over-correction triggers the early-penalty term.
- [!] Extend the LSTM sweep past its grid boundary. **Deprioritised on evidence:** the
      variance study shows the across-seed spread (1.7 for the GRU, 5.7 for attention) exceeds
      the gaps a wider single-seed grid would be ranking. Extending it would produce
      differences below the noise floor. Worth doing only alongside seed-averaged sweeping,
      which multiplies an already ~15-minute sweep by the seed count.
- [ ] Cross-condition generalisation (train FD001 → test FD002/FD004).
- [x] Attention model + interpretability look at driving cycles/sensors.
      → `AttentionRegressor` in `src/seq_models.py` (GRU encoder + additive attention
      pooling) and `src/interpret.py`. Additive pooling chosen over multi-head self-attention
      deliberately: it yields exactly one weight per cycle summing to 1, and the prediction
      *is* that weighted sum, so "which cycles drove this?" is answerable rather than
      suggestive. Interpretability pairs that with model-agnostic permutation importance over
      sensors, since attention can be high on a cycle whose sensors carry no signal.
      Contract verified (output shape, weights sum to 1, varying seq_len) and the permutation
      shuffle unit-checked. **Results:** seed-42 RMSE 6.248 / PHM 31.0 — best in the project —
      but a variance check shows the means tie with the GRU (8.210 vs 8.312) and attention is
      **3.2x less stable** across seeds (spread 5.733 vs 1.767). The GRU stays the better
      engineering choice; attention earns its place for interpretability. Attention puts 62.1%
      of its weight on the most recent 13 of 50 cycles (uniform 26%); permutation importance
      ranks raw sensors above engineered ones.
- [ ] Compare best numbers to published C-MAPSS results; note the gap honestly.

## Added during the run
- [x] Persist expensive results **before** rendering figures. Applied to `interpret`,
      `variance`, `ensemble`, `ablation` and `uncertainty_per_engine`. Verified by injecting a
      rendering fault into `uncertainty_per_engine` and confirming the JSON survived the crash
      with all 8 rows intact — rather than assuming the reordering worked.
- [x] `requirements.txt` floors exceeded the installed env → resolved by building a
      native arm64 `.venv` (py3.11) that satisfies all floors. Suite green on
      numpy 2.4 / pandas 3.0 / sklearn 1.9.
- [x] `generate_synthetic.py` fidelity: non-trending sensors got σ=0.5 noise, so
      `select_informative_sensors` dropped 0/21 sensors — real FD001 drops 6. Generator v2
      emits `s1, s5, s10, s16, s18, s19` as exact constants; the fallback now drops 6/21
      and builds 45 features instead of 63. Baseline moved 10.309 → 12.513 RMSE / 85.0 →
      115.4 PHM, of which only 0.189 RMSE is the fix and 2.015 is the new random draw.
      Baseline, error analysis and the live-twin demo were re-run; **everything else in
      `outputs/` is still v1 and not comparable** — see `outputs/synthetic_fidelity.md`.
- [ ] Re-run ablation, comparison, sweeps, rerank, variance, uncertainty and ensemble on
      v2 synthetic data — or skip straight to real FD001, which supersedes all of it.
- [ ] Re-run baseline + error analysis + ablation on real FD001 once
      `data/CMAPSSData/` is populated; treat all current metrics as plumbing-only.
- [ ] Consider `w=3` as the rolling default — only after confirming on real data.
- [x] `run_continuous.sh` hardening. Three silent-failure modes fixed, each verified:
      missing the agent CLI on PATH (was ~4,700 no-op iterations exiting 0), committing from a
      repo root that isn't this directory (tested both ways with a stub CLI), and
      hot-looping on instant failures (now counted, backed off, aborts after 3). Added
      `--dry-run`. Also fixed a git fatal printed on a repo with no commits — the exact
      state the repo-root guard tells you to create.
- [ ] Stray git repo at `/Users/keyur` (0 commits, tracks home-dir dotfiles) — not
      touched, flagged for the user.
- [x] Repo relocated (2026-07-30) out of the session-scoped agent outputs dir to
      `~/Documents/Side Projects/twin-turbofan`, alongside the rest of the twin-* family.
      All 9 commits intact. The move broke all 35 `.venv/bin` console scripts (pip bakes
      an absolute interpreter path into each shebang) while the venv itself still imported
      fine — a confusing failure mode, since `python -m pytest` worked and `pytest` did
      not. Fixed by `scripts/fix_venv.py`; the Makefile now invokes every tool as
      `$(PY) -m <tool>` so a future move cannot break the build at all.
