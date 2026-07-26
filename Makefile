# twin-turbofan — common tasks.
#
# Everything runs through the project venv so results are environment-stable.
# Create it once with `make setup`, then `make all` reproduces every artifact in
# outputs/ from scratch.
#
# Every target invokes tools as `$(PY) -m <tool>` rather than `.venv/bin/<tool>`.
# That is deliberate: console scripts in .venv/bin embed an ABSOLUTE path to the
# interpreter, so moving the repo breaks all of them (`bad interpreter`) even though
# the venv itself still imports fine. Module invocation goes through `.venv/bin/python`,
# which is a symlink to the unmoved base interpreter and therefore survives a move.
# If you do relocate the repo and want the bin/ scripts back, run `make fix-venv`.

PY ?= .venv/bin/python
PYTHON_BASE ?= /Library/Frameworks/Python.framework/Versions/3.11/bin/python3

# One list, used by lint/format here AND by the CI lint job, so the two cannot drift.
# app.py and scripts/ were previously omitted from both, meaning CI would have accepted
# lint errors in them.
LINT_PATHS ?= src tests conftest.py app.py scripts

.PHONY: help setup fix-venv data test lint format typecheck check check-minimal coverage \
        check-docs check-docs-strict \
        test-all test-slow variance rerank \
        baseline error-analysis ablation uncertainty uncertainty-per-engine \
        ensemble seq sweep compare demo \
        dashboard all clean clean-outputs

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## create .venv (native arm64 py3.11) and install dependencies
	$(PYTHON_BASE) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install torch ruff black mypy pytest-cov
	@echo "note: real C-MAPSS data is NOT installed by this target — see data/README.md"

fix-venv: ## repoint .venv/bin console scripts after moving the repo
	$(PY) scripts/fix_venv.py

data: ## generate the synthetic fallback dataset
	$(PY) -m src.generate_synthetic

test: ## run the fast suite (excludes model-training tests)
	$(PY) -m pytest

test-all: ## run everything, including the slow model-training tests
	$(PY) -m pytest -m ""

test-slow: ## run ONLY the slow model-training tests
	$(PY) -m pytest -m slow

lint: ## ruff + black --check over $(LINT_PATHS)
	$(PY) -m ruff check $(LINT_PATHS)
	$(PY) -m black --check $(LINT_PATHS)

format: ## black + ruff --fix over $(LINT_PATHS)
	$(PY) -m black $(LINT_PATHS)
	$(PY) -m ruff check --fix $(LINT_PATHS)

typecheck: ## mypy on src/
	$(PY) -m mypy

# Every headline number in README.md and docs/benchmarks.md was copied by hand out of
# outputs/. Re-running a sweep changes the artifact and leaves the prose quoting the old
# value, with nothing to notice. This re-derives all of them.
#
# outputs/*.json is gitignored, so a fresh checkout has nothing to check against: the
# default run SKIPS those specs (and says how many) rather than failing. Once `make all`
# has repopulated outputs/, use check-docs-strict to make an absent artifact an error.
check-docs: ## verify README/docs numbers still match outputs/
	$(PY) -m src.validate_docs

check-docs-strict: ## same, but rounding drift and missing artifacts also fail
	$(PY) -m src.validate_docs --strict --require-artifacts

check: lint typecheck check-docs test ## lint + types + doc numbers + tests

check-minimal: ## prove the core pipeline runs with NO torch/streamlit/paho
	@echo ">> hiding optional deps (see scripts/no_optional_deps/)"
	PYTHONPATH=scripts/no_optional_deps $(PY) -m pytest
	PYTHONPATH=scripts/no_optional_deps $(PY) -m src.train_baseline >/dev/null
	PYTHONPATH=scripts/no_optional_deps $(PY) -m src.error_analysis >/dev/null
	PYTHONPATH=scripts/no_optional_deps $(PY) -m src.telemetry simulate --unit 1 --quiet
	@echo ">> core pipeline OK without the optional extras"

coverage: ## test suite with a coverage report
	$(PY) -m pytest --cov=src --cov-report=term-missing --cov-report=html

baseline: ## train the RandomForest baseline
	$(PY) -m src.train_baseline

error-analysis: ## residual + trajectory plots
	$(PY) -m src.error_analysis

ablation: ## raw vs rolling feature ablation
	$(PY) -m src.ablation

seq: ## train the LSTM twin
	$(PY) -m src.model_lstm

sweep: ## hyperparameter sweep (ARCH=lstm|gru|cnn)
	$(PY) -m src.sweep --arch $(or $(ARCH),lstm)

compare: ## model x dataset comparison table
	$(PY) -m src.compare

# Reads outputs/comparison.json rather than training anything, so it is instant and safe to
# put in `all` — but it needs `compare` to have run first, which is why it follows it there.
compare-published: ## position the best model against published C-MAPSS FD001 results
	$(PY) -m src.compare_published --model $(or $(MODEL),ATTENTION)

demo: ## render docs/demo.gif from the twin's live output
	$(PY) -m src.make_demo_gif --unit $(or $(UNIT),1)

dashboard: ## live Streamlit twin dashboard
	$(PY) -m streamlit run app.py

uncertainty: ## conformal intervals + conservatism sweep
	$(PY) -m src.uncertainty

uncertainty-per-engine: ## per-engine vs global conservatism (matched-amount comparison)
	$(PY) -m src.uncertainty_per_engine

variance: ## measure run-to-run spread at a fixed config
	$(PY) -m src.variance

rerank: ## re-rank a sweep's finalists across seeds (ARCH=lstm|gru|cnn|attention)
	$(PY) -m src.rerank --arch $(or $(ARCH),cnn)

ensemble: ## blend RF with a sequence model (ARCH=lstm|gru|cnn)
	$(PY) -m src.ensemble --arch $(or $(ARCH),gru)

# `ensemble` and `sweep` are excluded: both train sequence models and take minutes.
all: baseline error-analysis ablation uncertainty uncertainty-per-engine compare compare-published demo ## regenerate every artifact

clean-outputs: ## delete generated artifacts, keep the data
	rm -f outputs/*.png outputs/*.md outputs/*.json outputs/*.pkl outputs/*.pt

clean: clean-outputs ## also drop caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
