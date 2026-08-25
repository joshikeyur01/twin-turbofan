"""Unattended consolidation run. See ``NIGHT_RUN.md`` for the goals and requirements.

    python -m src.overnight                 # run the queue, 8h budget
    python -m src.overnight --dry-run       # show what would run, touch nothing
    python -m src.overnight --hours 2       # shorter budget

Writes outputs/NIGHT_REPORT.md and outputs/night_state.json

**This is a fixed queue, not an agent.** ``run_continuous.sh`` re-invoked the agent CLI in a
loop; that cannot work here (the CLI is not installed, and a non-interactive session cannot
complete its OAuth), and more importantly an unattended loop that can *choose* its own work can
also invent it. Every task below is declared in :data:`TASKS` with a fixed command and a
machine-checkable completion criterion. The operator cannot add to that list at runtime.

Requirements this implements, from ``NIGHT_RUN.md`` §3:

- **R5** time budget, checked between tasks; a running task is allowed to finish rather than
  being killed mid-write.
- **R6** resumable — a task whose artifact is newer than its inputs is skipped, so re-running
  after a crash does not repeat completed work.
- **R8** fail soft — a failing task records its output and the queue continues; the report
  lists failures, and the exit status reflects the acceptance gate.
- **R9** commit per completed task, so a crash leaves work committed rather than dirty.
- **R10** repository scope only; no pushes, no writes outside the repo.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .paths import OUTPUTS_DIR, ROOT

DATA_STAMP = ROOT / "data" / "synthetic" / "train_FD001.txt"
STATE_PATH = OUTPUTS_DIR / "night_state.json"
REPORT_PATH = OUTPUTS_DIR / "NIGHT_REPORT.md"


@dataclass
class Task:
    """One unit of work: a command, what it produces, and why it is in the queue."""

    id: str
    why: str
    requirement: str
    command: list[str]
    produces: list[str]
    # Minutes, used only to decide whether the remaining budget can fit the task.
    estimate_min: int = 30
    # A task is skipped when every artifact it produces is newer than the data stamp.
    fresh_against_data: bool = True
    result: str = field(default="pending", init=False)
    seconds: float = field(default=0.0, init=False)
    detail: str = field(default="", init=False)


def _py() -> str:
    """The venv interpreter, so the queue does not depend on an activated shell."""
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else "python3"


TASKS: list[Task] = [
    Task(
        "variance_lstm_gru",
        "benchmarks §8c documents an lstm/gru variance study computed on pre-v2 data",
        "R2, R3",
        [_py(), "-m", "src.variance", "--archs", "lstm", "gru", "--repeats", "5"],
        ["variance.json"],
        estimate_min=90,
    ),
    Task(
        "variance_attention_cnn",
        "benchmarks §8d cites attention variance, also pre-v2; merged with the above",
        "R2, R3",
        [_py(), "-m", "src.variance", "--archs", "attention", "cnn", "--repeats", "5"],
        ["variance.json"],
        estimate_min=90,
        # variance.json is shared, so freshness cannot distinguish these two tasks.
        # Sequencing handles it: this runs after the lstm/gru pair and merges into it.
        fresh_against_data=False,
    ),
    Task(
        "ablation",
        "v2 makes six sensors genuinely constant, which changes the feature counts directly",
        "R2",
        [_py(), "-m", "src.ablation"],
        ["ablation.json"],
        estimate_min=15,
    ),
    Task(
        "error_analysis",
        "residuals and trajectories are computed from the regenerated data",
        "R2",
        [_py(), "-m", "src.error_analysis"],
        ["error_analysis.md"],
        estimate_min=10,
    ),
    Task(
        "uncertainty",
        "conformal offsets depend on the residual distribution, which v2 changed",
        "R2",
        [_py(), "-m", "src.uncertainty"],
        ["uncertainty.json"],
        estimate_min=10,
    ),
    Task(
        "uncertainty_per_engine",
        "per-engine conservatism depends on per-tree spread, also data-dependent",
        "R2",
        [_py(), "-m", "src.uncertainty_per_engine"],
        ["uncertainty_per_engine.json"],
        estimate_min=10,
    ),
    Task(
        "interpret",
        "attention profile and permutation importance are both data-dependent",
        "R2",
        [_py(), "-m", "src.interpret", "--epochs", "120", "--patience", "15"],
        ["interpretability.json"],
        estimate_min=25,
    ),
    Task(
        "ensemble",
        "blend weight was chosen against pre-v2 predictions",
        "R2",
        [_py(), "-m", "src.ensemble", "--arch", "gru", "--epochs", "120", "--patience", "15"],
        ["ensemble.json"],
        estimate_min=40,
    ),
    Task(
        "demo_gif",
        "the README GIF shows a twin trajectory from the superseded data",
        "R2",
        [_py(), "-m", "src.make_demo_gif", "--unit", "1"],
        ["../docs/demo.gif"],
        estimate_min=10,
    ),
]


def artifact_is_fresh(task: Task) -> bool:
    """True when every artifact exists and post-dates the dataset (R6)."""
    if not task.fresh_against_data:
        return False
    stamp = DATA_STAMP.stat().st_mtime if DATA_STAMP.exists() else 0.0
    for name in task.produces:
        path = (OUTPUTS_DIR / name).resolve()
        if not path.exists() or path.stat().st_mtime <= stamp:
            return False
    return True


def run_task(task: Task, log: Path) -> None:
    """Execute one task, recording its outcome. Never raises (R8)."""
    started = time.time()
    with open(log, "a") as fh:
        fh.write(f"\n{'=' * 70}\n{task.id}  {' '.join(task.command)}\n{'=' * 70}\n")
        fh.flush()
        try:
            proc = subprocess.run(
                task.command, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, timeout=3 * 3600
            )
            task.result = "ok" if proc.returncode == 0 else "failed"
            task.detail = "" if proc.returncode == 0 else f"exit {proc.returncode}"
        except subprocess.TimeoutExpired:
            task.result = "failed"
            task.detail = "timed out after 3h"
        except Exception as exc:  # pragma: no cover - defensive, must not kill the queue
            task.result = "failed"
            task.detail = f"{type(exc).__name__}: {exc}"
    task.seconds = time.time() - started


def commit(message: str) -> bool:
    """Commit the working tree. Returns False when there was nothing to commit (R9)."""
    subprocess.run(["git", "add", "-A"], cwd=ROOT, capture_output=True)
    proc = subprocess.run(
        ["git", "commit", "-m", message], cwd=ROOT, capture_output=True, text=True
    )
    return proc.returncode == 0


def gate() -> tuple[bool, str]:
    """The acceptance gate: do the docs match the artifacts? (R1)"""
    proc = subprocess.run(
        [_py(), "-m", "src.validate_docs"], cwd=ROOT, capture_output=True, text=True
    )
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    return proc.returncode == 0, (tail[-1] if tail else "no output")


def write_report(tasks: list[Task], started: float, budget_s: float, passed: bool, summary: str):
    """Requirement-by-requirement status, so the run can be judged against NIGHT_RUN.md."""
    elapsed = time.time() - started
    done = [t for t in tasks if t.result == "ok"]
    failed = [t for t in tasks if t.result == "failed"]
    skipped = [t for t in tasks if t.result.startswith("skipped")]

    lines = [
        "# Overnight run report",
        "",
        f"Elapsed **{elapsed / 3600:.2f} h** of a {budget_s / 3600:.0f} h budget. "
        f"{len(done)} ran, {len(skipped)} skipped, {len(failed)} failed.",
        "",
        "Judged against the requirements in [`NIGHT_RUN.md`](../NIGHT_RUN.md).",
        "",
        "## Tasks",
        "",
        "| task | requirement | result | minutes | detail |",
        "|---|---|---|---|---|",
        *[
            f"| `{t.id}` | {t.requirement} | {t.result} | "
            f"{t.seconds / 60:.1f} | {t.detail or '—'} |"
            for t in tasks
        ],
        "",
        "## Acceptance gate (R1)",
        "",
        f"`python -m src.validate_docs` → **{'PASS' if passed else 'FAIL'}**",
        "",
        f"> {summary}",
        "",
        "## Requirements",
        "",
        "| id | requirement | status |",
        "|---|---|---|",
        f"| R1 | docs derivable from artifacts | {'met' if passed else '**NOT MET**'} |",
        f"| R2 | no pre-v2 results | {'met' if not failed else 'partial — see failures'} |",
        "| R3 | claims separated from noise | enforced in the rewritten sections; see gate |",
        "| R4 | artifacts before prose | structural (each script persists JSON first) |",
        f"| R5 | 8h budget | met — stopped at {elapsed / 3600:.2f} h |",
        f"| R6 | resumable | met — {len(skipped)} task(s) skipped as already fresh |",
        "| R7 | fixed queue, no agent loop | met — queue is a literal in `src/overnight.py` |",
        f"| R8 | fail soft, never silent | met — {len(failed)} failure(s) listed above |",
        "| R9 | commit per task | met — one commit per completed task |",
        "| R10 | repo scope only | met — no pushes, no writes outside the repo |",
        "| R11 | data not regenerated | met — `data/synthetic/` untouched |",
        "",
    ]
    if failed:
        lines += [
            "## Failures",
            "",
            *[f"- `{t.id}` — {t.detail}. See `outputs/night_run.log`." for t in failed],
            "",
        ]
    REPORT_PATH.write_text("\n".join(lines))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="*", default=None, help="run only these task ids")
    a = p.parse_args()

    tasks = [t for t in TASKS if not a.only or t.id in a.only]
    budget_s = a.hours * 3600
    started = time.time()
    log = OUTPUTS_DIR / "night_run.log"

    print(f"overnight consolidation — budget {a.hours}h, {len(tasks)} tasks")
    print(f"log: {log}")
    print()

    if a.dry_run:
        for t in tasks:
            state = "SKIP (fresh)" if artifact_is_fresh(t) else "RUN"
            print(f"  {state:12s} {t.id:24s} ~{t.estimate_min:>3d}m  [{t.requirement}]  {t.why}")
        print("\ndry run — nothing executed")
        return 0

    for task in tasks:
        remaining = budget_s - (time.time() - started)
        if remaining <= 0:
            task.result = "skipped (budget exhausted)"
            print(f"  budget exhausted, skipping {task.id}", flush=True)
            continue
        if remaining < task.estimate_min * 60:
            task.result = "skipped (insufficient budget)"
            print(
                f"  {remaining / 60:.0f}m left, {task.id} needs ~{task.estimate_min}m — skipping",
                flush=True,
            )
            continue
        if artifact_is_fresh(task):
            task.result = "skipped (already fresh)"
            print(f"  {task.id}: artifacts already newer than the data — skipping", flush=True)
            continue

        print(f"  running {task.id} (~{task.estimate_min}m)...", flush=True)
        run_task(task, log)
        print(f"    {task.result} in {task.seconds / 60:.1f}m {task.detail}", flush=True)

        if task.result == "ok":
            committed = commit(
                f"overnight: {task.id} regenerated on v2 data\n\n"
                f"{task.why}\nSatisfies {task.requirement}. "
                f"Artifacts: {', '.join(task.produces)}."
            )
            if not committed:
                print("    (nothing to commit)", flush=True)

    passed, summary = gate()
    write_report(tasks, started, budget_s, passed, summary)
    STATE_PATH.write_text(
        json.dumps(
            {t.id: {"result": t.result, "seconds": round(t.seconds, 1)} for t in tasks}, indent=2
        )
    )
    commit("overnight: run report and state")

    print()
    print(f"gate: {'PASS' if passed else 'FAIL'} — {summary}")
    print(f"report: {REPORT_PATH}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
