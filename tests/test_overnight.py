"""Tests for the unattended operator.

An overnight runner that is wrong is worse than no runner: it burns a night and leaves a
report claiming things it did not do. These cover the decisions it makes without supervision —
what to skip, what counts as fresh, and that the queue cannot grow at runtime.

Deliberately no test executes a real task; the point is the control logic.
"""

import subprocess
import sys

import pytest

from src import overnight
from src.overnight import TASKS, Task, artifact_is_fresh


@pytest.fixture
def fake_outputs(tmp_path, monkeypatch):
    """Redirect the operator's paths so nothing touches the real repo."""
    out = tmp_path / "outputs"
    out.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    stamp = data / "train_FD001.txt"
    stamp.write_text("x")
    monkeypatch.setattr(overnight, "OUTPUTS_DIR", out)
    monkeypatch.setattr(overnight, "DATA_STAMP", stamp)
    return out, stamp


def make_task(**kw) -> Task:
    base = {
        "id": "t",
        "why": "because",
        "requirement": "R2",
        "command": ["true"],
        "produces": ["thing.json"],
    }
    base.update(kw)
    return Task(**base)


class TestFreshness:
    """R6: a task whose artifact post-dates the data is already done."""

    def test_missing_artifact_is_not_fresh(self, fake_outputs):
        assert artifact_is_fresh(make_task()) is False

    def test_artifact_older_than_data_is_not_fresh(self, fake_outputs):
        out, stamp = fake_outputs
        art = out / "thing.json"
        art.write_text("{}")
        # artifact predates the dataset -> computed on superseded data (R2)
        import os

        os.utime(art, (stamp.stat().st_mtime - 100, stamp.stat().st_mtime - 100))
        assert artifact_is_fresh(make_task()) is False

    def test_artifact_newer_than_data_is_fresh(self, fake_outputs):
        out, stamp = fake_outputs
        art = out / "thing.json"
        art.write_text("{}")
        import os

        os.utime(art, (stamp.stat().st_mtime + 100, stamp.stat().st_mtime + 100))
        assert artifact_is_fresh(make_task()) is True

    def test_all_artifacts_must_be_fresh(self, fake_outputs):
        """One stale output is enough to re-run the task."""
        out, stamp = fake_outputs
        import os

        for name, offset in (("a.json", +100), ("b.json", -100)):
            p = out / name
            p.write_text("{}")
            os.utime(p, (stamp.stat().st_mtime + offset, stamp.stat().st_mtime + offset))
        assert artifact_is_fresh(make_task(produces=["a.json", "b.json"])) is False

    def test_shared_artifact_opts_out_of_freshness(self, fake_outputs):
        """`variance.json` is written by two tasks, so mtime cannot tell them apart.

        The second task sets ``fresh_against_data=False`` so sequencing decides instead —
        otherwise the first task's write would make the second look already done.
        """
        out, stamp = fake_outputs
        art = out / "thing.json"
        art.write_text("{}")
        import os

        os.utime(art, (stamp.stat().st_mtime + 100, stamp.stat().st_mtime + 100))
        assert artifact_is_fresh(make_task(fresh_against_data=False)) is False


class TestQueueIntegrity:
    """R7: the queue is a fixed literal; the operator cannot invent work."""

    def test_every_task_declares_a_requirement_and_reason(self):
        for t in TASKS:
            assert t.requirement.strip(), f"{t.id} cites no requirement"
            assert t.why.strip(), f"{t.id} gives no reason for being in the queue"

    def test_every_task_declares_what_it_produces(self):
        for t in TASKS:
            assert t.produces, f"{t.id} declares no artifact, so completion is unverifiable"

    def test_task_ids_are_unique(self):
        ids = [t.id for t in TASKS]
        assert len(ids) == len(set(ids))

    def test_commands_invoke_this_project_only(self):
        """R10: no task may shell out beyond the repo's own modules."""
        for t in TASKS:
            assert t.command[1:3] == ["-m"] or t.command[1] == "-m", t.id
            module = t.command[2]
            assert module.startswith("src."), f"{t.id} runs {module}, which is not this project"

    def test_estimates_fit_the_budget(self):
        """The queue must be plausible in 8h, else it is a plan that cannot finish."""
        total_h = sum(t.estimate_min for t in TASKS) / 60
        assert total_h < 8, f"queue estimates {total_h:.1f}h, over budget before it starts"


class TestDryRun:
    """The operator must be inspectable before it is trusted with a night."""

    def test_dry_run_executes_nothing_and_exits_zero(self):
        proc = subprocess.run(
            [sys.executable, "-m", "src.overnight", "--dry-run"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        assert "nothing executed" in proc.stdout
        # every task should be listed with its requirement
        for t in TASKS:
            assert t.id in proc.stdout

    def test_only_filter_narrows_the_queue(self):
        proc = subprocess.run(
            [sys.executable, "-m", "src.overnight", "--dry-run", "--only", "ablation"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        assert "ablation" in proc.stdout
        assert "interpret" not in proc.stdout
