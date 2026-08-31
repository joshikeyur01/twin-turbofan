"""Hide the optional dependencies, to test the dependency-light guarantee locally.

    make check-minimal

The README claims the core pipeline runs with only numpy/pandas/matplotlib(/sklearn), and
that the optional extras sit behind clean imports whose tests skip themselves. CI has a
torch-free job to prove it, but CI only runs on push — this makes the same check runnable
in one command against the venv you already have, with no second environment to build.

Python imports ``sitecustomize`` automatically at startup if it is on ``sys.path``, so
putting this directory on ``PYTHONPATH`` is enough; no test code knows it exists.

**Why a meta_path finder rather than a stub module.** A stub ``torch.py`` that raises
``ImportError`` is *not* a faithful simulation: the module then exists but fails internally,
and `pytest.importorskip` deliberately re-raises that case rather than skipping, so it
reports collection errors instead of skips. A genuinely uninstalled package raises
``ModuleNotFoundError``, which is what this reproduces. (Found the hard way — the stub
version reported two collection errors and looked like a real bug in the guarantee.)
"""

import sys

BLOCKED = ("torch", "streamlit", "paho")


class _Blocker:
    """Raise ModuleNotFoundError for the optional extras, as if never installed."""

    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


sys.meta_path.insert(0, _Blocker())
