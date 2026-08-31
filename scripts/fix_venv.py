"""Repoint `.venv/bin` console scripts after the repo has been moved.

    python scripts/fix_venv.py

**Why this is needed.** `pip` writes console scripts (`pytest`, `black`, `streamlit`, …)
with the interpreter's *absolute* path baked into the shebang. Move the repo and every one
of them dies with `bad interpreter: No such file or directory` — while the venv itself
still works perfectly, because `.venv/bin/python` is a symlink to the unmoved base
interpreter. So `python -m pytest` keeps working and `pytest` does not, which is a
confusing failure to debug from the error message alone.

This rewrites the stale prefix in place. It is safe to re-run, and a no-op when nothing
needs fixing.

The Makefile invokes every tool as `python -m <tool>` precisely so a move cannot break
the build in the first place; this script exists for the times you want the bare commands
back (or an activated shell).
"""

from __future__ import annotations

import pathlib
import re
import sys

BIN = pathlib.Path(".venv/bin")
# Matches a POSIX absolute path ending in /.venv/bin/python, optionally version-suffixed.
# Paths may contain spaces (this project lived under "Application Support"), so the match
# is anchored on the /.venv/bin/python tail rather than on whitespace.
STALE = re.compile(rb"/[^\x00\n]*?/\.venv/bin/python[0-9.]*")


def main() -> int:
    if not BIN.is_dir():
        print(f"no {BIN} — nothing to do (create it with `make setup`)")
        return 1

    correct_root = str(BIN.resolve().parent.parent).encode()
    fixed, checked = [], 0

    for path in sorted(BIN.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        checked += 1
        try:
            data = path.read_bytes()
        except OSError:
            continue

        def repoint(match: re.Match[bytes]) -> bytes:
            found = match.group(0)
            # Keep any pythonX.Y suffix; only the directory prefix is wrong.
            tail = found[found.rindex(b"/.venv/bin/") :]
            return correct_root + tail

        updated = STALE.sub(repoint, data)
        if updated != data:
            path.write_bytes(updated)
            fixed.append(path.name)

    if fixed:
        print(f"repointed {len(fixed)} of {checked} scripts to {correct_root.decode()}")
        print("  " + ", ".join(fixed))
    else:
        print(f"checked {checked} scripts — all already correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
