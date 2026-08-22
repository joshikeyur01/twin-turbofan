"""Streamlit entry point.

    streamlit run app.py

Thin shim, and it has to be here rather than in ``src/``. ``streamlit run`` executes its
target as a top-level script and puts that script's directory on ``sys.path``. Pointing
it at ``src/dashboard.py`` would run a package module as a script, so its relative
imports would fail with "attempted relative import with no known parent package".
Running this file instead puts the repo root on the path, which makes ``src`` a proper
importable package.
"""

from src.dashboard import render

render()
