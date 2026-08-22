"""Streamlit dashboard for the live twin.

    streamlit run app.py          # from the repo root — NOT `streamlit run src/dashboard.py`

``streamlit run`` executes its target as a top-level script, so a module inside a package
has no parent package and the relative imports below raise
``ImportError: attempted relative import with no known parent package``. The root-level
``app.py`` shim exists to fix exactly that: streamlit puts *its* directory (the repo root)
on ``sys.path``, which makes ``src`` importable as a real package. Hence ``render()`` is
exported rather than executed at import time.

Replays a test engine cycle by cycle through the same ``OnlineFeatureBuilder`` the MQTT
twin uses, so what the dashboard shows is what a deployed twin would compute — not a
pre-baked offline result.

Three panels:

1. **Live RUL** — the twin's estimate against the actual, with the alert threshold.
2. **Divergence** — twin minus actual, so a developing fault shows as the twin pulling
   away from truth. Negative is conservative (the twin thinks the engine is worse than
   it is); positive means the twin is optimistic, which is the dangerous direction.
3. **Fleet view** — last known RUL for every engine, sorted by urgency, which is the
   actual maintenance-planning question.

Note the mid-life divergence you will see on most engines is largely an artefact of the
piecewise-linear RUL cap: the target is clipped at 125 while the model keeps reporting
continuous degradation, so the twin looks pessimistic long before the true countdown
starts. That is a property of the label, not a model defect.
"""

from __future__ import annotations

import pickle
import time

import pandas as pd
import streamlit as st

from .data_loader import load_cmapss
from .models import RidgeFallback  # noqa: F401  (lets a pickled fallback model reload)
from .online import FeatureSpec, OnlineFeatureBuilder
from .paths import OUTPUTS_DIR

st.set_page_config(page_title="twin-turbofan", page_icon="🛠", layout="wide")


@st.cache_resource
def load_twin():
    with open(OUTPUTS_DIR / "baseline.pkl", "rb") as f:
        model = pickle.load(f)
    return model, FeatureSpec.load()


@st.cache_data
def load_data(subset: str):
    _, test = load_cmapss(subset)
    return test


@st.cache_data(show_spinner="Scoring the fleet…")
def fleet_table(subset: str) -> pd.DataFrame:
    """Last known RUL for every engine, replayed through the online path.

    Cached because Streamlit re-runs the whole script on every widget interaction, and
    this replays each engine's entire history one cycle at a time — roughly 7,500
    single-row predictions on FD001. Uncached, moving the threshold slider would pay
    that cost again. Keyed on ``subset`` (not the model) because the model and feature
    spec are themselves cached resources loaded once per session.
    """
    model, spec = load_twin()
    test = load_data(subset)
    builder = OnlineFeatureBuilder(spec)

    rows = []
    for u, g in test.sort_values(["unit", "cycle"]).groupby("unit"):
        builder.reset(int(u))
        # Distinct from the replay loop's per-cycle `pred`: this is the engine's LAST
        # estimate after replaying its whole recorded history.
        last_pred: float | None = None
        for _, row in g.iterrows():
            d = row.to_dict()
            last_pred = builder.predict(
                model, int(u), {k: float(v) for k, v in d.items() if k.startswith("s")}
            )
        rows.append(
            {"engine": int(u), "twin RUL": last_pred, "actual RUL": float(g["RUL"].iloc[-1])}
        )

    df = pd.DataFrame(rows)
    df["error"] = df["twin RUL"] - df["actual RUL"]
    return df.sort_values("twin RUL")


def render():
    st.title("twin-turbofan — live digital twin")
    st.caption(
        "Predicted remaining useful life from streaming sensor telemetry. "
        "Features are built incrementally with training-set statistics, exactly as a "
        "deployed twin would."
    )

    try:
        model, spec = load_twin()
    except FileNotFoundError as e:
        st.error(f"{e}\n\nRun `make baseline` first.")
        return

    with st.sidebar:
        st.header("Controls")
        subset = st.selectbox("Dataset", ["FD001", "FD002", "FD003", "FD004"])
        try:
            test = load_data(subset)
        except FileNotFoundError as e:
            st.error(str(e))
            return

        units = sorted(test["unit"].unique().tolist())
        unit = st.selectbox("Engine", units)
        threshold = st.slider("Alert threshold (cycles)", 5, 60, 25)
        speed = st.select_slider(
            "Replay speed", options=["instant", "fast", "real-ish"], value="fast"
        )
        run = st.button("▶ Replay engine", type="primary")

    delay = {"instant": 0.0, "fast": 0.01, "real-ish": 0.1}[speed]
    eng = test[test["unit"] == unit].sort_values("cycle")

    col1, col2, col3, col4 = st.columns(4)
    m_cycle = col1.empty()
    m_rul = col2.empty()
    m_true = col3.empty()
    m_status = col4.empty()
    chart_slot = st.empty()
    diverge_slot = st.empty()

    if run:
        builder = OnlineFeatureBuilder(spec)
        rows = []
        first_alert = None

        for _, row in eng.iterrows():
            d = row.to_dict()
            sensors = {k: float(v) for k, v in d.items() if k.startswith("s")}
            pred = builder.predict(model, unit, sensors)
            true = float(d["RUL"])
            rows.append({"cycle": int(d["cycle"]), "twin estimate": pred, "actual": true})

            if pred < threshold and first_alert is None:
                first_alert = int(d["cycle"])

            df = pd.DataFrame(rows).set_index("cycle")
            m_cycle.metric("Cycle", int(d["cycle"]))
            m_rul.metric("Twin RUL", f"{pred:.1f}", delta=f"{pred - true:+.1f} vs actual")
            m_true.metric("Actual RUL", f"{true:.0f}")
            if first_alert:
                m_status.metric("Status", "⚠ MAINTENANCE", delta=f"since cycle {first_alert}")
            else:
                m_status.metric("Status", "✓ nominal")

            chart_slot.line_chart(df, height=320)
            diverge_slot.area_chart(
                (df["twin estimate"] - df["actual"]).rename("twin − actual"), height=180
            )
            if delay:
                time.sleep(delay)

        st.success(
            f"Replay complete — {len(rows)} cycles. "
            + (
                f"First alert at cycle {first_alert}."
                if first_alert
                else f"RUL never fell below {threshold}."
            )
        )
    else:
        st.info("Pick an engine and press **Replay engine** in the sidebar.")

    st.divider()
    st.subheader("Fleet view — last known RUL per engine")
    st.caption(
        "Scored at each engine's final recorded cycle, the operational decision point. "
        "Sorted by urgency."
    )

    fdf = fleet_table(subset)

    st.dataframe(
        fdf.style.format({"twin RUL": "{:.1f}", "actual RUL": "{:.0f}", "error": "{:+.1f}"}),
        use_container_width=True,
        hide_index=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Engines below threshold", int((fdf["twin RUL"] < threshold).sum()))
    c2.metric("Mean absolute error", f"{fdf['error'].abs().mean():.2f}")
    c3.metric(
        "Optimistic (late) predictions",
        f"{int((fdf['error'] > 0).sum())}/{len(fdf)}",
        help="Above zero means the twin over-estimates remaining life — the unsafe direction.",
    )
