"""MQTT telemetry bus for the live twin.

Three entry points:

    # 1. broker-free: publisher and twin in one process (no infrastructure needed)
    python -m src.telemetry simulate --unit 1

    # 2. real bus, two terminals (needs a broker, e.g. `brew install mosquitto`)
    python -m src.telemetry twin
    python -m src.telemetry publish --unit 1 --delay 0.05

``simulate`` exists because the interesting logic — incremental features, prediction,
alerting, divergence tracking — is identical either way, and requiring a broker just to
see the twin work makes the demo fragile. It also means the divergence figure and the
test suite need no network.

Predictions use ``online.OnlineFeatureBuilder``, so the twin consumes one reading at a
time with the *training* standardisation statistics, exactly as a deployed system would.

**Alerting.** The twin raises an alert when predicted RUL crosses below a threshold
(default 25 cycles). Because the PHM metric punishes late predictions, the operationally
honest threshold is one that fires early; the error analysis showed this model is
slightly early near failure, which is the safe direction.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np

from .config import load_config
from .data_loader import load_cmapss
from .models import RidgeFallback  # noqa: F401  (lets a pickled fallback model reload)
from .online import FeatureSpec, OnlineFeatureBuilder
from .paths import OUTPUTS_DIR

# Defaults come from config.yaml (see src/config.py) so the alert threshold and broker
# address are declared with the rest of the protocol rather than buried here. These module
# constants remain the fallback when PyYAML is absent, and are still importable by tests.
_CFG = load_config()
DEFAULT_TOPIC = _CFG.mqtt_topic
DEFAULT_HOST = _CFG.mqtt_host
DEFAULT_PORT = _CFG.mqtt_port
DEFAULT_THRESHOLD = _CFG.alert_threshold


def _require_paho():
    try:
        import paho.mqtt.client as mqtt
    except ImportError as e:  # pragma: no cover - exercised only without paho
        raise SystemExit(
            "MQTT mode needs paho-mqtt:  pip install paho-mqtt\n"
            "Or run without a broker:  python -m src.telemetry simulate --unit 1"
        ) from e
    return mqtt


def _make_client(mqtt, client_id: str):
    """paho 2.x requires an explicit callback API version; 1.x does not accept it."""
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except AttributeError:  # paho-mqtt 1.x
        return mqtt.Client(client_id=client_id)


def load_twin():
    """Load the trained model and its feature contract."""
    with open(OUTPUTS_DIR / "baseline.pkl", "rb") as f:
        model = pickle.load(f)
    return model, OnlineFeatureBuilder(FeatureSpec.load())


def engine_stream(unit: int, subset: str = "FD001"):
    """Yield raw per-cycle payloads for one test engine, oldest first."""
    _, test = load_cmapss(subset)
    eng = test[test["unit"] == unit].sort_values("cycle")
    if eng.empty:
        raise SystemExit(f"engine {unit} not found in {subset} test set")
    for _, row in eng.iterrows():
        d = row.to_dict()
        yield {
            "unit": int(d["unit"]),
            "cycle": int(d["cycle"]),
            "rul_true": float(d["RUL"]),
            "sensors": {k: float(v) for k, v in d.items() if k.startswith("s")},
        }


def handle_reading(builder, model, payload, threshold=DEFAULT_THRESHOLD, verbose=True):
    """Score one reading. Returns (cycle, predicted, true, alert)."""
    pred = builder.predict(model, payload["unit"], payload["sensors"])
    true = payload.get("rul_true")
    alert = pred < threshold

    if verbose:
        bar = "#" * int(min(pred, 125) / 5)
        flag = "  ** ALERT: schedule maintenance **" if alert else ""
        delta = f" (true {true:5.1f}, err {pred - true:+6.1f})" if true is not None else ""
        print(f"  cycle {payload['cycle']:3d} | RUL ~{pred:6.1f}{delta}  {bar}{flag}")

    return payload["cycle"], pred, true, alert


def cmd_simulate(a):
    """Publisher and twin in one process — no broker required."""
    model, builder = load_twin()
    print(f"[twin-turbofan] simulating engine {a.unit} (no broker)")

    cycles, preds, trues = [], [], []
    first_alert = None
    for payload in engine_stream(a.unit, a.subset):
        c, p, t, alert = handle_reading(builder, model, payload, a.threshold, not a.quiet)
        cycles.append(c)
        preds.append(p)
        trues.append(t)
        if alert and first_alert is None:
            first_alert = c
        if a.delay:
            time.sleep(a.delay)

    if first_alert:
        print(f"\nfirst alert at cycle {first_alert} (threshold {a.threshold:g})")
    else:
        print(f"\nno alert raised (RUL never fell below {a.threshold:g})")

    if a.plot:
        path = plot_divergence(cycles, preds, trues, a.unit, a.threshold)
        print(f"Saved -> {path}")


def plot_divergence(cycles, preds, trues, unit, threshold=DEFAULT_THRESHOLD):
    """Twin-vs-actual divergence as the fault develops."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preds_a = np.asarray(preds, dtype=float)
    trues_a = np.asarray([np.nan if t is None else t for t in trues], dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(cycles, trues_a, "k-", lw=2, label="actual RUL")
    ax1.plot(cycles, preds_a, "-", color="tab:orange", lw=1.8, label="twin estimate")
    ax1.axhline(threshold, color="tab:red", ls="--", lw=1.2, label=f"alert at {threshold:g}")
    below = np.where(preds_a < threshold)[0]
    if len(below):
        ax1.axvline(cycles[below[0]], color="tab:red", alpha=0.4, lw=1)
        ax1.annotate(
            f"alert @ cycle {cycles[below[0]]}",
            xy=(cycles[below[0]], threshold),
            xytext=(8, 24),
            textcoords="offset points",
            fontsize=8,
            color="tab:red",
        )
    ax1.set_ylabel("RUL (cycles)")
    ax1.set_title(f"twin-turbofan — live twin vs actual, engine {unit}")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.axhline(0, color="k", ls="--", lw=1)
    ax2.fill_between(cycles, preds_a - trues_a, 0, color="tab:orange", alpha=0.6)
    ax2.set_xlabel("cycle")
    ax2.set_ylabel("divergence")
    ax2.set_title("twin − actual  (above zero = twin is optimistic = late)", fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = OUTPUTS_DIR / f"live_twin_engine{unit}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def cmd_publish(a):
    """Replay one engine's cycles onto the MQTT bus."""
    mqtt = _require_paho()
    client = _make_client(mqtt, "twin-turbofan-publisher")
    client.connect(a.host, a.port, keepalive=60)
    client.loop_start()

    n = 0
    for payload in engine_stream(a.unit, a.subset):
        client.publish(a.topic, json.dumps(payload), qos=1)
        n += 1
        if not a.quiet:
            print(f"  published cycle {payload['cycle']}")
        if a.delay:
            time.sleep(a.delay)

    client.publish(a.topic, json.dumps({"event": "end_of_stream", "unit": a.unit}), qos=1)
    time.sleep(0.3)  # let QoS-1 delivery drain before tearing the loop down
    client.loop_stop()
    client.disconnect()
    print(f"published {n} cycles to {a.topic}")


def cmd_twin(a):
    """Subscribe to the bus and predict RUL live."""
    mqtt = _require_paho()
    model, builder = load_twin()

    state: dict[int, dict] = {}

    def on_connect(client, userdata, flags, reason_code, properties=None):
        client.subscribe(a.topic, qos=1)
        print(f"[twin-turbofan] subscribed to {a.topic} on {a.host}:{a.port}")

    def on_message(client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        if payload.get("event") == "end_of_stream":
            unit = payload.get("unit")
            print(f"[twin-turbofan] end of stream for engine {unit}")
            if a.plot and unit in state:
                s = state[unit]
                print(f"Saved -> {plot_divergence(s['c'], s['p'], s['t'], unit, a.threshold)}")
            return

        c, p, t, _ = handle_reading(builder, model, payload, a.threshold, not a.quiet)
        s = state.setdefault(payload["unit"], {"c": [], "p": [], "t": []})
        s["c"].append(c)
        s["p"].append(p)
        s["t"].append(t)

    client = _make_client(mqtt, "twin-turbofan-twin")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(a.host, a.port, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[twin-turbofan] stopped")
        client.disconnect()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--host", default=DEFAULT_HOST)
        sp.add_argument("--port", type=int, default=DEFAULT_PORT)
        sp.add_argument("--topic", default=DEFAULT_TOPIC)
        sp.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
        sp.add_argument("--quiet", action="store_true")
        sp.add_argument("--subset", default="FD001")

    sp = sub.add_parser("simulate", help="publisher + twin in one process, no broker")
    common(sp)
    sp.add_argument("--unit", type=int, default=1)
    sp.add_argument("--delay", type=float, default=0.0)
    sp.add_argument("--plot", action="store_true", default=True)
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("publish", help="replay an engine onto the bus")
    common(sp)
    sp.add_argument("--unit", type=int, default=1)
    sp.add_argument("--delay", type=float, default=0.05)
    sp.set_defaults(func=cmd_publish)

    sp = sub.add_parser("twin", help="subscribe and predict live")
    common(sp)
    sp.add_argument("--plot", action="store_true", default=True)
    sp.set_defaults(func=cmd_twin)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
