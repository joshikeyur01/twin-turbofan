"""Tests for the telemetry bus and live-twin logic.

Covers the parts that do not need a broker: payload construction, per-reading scoring,
alerting, and the divergence figure. The MQTT publish/subscribe wiring itself is thin
glue over paho and is exercised by running ``simulate``, which shares all the scoring
logic with the subscriber via ``handle_reading``.
"""

import numpy as np
import pytest

from src.data_loader import load_cmapss
from src.features import build_xy, select_informative_sensors
from src.online import FeatureSpec, OnlineFeatureBuilder
from src.telemetry import DEFAULT_THRESHOLD, engine_stream, handle_reading, plot_divergence


class ConstantModel:
    """Predicts a fixed RUL, so alert logic is testable without training anything."""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


@pytest.fixture
def builder(cmapss_dir):
    train, test = load_cmapss("FD001")
    _, _, feat_cols, (mean, std) = build_xy(train, test, window=5)
    spec = FeatureSpec.from_build_xy(
        sensors=select_informative_sensors(train),
        feature_cols=feat_cols,
        window=5,
        mean=mean,
        std=std,
    )
    return OnlineFeatureBuilder(spec)


class TestEngineStream:
    def test_yields_every_cycle_in_order(self, cmapss_dir):
        payloads = list(engine_stream(1))
        cycles = [p["cycle"] for p in payloads]
        assert cycles == sorted(cycles)
        assert len(payloads) == cmapss_dir["test_truth"][1][0]

    def test_payload_shape(self, cmapss_dir):
        p = next(iter(engine_stream(1)))
        assert set(p) == {"unit", "cycle", "rul_true", "sensors"}
        assert p["unit"] == 1
        assert len(p["sensors"]) == 21

    def test_sensors_only_contains_sensor_keys(self, cmapss_dir):
        p = next(iter(engine_stream(1)))
        assert all(k.startswith("s") for k in p["sensors"])
        # operational settings must not leak in under the "s" prefix filter
        assert not any(k.startswith("os") for k in p["sensors"])

    def test_rul_true_decreases(self, cmapss_dir):
        ruls = [p["rul_true"] for p in engine_stream(1)]
        assert ruls == sorted(ruls, reverse=True)

    def test_payload_is_json_serialisable(self, cmapss_dir):
        """Values must be plain floats/ints, not numpy scalars, to survive json.dumps."""
        import json

        p = next(iter(engine_stream(1)))
        json.loads(json.dumps(p))  # raises TypeError on numpy types

    def test_unknown_engine_exits(self, cmapss_dir):
        with pytest.raises(SystemExit, match="not found"):
            list(engine_stream(999))


class TestHandleReading:
    def test_returns_cycle_pred_true_alert(self, builder, cmapss_dir):
        payload = next(iter(engine_stream(1)))
        cycle, pred, true, alert = handle_reading(
            builder, ConstantModel(50.0), payload, verbose=False
        )
        assert cycle == payload["cycle"]
        assert pred == pytest.approx(50.0)
        assert true == payload["rul_true"]
        assert alert is False

    def test_alert_fires_below_threshold(self, builder, cmapss_dir):
        payload = next(iter(engine_stream(1)))
        _, _, _, alert = handle_reading(
            builder, ConstantModel(10.0), payload, threshold=25.0, verbose=False
        )
        assert alert is True

    def test_alert_boundary_is_strict(self, builder, cmapss_dir):
        """Exactly at the threshold is not yet an alert."""
        payload = next(iter(engine_stream(1)))
        _, _, _, alert = handle_reading(
            builder, ConstantModel(25.0), payload, threshold=25.0, verbose=False
        )
        assert alert is False

    def test_negative_predictions_clipped(self, builder, cmapss_dir):
        payload = next(iter(engine_stream(1)))
        _, pred, _, _ = handle_reading(builder, ConstantModel(-30.0), payload, verbose=False)
        assert pred == 0.0

    def test_default_threshold_is_conservative(self):
        """A sanity check on the operational default, not on model behaviour."""
        assert 0 < DEFAULT_THRESHOLD <= 50


class TestDivergencePlot:
    def test_writes_a_figure(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.telemetry.OUTPUTS_DIR", tmp_path)
        cycles = list(range(1, 21))
        trues = [float(120 - c) for c in cycles]
        preds = [t - 3 for t in trues]
        path = plot_divergence(cycles, preds, trues, unit=7)
        assert path.exists()
        assert path.stat().st_size > 0
        assert "engine7" in path.name

    def test_handles_missing_truth(self, tmp_path, monkeypatch):
        """A live bus may deliver readings with no ground truth attached."""
        monkeypatch.setattr("src.telemetry.OUTPUTS_DIR", tmp_path)
        cycles = list(range(1, 11))
        path = plot_divergence(cycles, [50.0] * 10, [None] * 10, unit=2)
        assert path.exists()

    def test_handles_never_alerting_engine(self, tmp_path, monkeypatch):
        """No crossing of the threshold means no annotation — must not IndexError."""
        monkeypatch.setattr("src.telemetry.OUTPUTS_DIR", tmp_path)
        cycles = list(range(1, 11))
        path = plot_divergence(cycles, [120.0] * 10, [125.0] * 10, unit=3, threshold=25)
        assert path.exists()
