"""Online (streaming) feature construction for the live twin.

**Why this module exists.** ``features.build_xy`` is a batch operation: it needs the
whole training set to compute standardisation statistics, and the whole engine history
to compute rolling windows. A live twin has neither. It sees one cycle at a time and
must produce a feature vector immediately.

The original ``stream_demo`` sidestepped this by calling ``build_xy`` over the entire
test set up front and then replaying the pre-computed rows. That produces the right
numbers but is not a streaming system — it depends on data from the future, and it
would silently break the moment real telemetry arrived instead of a saved file.

So two pieces:

- ``FeatureSpec`` — the fitted feature contract (selected sensors, column order,
  rolling window, and the train-set mean/std), persisted at training time. Serving must
  reuse the *training* statistics; recomputing them from live data is the classic
  train/serve skew bug and would drift the model's inputs out from under it.
- ``OnlineFeatureBuilder`` — keeps a per-engine ring buffer of the last ``window``
  cycles and emits a standardised feature vector per reading.

``tests/test_online.py`` asserts the streaming path reproduces the batch path
bit-for-bit on the same engine, which is the property that makes the live twin
trustworthy.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from .paths import OUTPUTS_DIR

SPEC_PATH = OUTPUTS_DIR / "feature_spec.json"


@dataclass
class FeatureSpec:
    """The fitted feature contract shared by training and serving."""

    sensors: list[str]
    feature_cols: list[str]
    window: int
    mean: dict[str, float] = field(repr=False)
    std: dict[str, float] = field(repr=False)
    use_rolling: bool = True

    @classmethod
    def from_build_xy(cls, sensors, feature_cols, window, mean, std, use_rolling=True):
        """Capture the spec from a ``features.build_xy`` call."""
        return cls(
            sensors=list(sensors),
            feature_cols=list(feature_cols),
            window=int(window),
            mean={k: float(v) for k, v in dict(mean).items()},
            std={k: float(v) for k, v in dict(std).items()},
            use_rolling=bool(use_rolling),
        )

    def save(self, path=None):
        # Resolved at call time, not bound as a default. A default argument is evaluated
        # once at import, which silently pins the path: redirecting OUTPUTS_DIR (in tests,
        # or for a second experiment directory) would still write to the original
        # location. Found by a smoke test writing into the real outputs/ dir.
        path = path or SPEC_PATH
        path.write_text(json.dumps(self.__dict__, indent=2))
        return path

    @classmethod
    def load(cls, path=None):
        path = path or SPEC_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"No feature spec at {path}. Run `python -m src.train_baseline` first — "
                "it writes the spec alongside the model."
            )
        return cls(**json.loads(path.read_text()))

    @property
    def mean_vec(self) -> np.ndarray:
        return np.array([self.mean[c] for c in self.feature_cols], dtype=float)

    @property
    def std_vec(self) -> np.ndarray:
        return np.array([self.std[c] for c in self.feature_cols], dtype=float)


class OnlineFeatureBuilder:
    """Turn one cycle of raw sensor readings into a standardised feature vector.

    Holds a ``deque`` of the last ``window`` raw readings per engine, so rolling
    statistics match the batch computation's ``min_periods=1`` behaviour: early cycles
    use however much history exists rather than emitting NaN.

    Stateful and per-engine — call ``reset(unit)`` when an engine's run restarts.
    """

    def __init__(self, spec: FeatureSpec):
        self.spec = spec
        self._buf: dict[int, deque] = defaultdict(lambda: deque(maxlen=spec.window))

    def reset(self, unit: int | None = None) -> None:
        if unit is None:
            self._buf.clear()
        else:
            self._buf.pop(unit, None)

    def update(self, unit: int, reading: dict) -> np.ndarray:
        """Push one cycle and return its standardised feature vector.

        ``reading`` maps sensor name → value; only ``spec.sensors`` are consulted, so
        extra keys (cycle, operational settings, timestamps) are harmless.
        """
        raw = np.array([float(reading[s]) for s in self.spec.sensors], dtype=float)
        buf = self._buf[unit]
        buf.append(raw)

        if not self.spec.use_rolling:
            feats = raw
        else:
            hist = np.vstack(buf)
            rmean = hist.mean(axis=0)
            # Batch path uses pandas .std() (sample, ddof=1) and fills the
            # single-observation case with 0.0. Match both exactly.
            rstd = hist.std(axis=0, ddof=1) if len(hist) > 1 else np.zeros_like(raw)
            feats = np.concatenate([raw, rmean, rstd])

        return (feats - self.spec.mean_vec) / self.spec.std_vec

    def predict(self, model, unit: int, reading: dict) -> float:
        """Convenience: update then predict, clipped at 0 like the batch path."""
        x = self.update(unit, reading).reshape(1, -1)
        return float(np.clip(model.predict(x)[0], 0, None))
