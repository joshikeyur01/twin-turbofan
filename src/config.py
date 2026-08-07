"""Layered configuration and centralised logging.

Config values previously lived as literals spread across `src/` — the RUL cap in
`data_loader`, the rolling window in two places, the alert threshold in `telemetry`. That
makes an experiment protocol hard to audit: a reviewer has to read five modules to learn
what was actually run. `config.yaml` collects them.

**Precedence, lowest to highest:**

1. the dataclass defaults below — so the package still works with no YAML present at all
2. `config.yaml` at the repo root
3. a file named by the ``TWIN_CONFIG`` environment variable
4. explicit keyword overrides (which entry points feed from argparse)

That ordering is deliberate: defaults keep the code importable in a fresh checkout or a
container with no config mounted, and CLI flags always win so a one-off run never requires
editing a tracked file.

Usage from an entry point::

    from .config import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg.logging_level)
    log = logging.getLogger(__name__)

PyYAML is optional. Without it the dataclass defaults are used and a warning is logged,
which keeps the numpy/pandas-only guarantee intact.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .paths import ROOT

DEFAULT_CONFIG_PATH = ROOT / "config.yaml"
ENV_VAR = "TWIN_CONFIG"

log = logging.getLogger(__name__)


@dataclass
class Config:
    """Flat view of the settings, with nested YAML mapped onto dotted names."""

    # data
    subset: str = "FD001"
    rul_cap: int = 125
    # features
    rolling_window: int = 5
    variance_threshold: float = 1e-3
    use_rolling: bool = True
    # training
    seed: int = 42
    val_frac: float = 0.2
    device: str = "auto"
    # sequence model
    arch: str = "gru"
    seq_len: int = 50
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.2
    lr: float = 3e-4
    epochs: int = 80
    batch: int = 256
    patience: int = 10
    # uncertainty
    calibration_frac: float = 0.3
    # twin
    alert_threshold: float = 25.0
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic: str = "twin-turbofan/telemetry"
    # logging
    logging_level: str = "INFO"
    logging_timestamps: bool = False

    # Provenance, for the record — not settable from YAML.
    sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# YAML nesting -> flat dataclass field. Kept explicit rather than derived so a typo in the
# YAML is caught (see _flatten) instead of silently ignored.
_SCHEMA: dict[str, dict[str, str]] = {
    "data": {"subset": "subset", "rul_cap": "rul_cap"},
    "features": {
        "rolling_window": "rolling_window",
        "variance_threshold": "variance_threshold",
        "use_rolling": "use_rolling",
    },
    "training": {"seed": "seed", "val_frac": "val_frac", "device": "device"},
    "sequence_model": {
        "arch": "arch",
        "seq_len": "seq_len",
        "hidden": "hidden",
        "layers": "layers",
        "dropout": "dropout",
        "lr": "lr",
        "epochs": "epochs",
        "batch": "batch",
        "patience": "patience",
    },
    "uncertainty": {"calibration_frac": "calibration_frac"},
    "twin": {"alert_threshold": "alert_threshold"},
    "logging": {"level": "logging_level", "timestamps": "logging_timestamps"},
}
_MQTT = {"host": "mqtt_host", "port": "mqtt_port", "topic": "mqtt_topic"}


def _flatten(raw: dict[str, Any]) -> dict[str, Any]:
    """Map nested YAML onto flat field names, raising on anything unrecognised.

    Silently dropping unknown keys is the worst option here: a typo like `rul_capp` would
    leave the default in place and the run would look fine while measuring something else.
    """
    flat: dict[str, Any] = {}
    known = {f.name for f in fields(Config)}

    for section, body in (raw or {}).items():
        if section not in _SCHEMA:
            raise ValueError(f"unknown config section {section!r}; expected {sorted(_SCHEMA)}")
        if not isinstance(body, dict):
            raise ValueError(f"config section {section!r} must be a mapping, got {type(body)}")

        for key, value in body.items():
            if section == "twin" and key == "mqtt":
                if not isinstance(value, dict):
                    raise ValueError("twin.mqtt must be a mapping")
                for mk, mv in value.items():
                    if mk not in _MQTT:
                        raise ValueError(f"unknown twin.mqtt key {mk!r}")
                    flat[_MQTT[mk]] = mv
                continue
            if key not in _SCHEMA[section]:
                raise ValueError(
                    f"unknown config key {section}.{key!r}; " f"expected {sorted(_SCHEMA[section])}"
                )
            target = _SCHEMA[section][key]
            assert target in known
            flat[target] = value
    return flat


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Build a Config from defaults, YAML, ``TWIN_CONFIG``, then keyword overrides.

    ``overrides`` with a value of ``None`` are ignored, so an entry point can forward
    argparse results wholesale without unset flags clobbering the file's values.
    """
    values: dict[str, Any] = {}
    sources: list[str] = ["defaults"]

    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        candidates.append(DEFAULT_CONFIG_PATH)
        if os.environ.get(ENV_VAR):
            candidates.append(Path(os.environ[ENV_VAR]))

    for candidate in candidates:
        if not candidate.exists():
            if path is not None:
                raise FileNotFoundError(f"config file not found: {candidate}")
            continue
        try:
            import yaml
        except ImportError:
            log.warning("PyYAML not installed — ignoring %s and using defaults", candidate)
            break
        raw = yaml.safe_load(candidate.read_text()) or {}
        values.update(_flatten(raw))
        sources.append(str(candidate))

    clean = {k: v for k, v in overrides.items() if v is not None}
    unknown = set(clean) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown config override(s): {sorted(unknown)}")
    if clean:
        values.update(clean)
        sources.append("cli/overrides")

    values.pop("sources", None)
    return Config(**values, sources=sources)


def setup_logging(level: str | int = "INFO", timestamps: bool = False) -> None:
    """Configure root logging once, consistently across every entry point.

    ``force=True`` because libraries in the dependency tree (matplotlib, streamlit) install
    their own handlers on import; without it the first one to touch logging wins and this
    call would silently do nothing.
    """
    fmt = "%(levelname)s %(name)s: %(message)s"
    if timestamps:
        fmt = "%(asctime)s " + fmt
    logging.basicConfig(
        level=level if isinstance(level, int) else str(level).upper(),
        format=fmt,
        datefmt="%H:%M:%S",
        force=True,
    )
    # These are chatty at DEBUG and never carry information we want.
    for noisy in ("matplotlib", "PIL", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def add_config_args(parser) -> None:
    """Attach ``--config`` and ``--log-level`` to an argparse parser."""
    parser.add_argument("--config", default=None, help="path to a YAML config file")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="override logging.level from the config",
    )
