"""Tests for layered configuration.

The behaviour worth pinning down is the precedence chain and the strictness of key
validation. A config layer that silently ignores an unrecognised key is worse than no
config layer at all: `rul_capp: 100` would leave the real cap at its default and the run
would look correct while measuring something else entirely.
"""

import logging

import pytest

from src.config import Config, load_config, setup_logging

yaml = pytest.importorskip("yaml", reason="config layering needs PyYAML")


def write(path, text):
    path.write_text(text)
    return path


class TestDefaults:
    def test_loads_with_no_file_at_all(self, tmp_path, monkeypatch):
        """A fresh checkout or a container with no config mounted must still work."""
        monkeypatch.setattr("src.config.DEFAULT_CONFIG_PATH", tmp_path / "absent.yaml")
        monkeypatch.delenv("TWIN_CONFIG", raising=False)
        cfg = load_config()
        assert cfg.rul_cap == 125
        assert cfg.sources == ["defaults"]

    def test_repo_config_matches_the_dataclass_defaults(self):
        """Drift guard: the tracked YAML should not silently contradict the code.

        If these diverge, a reader of config.yaml and a reader of the code disagree about
        what the experiment used.
        """
        cfg = load_config()
        defaults = Config()
        for name in ("rul_cap", "rolling_window", "seed", "val_frac", "alert_threshold"):
            assert getattr(cfg, name) == getattr(defaults, name), name

    def test_missing_explicit_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="config file not found"):
            load_config(tmp_path / "nope.yaml")


class TestPrecedence:
    def test_yaml_overrides_defaults(self, tmp_path):
        p = write(tmp_path / "c.yaml", "data:\n  rul_cap: 90\n")
        assert load_config(p).rul_cap == 90

    def test_overrides_beat_yaml(self, tmp_path):
        p = write(tmp_path / "c.yaml", "data:\n  rul_cap: 90\n")
        assert load_config(p, rul_cap=42).rul_cap == 42

    def test_none_overrides_are_ignored(self, tmp_path):
        """Unset argparse flags arrive as None and must not clobber the file."""
        p = write(tmp_path / "c.yaml", "data:\n  rul_cap: 90\n")
        assert load_config(p, rul_cap=None).rul_cap == 90

    def test_env_var_is_applied(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.config.DEFAULT_CONFIG_PATH", tmp_path / "absent.yaml")
        p = write(tmp_path / "env.yaml", "training:\n  seed: 7\n")
        monkeypatch.setenv("TWIN_CONFIG", str(p))
        cfg = load_config()
        assert cfg.seed == 7
        assert str(p) in cfg.sources

    def test_env_var_beats_repo_config(self, tmp_path, monkeypatch):
        base = write(tmp_path / "base.yaml", "training:\n  seed: 1\n")
        over = write(tmp_path / "over.yaml", "training:\n  seed: 2\n")
        monkeypatch.setattr("src.config.DEFAULT_CONFIG_PATH", base)
        monkeypatch.setenv("TWIN_CONFIG", str(over))
        assert load_config().seed == 2

    def test_sources_records_provenance(self, tmp_path):
        p = write(tmp_path / "c.yaml", "data:\n  rul_cap: 90\n")
        cfg = load_config(p, seed=3)
        assert cfg.sources[0] == "defaults"
        assert str(p) in cfg.sources
        assert cfg.sources[-1] == "cli/overrides"


class TestStrictness:
    def test_unknown_section_raises(self, tmp_path):
        p = write(tmp_path / "c.yaml", "nonsense:\n  x: 1\n")
        with pytest.raises(ValueError, match="unknown config section"):
            load_config(p)

    def test_unknown_key_raises(self, tmp_path):
        """A typo must fail loudly, not leave the default silently in place."""
        p = write(tmp_path / "c.yaml", "data:\n  rul_capp: 90\n")
        with pytest.raises(ValueError, match="unknown config key"):
            load_config(p)

    def test_unknown_override_raises(self):
        with pytest.raises(ValueError, match="unknown config override"):
            load_config(nonexistent_setting=1)

    def test_non_mapping_section_raises(self, tmp_path):
        p = write(tmp_path / "c.yaml", "data: 5\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_config(p)

    def test_empty_file_is_valid(self, tmp_path):
        p = write(tmp_path / "c.yaml", "")
        assert load_config(p).rul_cap == 125

    def test_partial_file_keeps_other_defaults(self, tmp_path):
        p = write(tmp_path / "c.yaml", "data:\n  rul_cap: 60\n")
        cfg = load_config(p)
        assert cfg.rul_cap == 60
        assert cfg.rolling_window == Config().rolling_window


class TestNestedMqtt:
    def test_mqtt_block_flattens(self, tmp_path):
        p = write(
            tmp_path / "c.yaml",
            "twin:\n  alert_threshold: 10\n  mqtt:\n    host: broker\n    port: 8883\n",
        )
        cfg = load_config(p)
        assert cfg.alert_threshold == 10
        assert cfg.mqtt_host == "broker"
        assert cfg.mqtt_port == 8883

    def test_unknown_mqtt_key_raises(self, tmp_path):
        p = write(tmp_path / "c.yaml", "twin:\n  mqtt:\n    hostt: broker\n")
        with pytest.raises(ValueError, match="unknown twin.mqtt key"):
            load_config(p)


class TestLogging:
    def test_sets_the_level(self):
        setup_logging("DEBUG")
        assert logging.getLogger().level == logging.DEBUG
        setup_logging("INFO")
        assert logging.getLogger().level == logging.INFO

    def test_overrides_a_preinstalled_handler(self):
        """force=True matters: imported libraries configure logging first otherwise."""
        logging.basicConfig(level=logging.ERROR)
        setup_logging("INFO")
        assert logging.getLogger().level == logging.INFO

    def test_noisy_libraries_are_quietened(self):
        setup_logging("DEBUG")
        assert logging.getLogger("matplotlib").level == logging.WARNING

    def test_timestamps_toggle_changes_the_format(self):
        setup_logging("INFO", timestamps=True)
        fmt = logging.getLogger().handlers[0].formatter._fmt
        assert "asctime" in fmt
        setup_logging("INFO", timestamps=False)
        fmt = logging.getLogger().handlers[0].formatter._fmt
        assert "asctime" not in fmt
