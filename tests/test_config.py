"""Typed config loading: defaults, JSON/YAML, unknown keys."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soar.config import AppConfig, RecordConfig, RobotConfig, load_config
from soar.errors import ConfigError


def test_defaults_are_policy_agnostic() -> None:
    config = AppConfig()
    assert config.robot.role == "follower"
    assert config.leader.role == "leader"
    assert config.run.hz == 200.0
    assert config.record.writer == "jsonl"


def test_load_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "robot": {"port": "/dev/ttyACM0", "max_speed_deg_s": 720},
                "leader": {"port": "/dev/ttyACM1"},
                "run": {"hz": 100},
                "record": {"root": "data", "cameras": ["wrist"], "task": "pick"},
            }
        )
    )
    config = load_config(path)
    assert isinstance(config.robot, RobotConfig)
    assert config.robot.max_speed_deg_s == 720
    assert config.run.hz == 100
    assert isinstance(config.record, RecordConfig)
    assert config.record.cameras == ("wrist",)


def test_load_yaml_when_available(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "config.yaml"
    path.write_text("robot:\n  port: /dev/ttyACM0\nrun:\n  hz: 50\n")
    config = load_config(path)
    assert config.robot.port == "/dev/ttyACM0"
    assert config.run.hz == 50


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"robot": {"prot": "/dev/ttyACM0"}}))
    with pytest.raises(ConfigError, match="unknown RobotConfig"):
        load_config(path)
    path.write_text(json.dumps({"nope": 1}))
    with pytest.raises(ConfigError, match="unknown top-level"):
        load_config(path)


def test_missing_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("/nonexistent/config.json")


def test_shipped_configs_are_valid() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    for path in sorted(root.glob("*.json")):
        config = load_config(path)
        assert config.robot.role in ("follower", "leader")


def test_run_config_rejects_invalid_chunking() -> None:
    from soar.config import RunConfig
    from soar.errors import ConfigError

    with pytest.raises(ConfigError, match="replan_every"):
        RunConfig(replan_every=0)
    with pytest.raises(ConfigError, match="read_state_every"):
        RunConfig(read_state_every=-1)
