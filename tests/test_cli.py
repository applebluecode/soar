"""CLI smoke tests on FakeBus: teleop -> dataset -> replay -> eval."""

from __future__ import annotations

import json
from pathlib import Path

from soar.runtime.cli import main


def test_info(capsys) -> None:
    assert main(["info", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["soar"]
    assert "constant" in payload["policies"]


def test_teleop_records_and_dataset_validates(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "teleop",
                "--fake",
                "--hz",
                "50",
                "--seconds",
                "0.12",
                "--record",
                str(tmp_path),
                "--task",
                "cli",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["frames"] > 0
    assert payload["episode_path"]
    assert main(["dataset", "validate", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["dataset", "list", str(tmp_path)]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["episodes"] == 1
    assert len(listing["episode_list"]) == 1
    assert main(["train", str(tmp_path), "--json"]) == 0
    spec = json.loads(capsys.readouterr().out)
    assert spec["spec"]["state_dim"] == 6
    assert spec["spec"]["language"] is True


def test_replay_cli(tmp_path: Path, capsys) -> None:
    main(
        [
            "teleop",
            "--fake",
            "--hz",
            "50",
            "--seconds",
            "0.1",
            "--record",
            str(tmp_path),
            "--task",
            "replay",
            "--json",
        ]
    )
    episode = json.loads(capsys.readouterr().out)["episode_path"]
    assert (
        main(["replay", "--episode", episode, "--fake", "--hz", "50", "--speed", "2", "--json"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["completed"] is True
    assert payload["policy"] == "replay"


def test_eval_cli_with_registered_policy(capsys) -> None:
    assert (
        main(
            [
                "eval",
                "--policy",
                "constant",
                "--fake",
                "--episodes",
                "2",
                "--seconds",
                "0.05",
                "--hz",
                "50",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["episodes"] == 2
    assert payload["failures"] == []


def test_doctor_on_missing_device_is_a_clean_failure() -> None:
    assert main(["doctor", "--port", "/dev/does-not-exist", "--json"]) == 1
