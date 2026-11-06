"""Calibration service: middle convention, rollback, JSON paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pysoarm import SO101, FakeBus, read_calibration
from pysoarm.exceptions import CalibrationError

from soar.hardware.calibration import (
    calibrate_with_rollback,
    calibration_dir,
    calibration_path,
    save_calibration,
)


def make_bus(*, homing: int = 0, low: int = 0, high: int = 4095) -> FakeBus:
    bus = FakeBus()
    bus.open()
    for mid in bus.motor_ids:
        bus.set_register(mid, "homing_offset", homing)
        bus.set_register(mid, "min_position_limit", low)
        bus.set_register(mid, "max_position_limit", high)
    return bus


def scripted_stop(bus: FakeBus, *, low: int = 1000, high: int = 3000, steps: int = 8):
    state = {"n": 0}

    def stop() -> bool:
        n = state["n"]
        state["n"] += 1
        value = low + (high - low) * min(n, steps) // steps
        for mid in bus.motor_ids:
            bus.set_register(mid, "present_position", value)
        return n >= steps

    return stop


def test_middle_pose_determines_the_homing_offset(tmp_path: Path) -> None:
    bus = make_bus()
    try:

        def wait_middle() -> None:
            for mid in bus.motor_ids:
                bus.set_register(mid, "present_position", 2500)

        result = calibrate_with_rollback(
            bus,
            SO101,
            middle=True,
            wait_middle=wait_middle,
            should_stop=scripted_stop(bus),
            poll_s=0.0,
            output=tmp_path / "cal.json",
        )
        assert result.calibration.for_joint("shoulder_pan").homing_offset == 2500 - 2047
        assert result.json_path.is_file()
        assert json.loads(result.json_path.read_text()) == result.calibration.to_dict()
        assert bus.sync_read(SO101.motor_ids, "torque_enable") == dict.fromkeys(SO101.motor_ids, 0)
    finally:
        bus.close()


def test_legacy_mode_homes_at_the_current_pose(tmp_path: Path) -> None:
    bus = make_bus()
    for mid in bus.motor_ids:
        bus.set_register(mid, "present_position", 1500)
    try:
        result = calibrate_with_rollback(
            bus,
            SO101,
            middle=False,
            should_stop=scripted_stop(bus),
            poll_s=0.0,
            output=tmp_path / "cal.json",
        )
        assert result.calibration.for_joint("shoulder_pan").homing_offset == 1500 - 2047
    finally:
        bus.close()


def test_failure_restores_the_previous_motor_calibration(tmp_path: Path) -> None:
    bus = make_bus(homing=123, low=800, high=3200)
    try:
        with pytest.raises(CalibrationError, match="No motion"):
            calibrate_with_rollback(
                bus,
                SO101,
                middle=True,
                wait_middle=lambda: None,
                should_stop=lambda: True,
                poll_s=0.0,
                output=tmp_path / "cal.json",
            )
        restored = read_calibration(bus, SO101)
        for joint in SO101.joints:
            entry = restored.for_joint(joint)
            assert (entry.homing_offset, entry.range_min, entry.range_max) == (123, 800, 3200)
        assert (tmp_path / "cal.pre-calibrate.json").is_file()
    finally:
        bus.close()


def test_save_calibration_paths_and_backup(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_LEROBOT_CALIBRATION", raising=False)
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "hf"))
    bus = make_bus()
    try:
        calibration = read_calibration(bus, SO101)
    finally:
        bus.close()

    default = save_calibration(calibration, cal_id="arm")
    assert default == tmp_path / "arm.json"
    default.write_text('{"old": true}\n')
    save_calibration(calibration, cal_id="arm")
    assert list(tmp_path.glob("arm.bak-*.json"))

    lerobot = save_calibration(calibration, role="leader", cal_id="arm", lerobot_layout=True)
    assert lerobot == tmp_path / "hf" / "calibration" / "teleoperators" / "so_leader" / "arm.json"
    assert calibration_path("follower", "x") == (
        tmp_path / "hf" / "calibration" / "robots" / "so_follower" / "x.json"
    )
    assert calibration_dir("follower") == tmp_path / "hf" / "calibration" / "robots" / "so_follower"
