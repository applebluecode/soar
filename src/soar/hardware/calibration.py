"""Calibration as a device-initialisation service (LeRobot-compatible JSON).

The flow follows LeRobot / pysoarm 0.1.3:

    torque off -> reset -> move to mid-travel -> half-turn homing -> record travel -> write

Interaction is injected: this module never reads stdin and never prints. The CLI supplies
``wait_middle``/``should_stop``/``progress`` callables. A failed calibration restores the
pre-call motor registers (best effort, 3 attempts) because a half-applied calibration is
worse than an aborted one.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from pysoarm import (
    Calibration,
    JointCalibration,
    get_model,
    read_calibration,
    reset_calibration,
    set_half_turn_homings,
    write_calibration,
)
from pysoarm.bus.base import Bus
from pysoarm.calibration import FULL_TURN_JOINTS
from pysoarm.constants import RESOLUTION_MAX
from pysoarm.exceptions import CalibrationError, PysoarmError
from pysoarm.models.base import RobotModel

from soar.errors import ConfigError

__all__ = [
    "CalibrationResult",
    "calibrate_with_rollback",
    "calibration_dir",
    "calibration_path",
    "default_output_path",
    "read_positions",
    "save_calibration",
    "sha256_of",
]

_ROLE_DIRS: dict[str, tuple[str, str]] = {
    "follower": ("robots", "so_follower"),
    "leader": ("teleoperators", "so_leader"),
}


def calibration_dir(role: str) -> Path:
    """Directory LeRobot uses for ``role``'s JSON files."""
    try:
        category, name = _ROLE_DIRS[role]
    except KeyError as exc:
        raise ConfigError(f"unknown role {role!r}") from exc
    explicit = os.environ.get("HF_LEROBOT_CALIBRATION")
    if explicit:
        base = Path(explicit).expanduser()
    else:
        home = os.environ.get("HF_LEROBOT_HOME")
        base = (
            Path(home).expanduser() if home else Path.home() / ".cache" / "huggingface" / "lerobot"
        ) / "calibration"
    return base / category / name


def calibration_path(role: str, cal_id: str) -> Path:
    """LeRobot path for ``<id>.json``."""
    return calibration_dir(role) / f"{cal_id}.json"


def default_output_path(cal_id: str) -> Path:
    """Default destination: ``./<id>.json``."""
    return Path.cwd() / f"{cal_id}.json"


def sha256_of(path: str | Path) -> str:
    """Content hash used in episode metadata to detect calibration drift."""
    digest = hashlib.sha256(Path(path).read_bytes())
    return digest.hexdigest()


def read_positions(bus: Bus, model: RobotModel) -> dict[int, int]:
    """Read ``present_position`` for every motor of the model."""
    return bus.sync_read(model.motor_ids, "present_position")


def save_calibration(
    calibration: Calibration,
    *,
    path: str | Path | None = None,
    role: str = "follower",
    cal_id: str = "default",
    lerobot_layout: bool = False,
    backup: bool = True,
) -> Path:
    """Write LeRobot-compatible JSON, keeping a timestamped backup when it exists."""
    if path is not None:
        target = Path(path).expanduser()
    elif lerobot_layout:
        target = calibration_path(role, cal_id)
    else:
        target = default_output_path(cal_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if backup and target.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        shutil.copy2(target, target.with_name(f"{target.stem}.bak-{stamp}{target.suffix}"))
    calibration.to_json(target)
    return target


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Outcome of a calibration run.

    Attributes:
        calibration: The calibration written to the motors.
        json_path: Where the JSON copy was saved.
        snapshot_path: Pre-calibration snapshot (for rollback/debugging).
        changed_joints: Joints whose values differ from the pre-call state.
    """

    calibration: Calibration
    json_path: Path
    snapshot_path: Path
    changed_joints: tuple[str, ...]

    @property
    def sha256(self) -> str:
        return sha256_of(self.json_path)


def _snapshot(calibration: Calibration, target: Path) -> Path:
    path = target.with_name(f"{target.stem}.pre-calibrate{target.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_json(path)
    return path


def _restore(bus: Bus, model: RobotModel, calibration: Calibration) -> bool:
    for _ in range(3):
        try:
            write_calibration(bus, model, calibration)
            return True
        except PysoarmError:
            time.sleep(0.2)
    return False


def _record_travel(
    bus: Bus,
    model: RobotModel,
    *,
    should_stop: Callable[[], bool],
    poll_s: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    progress: Callable[[str, Mapping[int, int], Mapping[str, int], Mapping[str, int]], None] | None,
) -> tuple[dict[str, int], dict[str, int]]:
    mins: dict[str, int] = {}
    maxs: dict[str, int] = {}
    while True:
        positions = read_positions(bus, model)
        for joint, mid in zip(model.joints, model.motor_ids, strict=True):
            value = positions[mid]
            mins[joint] = min(mins.get(joint, value), value)
            maxs[joint] = max(maxs.get(joint, value), value)
        if progress is not None:
            progress("record", positions, mins, maxs)
        if should_stop():
            break
        sleep(poll_s)
    return mins, maxs


def calibrate_with_rollback(
    bus: Bus,
    model: RobotModel,
    *,
    role: str = "follower",
    middle: bool = True,
    wait_middle: Callable[[], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    full_turn: bool = False,
    poll_s: float = 0.05,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    progress: Callable[[str, Mapping[int, int], Mapping[str, int], Mapping[str, int]], None]
    | None = None,
    output: str | Path | None = None,
    cal_id: str = "default",
    lerobot_layout: bool = False,
    backup: bool = True,
    snapshot: bool = True,
) -> CalibrationResult:
    """Run the motor-side calibration and persist LeRobot JSON.

    Args:
        bus: Open transport.
        model: Arm model.
        role: ``"follower"`` or ``"leader"``.
        middle: Wait for mid-travel before half-turn homing (LeRobot convention). When
            false, the current pose becomes 2048 (legacy).
        wait_middle: Blocking callable; must return only when the operator placed every
            joint near mid-travel. Required when ``middle`` is true (no stdin here).
        should_stop: Polled while recording; required.
        full_turn: Record ``wrist_roll`` instead of forcing 0..4095.
        poll_s: Sampling period.
        sleep, clock: Injectable timing for headless tests.
        progress: Optional ``(stage, positions, mins, maxs)`` callback for a console view.
        output: JSON destination (default ``./<id>.json`` or LeRobot layout).
        cal_id: Name used for the default paths.
        lerobot_layout: Use LeRobot's ``calibration/`` directory.
        backup: Keep a timestamped backup of an existing JSON.
        snapshot: Write ``<id>.pre-calibrate.json`` and restore motors on failure.

    Raises:
        ConfigError: Missing ``wait_middle``/``should_stop`` for an interactive stage.
        CalibrationError: A joint never moved.
        PysoarmError: Bus failure; the pre-call calibration is restored when possible.
    """
    if role not in _ROLE_DIRS:
        raise ConfigError(f"unknown role {role!r}")
    if middle and wait_middle is None:
        raise ConfigError("middle=True needs a wait_middle callable")
    if should_stop is None:
        raise ConfigError("calibration needs a should_stop callable")

    target = (
        Path(output).expanduser()
        if output is not None
        else calibration_path(role, cal_id)
        if lerobot_layout
        else default_output_path(cal_id)
    )
    before = read_calibration(bus, model)
    snapshot_path = _snapshot(before, target) if snapshot else target

    try:
        bus.disable_torque(model.motor_ids)
        reset_calibration(bus, model)
        if middle:
            assert wait_middle is not None
            wait_middle()
        offsets = set_half_turn_homings(bus, model)
        mins, maxs = _record_travel(
            bus,
            model,
            should_stop=should_stop,
            poll_s=poll_s,
            sleep=sleep,
            clock=clock,
            progress=progress,
        )
        if not full_turn:
            still = [
                joint
                for joint in model.joints
                if joint not in FULL_TURN_JOINTS and maxs[joint] - mins[joint] < 1
            ]
            if still:
                raise CalibrationError(
                    f"No motion recorded for {still}; move every joint through its travel"
                )
        joints: dict[str, JointCalibration] = {}
        for joint, mid in zip(model.joints, model.motor_ids, strict=True):
            lo, hi = mins[joint], maxs[joint]
            if not full_turn and joint in FULL_TURN_JOINTS:
                lo, hi = 0, RESOLUTION_MAX
            joints[joint] = JointCalibration(
                motor_id=mid, range_min=lo, range_max=hi, homing_offset=offsets[joint]
            )
        calibration = Calibration(joints)
        calibration.validate(model)
        write_calibration(bus, model, calibration)
    except BaseException:
        if snapshot and not _restore(bus, model, before):
            raise PysoarmError(
                f"calibration failed and could not be restored; snapshot: {snapshot_path}"
            ) from None
        raise
    finally:
        with contextlib.suppress(PysoarmError):
            bus.disable_torque(model.motor_ids)

    path = save_calibration(
        calibration,
        path=target,
        role=role,
        cal_id=cal_id,
        lerobot_layout=lerobot_layout,
        backup=backup,
    )
    changed = tuple(
        joint for joint in model.joints if before.for_joint(joint) != calibration.for_joint(joint)
    )
    return CalibrationResult(
        calibration=calibration,
        json_path=path,
        snapshot_path=snapshot_path,
        changed_joints=changed,
    )


def model_for(name: str) -> RobotModel:
    """Resolve ``"so101"``/``"so100"`` to a model (kept here for CLI convenience)."""
    return get_model(name)


def calibration_table(calibration: Calibration, model: RobotModel) -> list[dict[str, object]]:
    """Human/JSON-friendly summary of a calibration."""
    rows: list[dict[str, object]] = []
    for joint in model.joints:
        entry = calibration.for_joint(joint)
        rows.append(
            {
                "joint": joint,
                "id": entry.motor_id,
                "drive_mode": entry.drive_mode,
                "homing_offset": entry.homing_offset,
                "range_min": entry.range_min,
                "range_max": entry.range_max,
                "span_ticks": entry.range_max - entry.range_min,
                "span_deg": round(entry.span_deg, 2),
            }
        )
    return rows


def dumps_calibration(calibration: Calibration) -> str:
    return json.dumps(calibration.to_dict(), indent=2)


def _iter_joints(model: RobotModel) -> Iterable[tuple[str, int]]:
    return zip(model.joints, model.motor_ids, strict=True)
