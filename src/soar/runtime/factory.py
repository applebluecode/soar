"""Composition root helpers: build runtime objects from config, nothing else.

Keeping construction here (instead of inside commands) means tests can swap any factory and
commands stay thin. No hidden global state: every object is passed explicitly.
"""

from __future__ import annotations

import time
from pathlib import Path

from soar.config import RecordConfig, RobotConfig, TeleopConfig
from soar.control.runner import PolicyRunner, Recorder, recorder_for
from soar.data.cameras import CameraRig, FakeCamera, OpenCVCamera
from soar.hardware.arm import make_arm
from soar.hardware.calibration import sha256_of
from soar.policy.teleop import TeleopPolicy

__all__ = [
    "build_camera_rig",
    "build_recorder",
    "build_runner",
    "build_teleop",
    "calibration_hash",
]


def calibration_hash(config: RobotConfig) -> str | None:
    """Hash of the explicit calibration file, if one was given."""
    if not config.calibration:
        return None
    path = Path(config.calibration).expanduser()
    return sha256_of(path) if path.is_file() else None


def build_camera_rig(names: tuple[str, ...], *, fake: bool) -> CameraRig:
    """Cameras by name; fake (test) or OpenCV (device index or path encoded in the name)."""
    if not names:
        return CameraRig(())
    cameras = []
    for name in names:
        if fake:
            cameras.append(FakeCamera(name))
        else:
            index: int | str = int(name) if name.isdigit() else name
            cameras.append(OpenCVCamera(name, index))
    return CameraRig(cameras)


def build_recorder(
    config: RecordConfig,
    arm: object,
    *,
    cameras: CameraRig,
    calibration_sha256: str | None = None,
) -> Recorder:
    """Recorder with episode metadata derived from the arm and cameras."""
    return recorder_for(
        root=config.root,
        task=config.task,
        arm=arm,
        record_config=config,
        calibration_sha256=calibration_sha256,
        cameras=cameras,
    )


def build_teleop(config: TeleopConfig) -> tuple[object, object, TeleopPolicy]:
    """Connect leader + follower and return ``(leader, follower, policy)``."""
    leader = make_arm(config.leader)
    follower = make_arm(config.follower)
    return leader, follower, TeleopPolicy(leader)


def build_runner(
    arm: object,
    policy: object,
    run: object,
    *,
    recorder: Recorder | None = None,
    cameras: CameraRig | None = None,
) -> PolicyRunner:
    return PolicyRunner(arm, policy, run, recorder=recorder, cameras=cameras)  # type: ignore[arg-type]


def timestamp_id(prefix: str = "run") -> str:
    """Sortable id for logs/checkpoints (filesystem-safe)."""
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"
