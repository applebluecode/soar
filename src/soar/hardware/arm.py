"""SO-ARM hardware boundary: a thin, typed wrapper around :class:`pysoarm.SOArm`.

Nothing above this module imports ``pysoarm`` directly. That keeps the dependency rule
simple (hardware -> pysoarm, everything else -> protocols) and makes the whole framework
runnable against ``FakeBus``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pysoarm import SafetyLimits, SOArm

from soar.config import RobotConfig
from soar.errors import ConfigError

__all__ = ["Arm", "arm_info", "make_arm"]


class Arm(Protocol):
    """The minimal device surface the rest of the framework relies on."""

    @property
    def role(self) -> str: ...

    def read_once(self, health: bool | None = False) -> object: ...

    def write_goal(self, q_deg: Sequence[float]) -> tuple[float, ...]: ...

    def hold(self) -> object: ...

    def stop(self) -> None: ...

    def set_speed_scale(self, scale: float) -> float: ...


def make_arm(config: RobotConfig) -> SOArm:
    """Connect one arm from a :class:`RobotConfig`.

    Args:
        config: Port/role/backend/calibration/limits. ``fake=True`` uses ``FakeBus``.

    Returns:
        A connected :class:`pysoarm.SOArm`.

    Raises:
        ConfigError: Missing port for a real arm, unknown role, or invalid limits.
        pysoarm.PysoarmError: Bus/calibration failures.
    """
    if config.role not in ("follower", "leader"):
        raise ConfigError(f"role must be 'follower' or 'leader', got {config.role!r}")
    if not config.fake and not config.port:
        raise ConfigError("robot.port is required unless robot.fake is true")

    overrides: dict[str, object] = {}
    if config.max_speed_deg_s is not None:
        overrides["max_speed_deg_s"] = config.max_speed_deg_s
    if config.max_step_deg is not None:
        overrides["max_step_deg"] = config.max_step_deg
    if config.min_step_deg:
        overrides["min_step_deg"] = config.min_step_deg
    if config.voltage_min_v is not None:
        overrides["voltage_min_v"] = config.voltage_min_v
    limits = SafetyLimits(**overrides) if overrides else None  # type: ignore[arg-type]

    bus = None
    if config.fake:
        from pysoarm import FakeBus

        bus = FakeBus()

    arm = SOArm(
        port=config.port or "fake",
        bus=bus,
        role=config.role,  # type: ignore[arg-type]
        backend=config.backend,  # type: ignore[arg-type]
        calibration=config.calibration,
        limits=limits,
    )
    arm.connect()
    return arm


def arm_info(arm: SOArm) -> dict[str, object]:
    """Serialisable identity of a connected arm, stored in episode metadata."""
    import pysoarm

    model = arm.model
    return {
        "model": model.name,
        "joints": list(model.joints),
        "motor_ids": list(model.motor_ids),
        "role": arm.role,
        "backend": arm.backend,
        "pysoarm_version": pysoarm.__version__,
        "servo_model_number": model.servo_model_number,
    }
