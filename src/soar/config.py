"""Typed configuration objects and a dependency-free loader.

Configuration is plain dataclasses so it can be constructed in tests, in notebooks or from a
YAML/JSON file without a framework (Hydra/OmegaConf are not required). Unknown keys are
rejected, which catches typos instead of silently ignoring them.

File format::

    {
      "robot":  {"port": "/dev/ttyACM0", "backend": "auto"},
      "leader": {"port": "/dev/ttyACM1", "role": "leader"},
      "run":    {"hz": 200.0, "on_error": "hold"},
      "record": {"root": "data", "task": "pick_cube"}
    }
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from soar.errors import ConfigError

__all__ = [
    "AppConfig",
    "EvalConfig",
    "RecordConfig",
    "ReplayConfig",
    "RobotConfig",
    "RunConfig",
    "TeleopConfig",
    "load_config",
]

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RobotConfig:
    """One arm on the bus.

    Args:
        port: Serial device (ignored when ``fake`` is true).
        role: ``"follower"`` (writable, torque on) or ``"leader"`` (read-only, torque off).
        backend: ``"auto"``, ``"scs"`` or ``"rustypot"``.
        calibration: Optional LeRobot JSON path; ``None`` reads the motor-side calibration.
        fake: Use an in-memory ``FakeBus`` instead of a serial port.
        max_speed_deg_s: Optional velocity ceiling; ``None`` means no software cap (0.1.2+
            pysoarm default). Teleoperation usually wants ``720``.
        max_step_deg: Optional absolute per-tick cap.
        min_step_deg: Deadband below which a command is not sent.
        voltage_min_v: Supply threshold; ``None`` uses pysoarm's role-aware default.
    """

    port: str | None = None
    role: str = "follower"
    backend: str = "auto"
    calibration: str | None = None
    fake: bool = False
    max_speed_deg_s: float | None = None
    max_step_deg: float | None = None
    min_step_deg: float = 0.0
    voltage_min_v: float | None = None


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Control-loop settings shared by teleoperation, replay and evaluation.

    Args:
        hz: Target loop rate.
        duration: Seconds to run; ``None`` runs until Ctrl-C.
        on_error: ``"abort"``, ``"hold"`` or ``"skip"`` (pysoarm loop policy).
        max_consecutive_errors: Cutoff before a handled error is re-raised.
        health_period_s: ``None`` disables background health refresh in the hot path.
        speed_scale: Multiplier applied only when a speed/step cap is configured.
        read_state_every: Read a fresh state every N ticks (``0`` reuses the cache).
        instruction: Natural-language prompt passed to the policy (VLA/WAM).
        replan_every: Execute at most this many steps of an action chunk before calling the
            policy again (``1`` = re-plan every tick, the pre-chunk behaviour).
    """

    hz: float = 200.0
    duration: float | None = None
    on_error: str = "hold"
    max_consecutive_errors: int = 10
    health_period_s: float | None = None
    speed_scale: float = 1.0
    read_state_every: int = 1
    instruction: str = ""
    replan_every: int = 1

    def __post_init__(self) -> None:
        if self.replan_every < 1:
            raise ConfigError("run.replan_every must be >= 1")
        if self.read_state_every < 0:
            raise ConfigError("run.read_state_every must be >= 0")


@dataclass(frozen=True, slots=True)
class RecordConfig:
    """Episode recording settings.

    Args:
        root: Dataset root directory; episodes are written as ``<root>/<episode_id>``.
        task: Natural-language task label stored in the episode metadata.
        cameras: Camera names to capture (must exist in the runtime camera rig).
        fps: Target video/frame rate hint for downstream training.
        queue_size: Bounded queue between the control loop and the writer thread.
        flush_interval_s: How often the writer flushes buffered JSONL.
        writer: ``"jsonl"`` (default, stdlib only) or ``"parquet"`` (needs pyarrow).
        save_images: Write camera frames to disk (otherwise only timestamps are stored).
    """

    root: str = "data"
    task: str = ""
    cameras: tuple[str, ...] = ()
    fps: float = 30.0
    queue_size: int = 4096
    flush_interval_s: float = 0.5
    writer: str = "jsonl"
    save_images: bool = True


@dataclass(frozen=True, slots=True)
class TeleopConfig:
    """Leader/follower teleoperation definition."""

    leader: RobotConfig = field(default_factory=lambda: RobotConfig(role="leader"))
    follower: RobotConfig = field(default_factory=RobotConfig)
    run: RunConfig = field(default_factory=RunConfig)
    record: RecordConfig | None = None


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Open-loop replay of a recorded episode.

    Args:
        episode: Path to an episode directory.
        speed: Time multiplier (``0.5`` half speed, ``2.0`` double).
        use_raw_actions: Replay the policy's raw action instead of the guarded goal that was
            actually sent (useful to study clipping).
    """

    episode: str = ""
    speed: float = 1.0
    use_raw_actions: bool = False


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Policy evaluation settings.

    Args:
        episodes: Number of episodes to run.
        max_duration_s: Per-episode timeout.
        policy: Registered policy name (see :mod:`soar.policy.registry`).
        policy_kwargs: Keyword arguments for the policy factory.
        success_fn: ``"module:function"`` receiving a ``RunResult`` and returning ``bool``.
        record_root: Optional dataset root; every episode is recorded when set.
        task: Task label for recorded evaluation episodes.
        instruction: Override the language prompt; defaults to ``task``.
    """

    episodes: int = 5
    max_duration_s: float = 30.0
    policy: str = "constant"
    policy_kwargs: Mapping[str, Any] = field(default_factory=dict)
    success_fn: str | None = None
    record_root: str | None = None
    task: str = "eval"
    instruction: str = ""


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level configuration.

    ``robot`` is the follower (or a single arm); ``leader`` is only used by teleoperation.
    """

    robot: RobotConfig = field(default_factory=RobotConfig)
    leader: RobotConfig = field(default_factory=lambda: RobotConfig(role="leader"))
    run: RunConfig = field(default_factory=RunConfig)
    record: RecordConfig = field(default_factory=RecordConfig)
    teleop: TeleopConfig | None = None
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type[T], data: Mapping[str, Any] | None) -> T:
    """Construct a dataclass from a mapping, rejecting unknown keys."""
    if data is None:
        return cls()  # type: ignore[call-arg]
    if not isinstance(data, Mapping):
        raise ConfigError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    allowed = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    payload = dict(data)
    if cls is RecordConfig and "cameras" in payload:
        payload["cameras"] = tuple(payload["cameras"])
    if cls is TeleopConfig:
        payload["leader"] = _build(RobotConfig, payload.get("leader"))
        payload["follower"] = _build(RobotConfig, payload.get("follower"))
        payload["run"] = _build(RunConfig, payload.get("run"))
        record = payload.get("record")
        payload["record"] = _build(RecordConfig, record) if record is not None else None
    return cls(**payload)  # type: ignore[call-arg]


def _read_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".json", ""):
        return json.loads(text)
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise ConfigError('YAML config needs "pip install soar-framework[yaml]"') from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, Mapping):
            raise ConfigError(f"{path}: expected a mapping at the top level")
        return loaded
    raise ConfigError(f"unsupported config extension {path.suffix!r} (use .json/.yaml)")


def load_config(path: str | Path) -> AppConfig:
    """Load an :class:`AppConfig` from JSON or YAML.

    Raises:
        ConfigError: Missing file, unsupported suffix, unknown keys or wrong types.
    """
    file = Path(path).expanduser()
    if not file.is_file():
        raise ConfigError(f"config not found: {file}")
    try:
        data = _read_mapping(file)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{file}: invalid JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError(f"{file}: expected a mapping at the top level")
    known = {f.name for f in dataclasses.fields(AppConfig)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown top-level keys: {sorted(unknown)}")
    app = AppConfig(
        robot=_build(RobotConfig, data.get("robot")),
        leader=_build(RobotConfig, data.get("leader")),
        run=_build(RunConfig, data.get("run")),
        record=_build(RecordConfig, data.get("record")),
        replay=_build(ReplayConfig, data.get("replay")),
        eval=_build(EvalConfig, data.get("eval")),
    )
    teleop = data.get("teleop")
    if teleop is not None:
        app = dataclasses.replace(app, teleop=_build(TeleopConfig, teleop))
    return app
