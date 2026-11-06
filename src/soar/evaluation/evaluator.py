"""Policy evaluation: N rollouts, success predicate, metrics, optional recording.

The evaluator knows nothing about policies, hardware or datasets; it receives factories:

    evaluator = Evaluator(policy, make_arm, run=RunConfig(...), config=EvalConfig(...),
                          recorder_factory=...)
    report = evaluator.run()

This is what makes the same harness usable for a scripted baseline, a replay sanity check,
an ACT checkpoint and a VLA served over HTTP.
"""

from __future__ import annotations

import importlib
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from soar.config import EvalConfig, RunConfig
from soar.control.runner import PolicyRunner, RunResult
from soar.data.cameras import CameraRig, NullCameraRig
from soar.data.recorder import Recorder
from soar.hardware.arm import Arm
from soar.policy.base import Policy

__all__ = ["EpisodeOutcome", "EvalReport", "Evaluator", "load_callable"]

SuccessFn = Callable[[RunResult], bool]
ArmFactory = Callable[[], Arm]
RecorderFactory = Callable[[Arm, CameraRig], Recorder]
CameraFactory = Callable[[], CameraRig]


def load_callable(spec: str) -> Callable[..., Any]:
    """Import ``"module:function"`` (used by ``success_fn`` in configs/CLI)."""
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"expected 'module:function', got {spec!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)  # type: ignore[no-any-return]
    except AttributeError as exc:
        raise ValueError(f"{module_name} has no attribute {attr!r}") from exc


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """One evaluated episode."""

    index: int
    success: bool | None
    result: RunResult | None
    error: str | None = None
    episode_path: str | None = None

    @property
    def duration_s(self) -> float:
        return self.result.duration_s if self.result is not None else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "success": self.success,
            "duration_s": round(self.duration_s, 3),
            "frames": self.result.frames if self.result else 0,
            "mean_latency_ms": round(self.result.mean_latency_s * 1e3, 3) if self.result else 0.0,
            "overruns": self.result.stats.overruns if self.result else 0,
            "errors": self.result.stats.errors if self.result else 0,
            "episode_path": self.episode_path,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregated evaluation result."""

    policy: str
    outcomes: tuple[EpisodeOutcome, ...]
    started_ns: int
    ended_ns: int
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def episodes(self) -> int:
        return len(self.outcomes)

    @property
    def successes(self) -> int:
        return sum(1 for o in self.outcomes if o.success is True)

    @property
    def evaluated(self) -> int:
        return sum(1 for o in self.outcomes if o.success is not None)

    @property
    def success_rate(self) -> float | None:
        return self.successes / self.evaluated if self.evaluated else None

    def as_dict(self) -> dict[str, Any]:
        latencies = [o.result.mean_latency_s for o in self.outcomes if o.result is not None]
        return {
            "policy": self.policy,
            "episodes": self.episodes,
            "successes": self.successes,
            "evaluated": self.evaluated,
            "success_rate": round(self.success_rate, 4) if self.success_rate is not None else None,
            "mean_duration_s": round(statistics.fmean([o.duration_s for o in self.outcomes]), 3)
            if self.outcomes
            else 0.0,
            "mean_latency_ms": round(statistics.fmean(latencies) * 1e3, 3) if latencies else 0.0,
            "total_overruns": sum(
                o.result.stats.overruns for o in self.outcomes if o.result is not None
            ),
            "failures": [
                {"index": o.index, "error": o.error} for o in self.outcomes if o.error is not None
            ],
            "outcomes": [o.as_dict() for o in self.outcomes],
            "wall_time_s": round((self.ended_ns - self.started_ns) / 1e9, 3),
            **self.extra,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return target


class Evaluator:
    """Run a policy for ``config.episodes`` rollouts.

    Args:
        policy: Policy instance (``reset`` is called per episode by the runner).
        arm_factory: Returns a connected follower; called once per episode.
        config: Episode count, timeout, success predicate, recording options.
        run: Loop settings (hz, error policy, speed scale).
        recorder_factory: Optional ``(arm, cameras) -> Recorder``; episodes are written
            when provided.
        camera_factory: Optional fresh camera rig per episode (closed after the episode).
    """

    def __init__(
        self,
        policy: Policy,
        arm_factory: ArmFactory,
        *,
        config: EvalConfig,
        run: RunConfig,
        recorder_factory: RecorderFactory | None = None,
        camera_factory: CameraFactory | None = None,
    ) -> None:
        if config.episodes < 1:
            raise ValueError("eval.episodes must be >= 1")
        self.policy = policy
        self._arm_factory = arm_factory
        self._config = config
        self._run = run
        self._recorder_factory = recorder_factory
        self._camera_factory = camera_factory
        self._success: SuccessFn | None = (
            load_callable(config.success_fn) if config.success_fn else None
        )

    def run(self) -> EvalReport:
        started = time.time_ns()
        outcomes: list[EpisodeOutcome] = []
        for index in range(self._config.episodes):
            arm: Arm | None = None
            cameras: CameraRig = (
                self._camera_factory() if self._camera_factory is not None else NullCameraRig()
            )
            recorder = None
            result: RunResult | None = None
            error: str | None = None
            success: bool | None = None
            try:
                arm = self._arm_factory()
                recorder = self._recorder_factory(arm, cameras) if self._recorder_factory else None
                runner = PolicyRunner(
                    arm,
                    self.policy,
                    self._run,
                    recorder=recorder,
                    cameras=cameras,
                    instruction=self._config.instruction or self._config.task,
                )
                result = runner.run(duration=self._config.max_duration_s)
                if self._success is not None:
                    success = bool(self._success(result))
            except Exception as exc:  # one failed episode must not kill the report
                error = f"{type(exc).__name__}: {exc}"
                success = False
            finally:
                cameras.close()
                if arm is not None:
                    arm.stop()
            outcomes.append(
                EpisodeOutcome(
                    index=index,
                    success=success,
                    result=result,
                    error=error,
                    episode_path=recorder.path.as_posix()
                    if recorder is not None and result
                    else None,
                )
            )
        return EvalReport(
            policy=self.policy.name,
            outcomes=tuple(outcomes),
            started_ns=started,
            ended_ns=time.time_ns(),
        )
