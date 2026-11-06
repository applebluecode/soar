"""The single execution loop used by teleoperation, replay and evaluation.

Responsibilities are deliberately narrow:

    read state (pysoarm) -> build Observation -> policy -> write goal (pysoarm)
                          -> enqueue Frame + images (async recorder)

Timing, deadline accounting and error policy are delegated to
:meth:`pysoarm.SOArm.run_control_loop`; the recorder never blocks the loop. Everything is
injectable (clock, wall clock, camera rig, recorder) so the path is testable without hardware.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pysoarm import ArmState, LoopStats

from soar.config import RunConfig
from soar.data.cameras import CameraRig, NullCameraRig
from soar.data.recorder import Recorder
from soar.data.schema import EpisodeMeta, Frame
from soar.errors import PolicyError
from soar.hardware.arm import Arm
from soar.policy.base import Policy, PolicySpec, get_policy_spec, normalize_policy_output
from soar.types import ActionChunk, Observation

__all__ = ["PolicyRunner", "RunResult"]


@dataclass(frozen=True, slots=True)
class RunResult:
    """Outcome of one rollout.

    Args:
        stats: pysoarm loop statistics (ticks/overruns/errors).
        frames: Frames handed to the recorder.
        dropped_frames: Frames dropped by the recorder queue.
        latency_s: Per-tick read+policy+write latency.
        episode_path: Published episode directory when recording.
        completed: Whether the loop ended by duration rather than an exception.
        policy_name: Policy used (for reports).
        started_host_ns, ended_host_ns: Wall-clock bounds.
        policy_spec: Capability descriptor declared by the policy (if any).
    """

    stats: LoopStats
    frames: int
    dropped_frames: int
    latency_s: tuple[float, ...]
    episode_path: str | None
    completed: bool
    policy_name: str
    started_host_ns: int
    ended_host_ns: int
    policy_spec: PolicySpec | None = None

    @property
    def mean_latency_s(self) -> float:
        return statistics.fmean(self.latency_s) if self.latency_s else 0.0

    @property
    def p95_latency_s(self) -> float:
        if not self.latency_s:
            return 0.0
        ordered = sorted(self.latency_s)
        index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
        return ordered[index]

    @property
    def duration_s(self) -> float:
        return (self.ended_host_ns - self.started_host_ns) / 1e9

    def as_dict(self) -> dict[str, Any]:
        payload = self.stats.as_dict()
        payload.update(
            {
                "mean_latency_s": self.mean_latency_s,
                "p95_latency_s": self.p95_latency_s,
                "max_latency_s": max(self.latency_s, default=0.0),
                "frames": self.frames,
                "dropped_frames": self.dropped_frames,
                "mean_latency_ms": round(self.mean_latency_s * 1e3, 3),
                "p95_latency_ms": round(self.p95_latency_s * 1e3, 3),
                "episode_path": self.episode_path,
                "completed": self.completed,
                "policy": self.policy_name,
                "duration_s": round(self.duration_s, 3),
            }
        )
        return payload


class PolicyRunner:
    """Run a policy on one arm at a fixed rate.

    Args:
        arm: Connected follower arm.
        policy: Any object satisfying :class:`soar.policy.base.Policy`.
        run: Loop configuration.
        recorder: Optional episode recorder; started and closed by the runner.
        cameras: Camera rig sampled once per tick (``NullCameraRig`` by default).
        clock: Monotonic clock for latency measurement.
        wall_clock: Host clock used for dataset timestamps.
        on_frame: Optional hook ``(frame, result)`` for live metrics/logging.
        instruction: Task prompt for VLA/WAM policies; defaults to ``run.instruction``.
    """

    def __init__(
        self,
        arm: Arm,
        policy: Policy,
        run: RunConfig,
        *,
        recorder: Recorder | None = None,
        cameras: CameraRig | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], int] = time.time_ns,
        on_frame: Callable[[Frame], None] | None = None,
        instruction: str | None = None,
    ) -> None:
        self.arm = arm
        self.policy = policy
        self.run_config = run
        self.recorder = recorder
        self.cameras = cameras or NullCameraRig()
        self._clock = clock
        self._wall_clock = wall_clock
        self._on_frame = on_frame
        self.instruction = run.instruction if instruction is None else instruction
        self._frame_index = 0
        self._latencies: list[float] = []
        self._chunk: list[tuple[float, ...]] = []
        self._chunk_pos = 0
        self._chunk_meta: Mapping[str, Any] = {}
        self._policy_spec: PolicySpec | None = None

    def _tick(self, state: ArmState, elapsed: float) -> None:
        del elapsed
        started = self._clock()
        images = self.cameras.read()
        observation = Observation(
            q_deg=tuple(state.q_deg),
            t_host_ns=self._wall_clock(),
            t_mono_s=state.t,
            images=images,
            instruction=self.instruction,
            meta={"health_age_s": state.health_age_s},
        )
        raw: tuple[float, ...] | None = None
        chunk_index = self._chunk_pos
        if self._chunk_pos >= len(self._chunk) or self._chunk_pos >= self.run_config.replan_every:
            try:
                chunk = normalize_policy_output(
                    self.policy.select_action(observation), joints=len(state.q_deg)
                )
            except Exception as exc:
                if self.recorder is not None:
                    self.recorder.event("policy_error", error=repr(exc))
                raise
            self._chunk = list(chunk.steps) if isinstance(chunk, ActionChunk) else []
            self._chunk_pos = 0
            self._chunk_meta = dict(chunk.meta) if isinstance(chunk, ActionChunk) else {}
            chunk_index = 0
        if self._chunk:
            raw = self._chunk[self._chunk_pos]
            chunk_index = self._chunk_pos
            self._chunk_pos += 1
        applied: tuple[float, ...] | None = None
        if raw is not None:
            applied = tuple(self.arm.write_goal(raw))
        latency = self._clock() - started
        self._latencies.append(latency)
        if self.recorder is not None:
            clipped = raw is not None and applied is not None and not _close(raw, applied)
            frame = Frame(
                frame_index=self._frame_index,
                t_host_ns=observation.t_host_ns,
                t_mono_s=state.t,
                q_deg=tuple(state.q_deg),
                q_raw_deg=raw,
                q_cmd_deg=applied,
                latency_s=latency,
                clipped=clipped,
                temperature_c=tuple(state.temperature_c),
                voltage_v=tuple(state.voltage_v),
                images=dict.fromkeys(images, ""),
                meta={
                    "policy": self.policy.name,
                    "instruction": self.instruction,
                    "chunk_index": chunk_index,
                    "chunk_meta": dict(self._chunk_meta),
                    "health_age_s": state.health_age_s,
                },
            )
            self.recorder.append(frame, images)
            if self._on_frame is not None:
                self._on_frame(frame)
        self._frame_index += 1

    def run(self, *, duration: float | None = None, success: bool | None = None) -> RunResult:
        """Execute the loop; recording, if any, is finalized before returning.

        Raises:
            pysoarm.PysoarmError: Hardware failure according to ``on_error``.
            PolicyError: The policy returned an invalid action.
            RecorderError: The writer failed.
        """
        started_ns = self._wall_clock()
        state = getattr(self.arm, "state", None)
        joints = len(state.q_deg) if state is not None else None
        self._policy_spec = get_policy_spec(self.policy, action_dim=joints)
        if joints is not None and self._policy_spec.action_dim not in (None, joints):
            raise PolicyError(
                f"policy {self.policy.name!r} expects {self._policy_spec.action_dim} joints, "
                f"arm has {joints}"
            )
        missing = set(self._policy_spec.image_keys) - set(self.cameras.names)
        if missing:
            raise PolicyError(
                f"policy {self.policy.name!r} needs cameras {sorted(missing)}, "
                f"rig has {sorted(self.cameras.names)}"
            )
        if self.recorder is not None:
            self.recorder.start()
        self.policy.reset()
        self._chunk = []
        self._chunk_pos = 0
        self._chunk_meta = {}
        self.arm.set_speed_scale(self.run_config.speed_scale)
        completed = False
        try:
            stats = self.arm.run_control_loop(
                self._tick,
                hz=self.run_config.hz,
                duration=self.run_config.duration if duration is None else duration,
                read_state_every=self.run_config.read_state_every,
                health_period_s=self.run_config.health_period_s,
                on_error=self.run_config.on_error,  # type: ignore[arg-type]
                max_consecutive_errors=self.run_config.max_consecutive_errors,
            )
            completed = True
        finally:
            episode_path = None
            if self.recorder is not None:
                path = self.recorder.close(success=success if completed else False)
                episode_path = str(path)
        return RunResult(
            stats=stats,
            frames=self.recorder.frame_count if self.recorder is not None else 0,
            dropped_frames=self.recorder.dropped_frames if self.recorder is not None else 0,
            latency_s=tuple(self._latencies),
            episode_path=episode_path,
            completed=completed,
            policy_name=self.policy.name,
            started_host_ns=started_ns,
            ended_host_ns=self._wall_clock(),
            policy_spec=self._policy_spec,
        )


def _close(a: tuple[float, ...], b: tuple[float, ...], tol: float = 1e-9) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


def episode_meta_for(
    *,
    task: str,
    joints: tuple[str, ...],
    robot: Mapping[str, Any],
    fps: float,
    camera_names: tuple[str, ...],
    calibration_sha256: str | None = None,
    tags: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> tuple[str, EpisodeMeta]:
    """Helper used by the factory: create a unique episode id and metadata."""
    import uuid

    from soar.data.schema import new_episode_meta

    stamp = time.strftime("%Y%m%d-%H%M%S")
    episode_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
    meta = new_episode_meta(
        episode_id=episode_id,
        task=task,
        joints=joints,
        robot=robot,
        fps=fps,
        camera_names=camera_names,
        calibration_sha256=calibration_sha256,
        tags=tags,
        extra=extra,
    )
    return episode_id, meta


def recorder_for(
    *,
    root: str | Path,
    task: str,
    arm: Any,
    record_config: Any,
    calibration_sha256: str | None = None,
    cameras: CameraRig | None = None,
) -> Recorder:
    """Build a recorder from config + connected arm (metadata included)."""
    from soar.hardware.arm import arm_info

    _episode_id, meta = episode_meta_for(
        task=task,
        joints=tuple(arm.model.joints),
        robot=arm_info(arm),
        fps=record_config.fps,
        camera_names=cameras.names if cameras is not None else (),
        calibration_sha256=calibration_sha256,
    )
    return Recorder(
        root=root,
        meta=meta,
        writer=record_config.writer,
        queue_size=record_config.queue_size,
        flush_interval_s=record_config.flush_interval_s,
    )


__all__ = ["PolicyRunner", "RunResult", "episode_meta_for", "recorder_for"]
