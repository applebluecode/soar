"""Replay a recorded episode as a policy (time-based, open loop).

The policy schedules recorded actions by the observation's host timestamp, so replay speed is
independent of the control rate. Actions are the *guarded* goals by default: what the robot
actually executed, not what a policy once requested.
"""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

from soar.data.reader import EpisodeReader
from soar.errors import PolicyError
from soar.policy.base import PolicyOutput
from soar.types import Action, Observation

__all__ = ["ReplayPolicy"]


class ReplayPolicy:
    """Open-loop replay of one episode.

    Args:
        episode: Episode directory or reader.
        speed: Time multiplier (``2.0`` plays twice as fast).
        use_raw_actions: Use the policy's raw action instead of the guarded goal.
    """

    name = "replay"

    def __init__(
        self,
        episode: str | Path | EpisodeReader,
        *,
        speed: float = 1.0,
        use_raw_actions: bool = False,
    ) -> None:
        if speed <= 0:
            raise PolicyError("replay speed must be positive")
        self.episode = episode if isinstance(episode, EpisodeReader) else EpisodeReader(episode)
        self.speed = float(speed)
        self._frames = list(self.episode)
        if not self._frames:
            raise PolicyError(f"episode has no frames: {self.episode.path}")
        self._times = [frame.t_host_ns for frame in self._frames]
        self._actions: list[tuple[float, ...]] = []
        for frame in self._frames:
            values = frame.q_raw_deg if use_raw_actions else frame.q_cmd_deg
            self._actions.append(tuple(values if values is not None else frame.q_deg))
        self._start_ns: int | None = None
        self._index = 0

    @property
    def duration_s(self) -> float:
        """Recorded duration divided by the speed multiplier."""
        span = self._times[-1] - self._times[0]
        return span / 1e9 / self.speed

    def reset(self) -> None:
        self._start_ns = None
        self._index = 0

    def _action_at(self, target_ns: int) -> tuple[float, ...]:
        index = max(0, bisect_right(self._times, target_ns) - 1)
        index = min(index, len(self._frames) - 1)
        if index >= len(self._frames) - 1:
            return self._actions[-1]
        t0, t1 = self._times[index], self._times[index + 1]
        a0, a1 = self._actions[index], self._actions[index + 1]
        if t1 <= t0:
            return a1
        alpha = (target_ns - t0) / (t1 - t0)
        return tuple(x0 + (x1 - x0) * alpha for x0, x1 in zip(a0, a1, strict=True))

    def select_action(self, observation: Observation) -> PolicyOutput:
        if self._start_ns is None:
            self._start_ns = observation.t_host_ns
            self._index = 0
        target = self._times[0] + int((observation.t_host_ns - self._start_ns) * self.speed)
        values = self._action_at(target)
        return Action(
            q_deg=values,
            meta={
                "policy": self.name,
                "episode": self.episode.episode_id,
                "t_target_ns": target,
            },
        )

    def close(self) -> None:
        return None
