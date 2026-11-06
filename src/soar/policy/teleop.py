"""Teleoperation as a policy: the leader arm is the "policy".

This is the key composition that keeps the framework policy-agnostic: teleoperation,
replay, scripted baselines and learned models all plug into the same
:class:`soar.control.runner.PolicyRunner`.
"""

from __future__ import annotations

from pysoarm.exceptions import CommunicationError

from soar.errors import PolicyError
from soar.hardware.arm import Arm
from soar.policy.base import PolicyOutput
from soar.types import Action, Observation

__all__ = ["TeleopPolicy"]


class TeleopPolicy:
    """Mirror a leader arm onto the follower.

    Args:
        leader: Connected read-only arm (``role="leader"``).
        max_failures: Consecutive leader read failures before the rollout is aborted.
            A single dropped frame holds the last goal instead of ending the session.
    """

    name = "teleop"

    def __init__(self, leader: Arm, *, max_failures: int = 10) -> None:
        if getattr(leader, "role", "leader") != "leader":
            raise PolicyError("TeleopPolicy needs an arm with role='leader'")
        self._leader = leader
        self._max_failures = max_failures
        self._failures = 0
        self._last: tuple[float, ...] | None = None

    def reset(self) -> None:
        self._failures = 0
        self._last = None

    def select_action(self, observation: Observation) -> PolicyOutput:
        try:
            state = self._leader.read_once(health=False)
        except CommunicationError as exc:
            self._failures += 1
            if self._failures >= self._max_failures:
                raise PolicyError(
                    f"leader read failed {self._failures} times in a row: {exc}"
                ) from exc
            return None  # hold the follower at its last goal
        self._failures = 0
        self._last = tuple(state.q_deg)
        return Action(
            q_deg=self._last,
            meta={"policy": self.name, "leader_t": state.t},
        )

    def close(self) -> None:
        return None
