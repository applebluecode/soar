"""Immutable values exchanged between the control loop, policies and the recorder.

``Observation`` is what a policy sees; ``Action`` is what it returns. Keeping both as plain
frozen dataclasses with tuples (not numpy arrays) makes the boundary serialisable, hashable
and independent of any ML framework.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Action", "ActionChunk", "ImageFrame", "Observation"]


@dataclass(frozen=True, slots=True)
class ImageFrame:
    """One camera frame captured for an observation or an episode.

    Args:
        name: Camera name (``"wrist"``, ``"front"``, ...).
        data: Encoded bytes (JPEG/PNG) as returned by the camera backend.
        ext: File extension including the dot.
        t_host_ns: Host wall-clock timestamp (``time.time_ns``) of the capture.
    """

    name: str
    data: bytes
    ext: str = ".jpg"
    t_host_ns: int = 0


@dataclass(frozen=True, slots=True)
class Observation:
    """A single, consistent view of the robot handed to a policy.

    Args:
        q_deg: Measured joint positions in degrees, in the model's joint order.
        t_host_ns: Host wall-clock timestamp (``time.time_ns``) of the read.
        t_mono_s: Monotonic timestamp from ``ArmState.t``; use this for control timing.
        images: Camera name -> frame captured closest to ``t_host_ns``.
        instruction: Natural-language task/instruction (VLA/WAM prompt); empty for
            instruction-free teleoperation.
        meta: Free-form context (episode id, counters, extra sensors) for policies/logging.
    """

    q_deg: tuple[float, ...]
    t_host_ns: int
    t_mono_s: float
    images: Mapping[str, ImageFrame] = field(default_factory=dict)
    instruction: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Action:
    """A joint-space command requested by a policy.

    Only ``q_deg`` is part of the contract; ``meta`` travels with the action for logging
    (policy name, sampling step, log-probability, ...) and is never sent to the hardware.
    """

    q_deg: tuple[float, ...]
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionChunk:
    """A sequence of joint targets produced by one policy call.

    Modern imitation/VLA policies (ACT, Diffusion Policy, pi0, WAM) predict a short
    horizon instead of a single action. The runner executes the chunk at the control rate
    and re-plans every ``run.replan_every`` ticks, so no algorithm lives in the loop.

    Args:
        steps: One joint vector per control tick; must be non-empty.
        meta: Free-form policy output (log-probabilities, tokens, chunk index, ...).
    """

    steps: tuple[tuple[float, ...], ...]
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return len(self.steps)
