"""Small reference policies used in tests, demos and baselines.

They are intentionally not "algorithms": everything trainable (ACT, Diffusion, VLA, RL)
belongs in user code or a separate package and only has to implement
:class:`soar.policy.base.Policy`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

from soar.errors import PolicyError
from soar.policy.base import PolicyOutput, PolicySpec
from soar.types import Action, ActionChunk, Observation

__all__ = [
    "CallablePolicy",
    "ChunkedHoldPolicy",
    "ConstantPolicy",
    "HoldPolicy",
    "RandomPolicy",
    "ScriptedPolicy",
]


class ConstantPolicy:
    """Always return the same joint target."""

    name = "constant"

    def __init__(self, q_deg: Sequence[float] = (0.0,) * 6) -> None:
        self.q_deg = tuple(float(v) for v in q_deg)

    def reset(self) -> None:
        return None

    def select_action(self, observation: Observation) -> Action:
        return Action(q_deg=self.q_deg, meta={"policy": self.name})

    def close(self) -> None:
        return None


class HoldPolicy:
    """Hold the measured pose (a safe default / policy fallback)."""

    name = "hold"

    def reset(self) -> None:
        return None

    def select_action(self, observation: Observation) -> Action:
        return Action(q_deg=observation.q_deg, meta={"policy": self.name})

    def close(self) -> None:
        return None


class ChunkedHoldPolicy:
    """Hold the measured pose for a fixed action horizon.

    A minimal example of the chunked interface used by ACT/Diffusion/VLA policies: one
    inference call returns several steps, and ``run.replan_every`` decides how many are
    executed before the next call.
    """

    name = "chunked"

    def __init__(self, horizon: int = 8) -> None:
        if horizon < 1:
            raise PolicyError("chunk horizon must be >= 1")
        self.horizon = horizon
        self.spec = PolicySpec(name=self.name, action_horizon=horizon)

    def reset(self) -> None:
        return None

    def select_action(self, observation: Observation) -> ActionChunk:
        return ActionChunk(
            steps=tuple(observation.q_deg for _ in range(self.horizon)),
            meta={"policy": self.name, "horizon": self.horizon},
        )

    def close(self) -> None:
        return None


class CallablePolicy:
    """Adapt a function ``f(observation, step) -> result`` to the Policy protocol."""

    name = "callable"

    def __init__(
        self,
        fn: Callable[[Observation, int], PolicyOutput],
        *,
        name: str = "callable",
    ) -> None:
        self._fn = fn
        self.name = name
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def select_action(self, observation: Observation) -> PolicyOutput:
        result = self._fn(observation, self._step)
        self._step += 1
        return result

    def close(self) -> None:
        return None


class ScriptedPolicy:
    """Play a fixed list of joint targets, then hold the last one."""

    name = "scripted"

    def __init__(self, trajectory: Sequence[Sequence[float]]) -> None:
        if not trajectory:
            raise PolicyError("scripted policy needs at least one action")
        self._trajectory = [tuple(float(v) for v in q) for q in trajectory]
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def select_action(self, observation: Observation) -> Action:
        index = min(self._index, len(self._trajectory) - 1)
        self._index += 1
        return Action(q_deg=self._trajectory[index], meta={"policy": self.name, "step": index})

    def close(self) -> None:
        return None


class RandomPolicy:
    """Uniform random targets inside ``[low, high]``; deterministic per seed."""

    name = "random"

    def __init__(
        self,
        low: Sequence[float] = (-30.0,) * 6,
        high: Sequence[float] = (30.0,) * 6,
        *,
        seed: int = 0,
    ) -> None:
        if len(low) != len(high):
            raise PolicyError("low/high must have the same length")
        self._low = tuple(float(v) for v in low)
        self._high = tuple(float(v) for v in high)
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def select_action(self, observation: Observation) -> Action:
        q = tuple(self._rng.uniform(lo, hi) for lo, hi in zip(self._low, self._high, strict=True))
        return Action(q_deg=q, meta={"policy": self.name})

    def close(self) -> None:
        return None


def sinusoid_policy(
    amplitude_deg: float = 20.0,
    period_s: float = 4.0,
    joint: int = 0,
    joints: int = 6,
) -> CallablePolicy:
    """Convenience policy: one joint follows a sine wave around the initial pose."""
    origin: dict[str, float] = {}

    def fn(observation: Observation, step: int) -> tuple[float, ...]:
        if "q0" not in origin:
            origin["q0"] = observation.q_deg[joint]
        phase = 2 * math.pi * observation.t_mono_s / period_s
        q = list(observation.q_deg)
        q[joint] = origin["q0"] + amplitude_deg * math.sin(phase)
        return tuple(q[:joints])

    return CallablePolicy(fn, name="sinusoid")
