"""The policy contract, action chunks and the capability descriptor.

A policy is deliberately tiny: reset for a new episode, map one observation to joint
commands (or ``None`` to hold). It never sees a bus, a camera backend or the filesystem, so
teleoperation, replay, scripted baselines, behaviour cloning, ACT, diffusion policies, VLAs
and world-action models are all the same kind of object to the runner.

``PolicySpec`` is optional metadata: it lets the runner validate action dimensions/horizons
and lets an experiment report which observation keys a model needs. Policies that omit it
still work (inferred from the live arm).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from soar.errors import PolicyError
from soar.types import Action, ActionChunk, Observation

__all__ = [
    "Policy",
    "PolicyOutput",
    "PolicySpec",
    "get_policy_spec",
    "normalize_action",
    "normalize_policy_output",
]

PolicyOutput = (
    Action | ActionChunk | Sequence[float] | Sequence[Sequence[float]] | Mapping[str, Any] | None
)


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """What a policy expects and produces.

    Args:
        name: Registry/display name.
        action_dim: Joint vector dimension; ``None`` means "infer from the arm".
        action_horizon: Maximum chunk length the policy may return (1 = per-tick).
        observation_keys: Fields the policy consumes (``q_deg``, ``images.wrist``,
            ``instruction``, extra sensors).
        image_keys: Camera names required in ``Observation.images``.
        language: Whether an instruction prompt is consumed.
        device: ``"cpu"``/``"cuda"``/``"remote"`` hint for reporting.
        extra: Model-specific metadata (checkpoint, dtype, tokenizer, ...).
    """

    name: str
    action_dim: int | None = None
    action_horizon: int = 1
    observation_keys: tuple[str, ...] = ("q_deg",)
    image_keys: tuple[str, ...] = ()
    language: bool = False
    device: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "observation_keys": list(self.observation_keys),
            "image_keys": list(self.image_keys),
            "language": self.language,
            "device": self.device,
            "extra": dict(self.extra),
        }


@runtime_checkable
class Policy(Protocol):
    """Everything the runner needs from a policy."""

    name: str

    def reset(self) -> None:
        """Start a new episode (clear buffers, step counters, episode-local state)."""

    def select_action(self, observation: Observation) -> PolicyOutput:
        """Return joint targets (single step or chunk), or ``None`` to hold."""

    def close(self) -> None:
        """Release policy-owned resources (models, internal cameras, ...)."""


def get_policy_spec(policy: Policy, *, action_dim: int | None = None) -> PolicySpec:
    """Read ``policy.spec``/``policy.policy_spec()`` or infer a minimal spec."""
    spec = getattr(policy, "spec", None)
    if callable(spec):
        spec = spec()
    if isinstance(spec, PolicySpec):
        return spec
    if isinstance(spec, Mapping):
        return PolicySpec(**spec)  # type: ignore[arg-type]
    factory = getattr(policy, "policy_spec", None)
    if callable(factory):
        candidate = factory()
        if isinstance(candidate, PolicySpec):
            return candidate
    return PolicySpec(name=getattr(policy, "name", type(policy).__name__), action_dim=action_dim)


def _as_step(values: Any, *, joints: int) -> tuple[float, ...]:
    # Accept numpy arrays / torch tensors without importing either: anything with
    # ``tolist`` or a length+iterator is treated as a sequence of numbers.
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
        raise PolicyError(f"action step must be a sequence, got {type(values).__name__}")
    step = tuple(float(v) for v in values)
    if len(step) != joints:
        raise PolicyError(f"policy returned {len(step)} joints, expected {joints}")
    if not all(math.isfinite(v) for v in step):
        raise PolicyError(f"policy returned non-finite values: {step}")
    return step


def normalize_policy_output(
    output: PolicyOutput,
    *,
    joints: int,
    max_horizon: int = 64,
) -> ActionChunk | None:
    """Validate a policy return value and convert it to an :class:`ActionChunk`.

    Accepted forms:

    * ``None`` — hold;
    * ``Action`` / flat sequence — one step;
    * ``ActionChunk`` / nested sequence — an action chunk;
    * ``{"actions": [[...], ...], "meta": {...}}`` — JSON-friendly remote output.

    Raises:
        PolicyError: Wrong length, non-finite value, empty chunk or unsupported type.
    """
    if output is None:
        return None
    if hasattr(output, "tolist"):
        output = output.tolist()

    meta: Mapping[str, Any] = {}
    if isinstance(output, Action):
        steps = (output.q_deg,)
        meta = output.meta
    elif isinstance(output, ActionChunk):
        steps = output.steps
        meta = output.meta
    elif isinstance(output, Mapping):
        candidate = output.get("actions", output.get("action"))
        if candidate is None:
            raise PolicyError("policy mapping output needs an 'actions'/'action' key")
        meta = output.get("meta") or {}
        chunk = normalize_policy_output(candidate, joints=joints, max_horizon=max_horizon)
        if chunk is None:
            raise PolicyError("policy mapping contained a null action")
        if meta:
            chunk = replace(chunk, meta={**chunk.meta, **dict(meta)})
        return chunk
    elif isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        if len(output) == 0:
            raise PolicyError("policy returned an empty action sequence")
        first = output[0]
        if hasattr(first, "tolist"):
            first = first.tolist()
        if isinstance(first, (int, float)):
            steps = (tuple(float(v) for v in output),)  # type: ignore[arg-type]
        else:
            steps = tuple(v for v in output)  # type: ignore[arg-type]
    else:
        raise PolicyError(
            f"policy returned {type(output).__name__}; expected Action/ActionChunk/sequence"
        )
    return _chunk_from_steps(steps, joints=joints, max_horizon=max_horizon, meta=meta)


def _chunk_from_steps(
    steps: Sequence[Any],
    *,
    joints: int,
    max_horizon: int,
    meta: Mapping[str, Any],
) -> ActionChunk:
    if not steps:
        raise PolicyError("policy returned an empty action chunk")
    if len(steps) > max_horizon:
        raise PolicyError(f"action chunk has {len(steps)} steps, max {max_horizon}")
    return ActionChunk(
        steps=tuple(_as_step(step, joints=joints) for step in steps),
        meta=dict(meta),
    )


def normalize_action(output: PolicyOutput, *, joints: int) -> Action | None:
    """Backward-compatible single-step helper (rejects chunks with >1 step)."""
    chunk = normalize_policy_output(output, joints=joints, max_horizon=1)
    if chunk is None:
        return None
    return Action(q_deg=chunk.steps[0], meta=chunk.meta)
