"""Dataset capability descriptor used by training/inference tooling.

This is the framework's answer to "what can a model expect from this dataset": joint order,
horizons, image keys, language presence and size. It is written next to checkpoints so an
inference server can validate a request without scanning the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["DatasetSpec"]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Static description of a windowed dataset.

    Args:
        joints: Joint order for state/action vectors.
        observation_horizon: States per sample.
        action_horizon: Actions per sample.
        image_keys: Camera names available in ``Window.images``.
        language: Whether an instruction string is attached to every sample.
        fps: Recording hint.
        episodes: Number of episodes used.
        frames: Total frames across episodes.
        use_raw_actions: Whether actions are policy requests instead of guarded goals.
        extra: Experiment-specific metadata.
    """

    joints: tuple[str, ...]
    observation_horizon: int
    action_horizon: int
    image_keys: tuple[str, ...] = ()
    language: bool = True
    fps: float = 30.0
    episodes: int = 0
    frames: int = 0
    use_raw_actions: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def state_dim(self) -> int:
        return len(self.joints)

    @property
    def action_dim(self) -> int:
        return len(self.joints)

    def as_dict(self) -> dict[str, Any]:
        return {
            "joints": list(self.joints),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "observation_horizon": self.observation_horizon,
            "action_horizon": self.action_horizon,
            "image_keys": list(self.image_keys),
            "language": self.language,
            "fps": self.fps,
            "episodes": self.episodes,
            "frames": self.frames,
            "use_raw_actions": self.use_raw_actions,
            "extra": dict(self.extra),
        }
