"""Per-joint normalisation statistics for behaviour cloning and VLA training.

Why in the framework: every mainstream policy (ACT, Diffusion Policy, pi0, WAM) needs the
same state/action mean and std, and every deployment needs the *training-time* values. Keeping
them next to the checkpoint removes the classic train/serve mismatch.

Pure Python on purpose; no numpy/torch import.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from soar.data.reader import EpisodeDataset
from soar.errors import DataError

__all__ = ["JointStats", "NormStats", "compute_norm_stats"]


@dataclass(frozen=True, slots=True)
class JointStats:
    """Mean/std of one vector space (state or action)."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    count: int

    def normalize(self, values: Sequence[float]) -> tuple[float, ...]:
        return tuple((v - m) / s for v, m, s in zip(values, self.mean, self.std, strict=True))

    def denormalize(self, values: Sequence[float]) -> tuple[float, ...]:
        return tuple(v * s + m for v, m, s in zip(values, self.mean, self.std, strict=True))

    def as_dict(self) -> dict[str, object]:
        return {"mean": list(self.mean), "std": list(self.std), "count": self.count}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> JointStats:
        return cls(
            mean=tuple(float(v) for v in data["mean"]),  # type: ignore[union-attr]
            std=tuple(float(v) for v in data["std"]),  # type: ignore[union-attr]
            count=int(data.get("count", 0)),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class NormStats:
    """State/action normalisation for one dataset."""

    state: JointStats
    action: JointStats

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"state": self.state.as_dict(), "action": self.action.as_dict()}, indent=2),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> NormStats:
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return cls(
            state=JointStats.from_dict(data["state"]),
            action=JointStats.from_dict(data["action"]),
        )

    def as_dict(self) -> dict[str, object]:
        return {"state": self.state.as_dict(), "action": self.action.as_dict()}


def _finalise(sums: list[float], squares: list[float], count: int) -> JointStats:
    if count == 0:
        raise DataError("cannot compute normalisation on an empty dataset")
    mean = [s / count for s in sums]
    std = []
    for total, square in zip(sums, squares, strict=True):
        variance = max(0.0, square / count - (total / count) ** 2)
        std.append(math.sqrt(variance) or 1.0)
    return JointStats(mean=tuple(mean), std=tuple(std), count=count)


def compute_norm_stats(
    dataset: str | Path | EpisodeDataset,
    *,
    task: str | None = None,
    success: bool | None = None,
    use_raw_actions: bool = False,
) -> NormStats:
    """Compute state/action mean and std over every frame of a dataset.

    Args:
        dataset: Dataset root or an :class:`EpisodeDataset`.
        task: Optional task filter.
        success: Optional success filter.
        use_raw_actions: Use the policy's raw action instead of the guarded goal.

    Raises:
        DataError: No frames found.
    """
    episodes = (
        dataset
        if isinstance(dataset, EpisodeDataset)
        else EpisodeDataset(dataset, task=task, success=success)
    )
    state_sums: list[float] | None = None
    state_squares: list[float] | None = None
    action_sums: list[float] | None = None
    action_squares: list[float] | None = None
    count = 0
    for episode in episodes:
        for frame in episode:
            state = list(frame.q_deg)
            action = frame.q_raw_deg if use_raw_actions else frame.q_cmd_deg
            if action is None:
                action = frame.q_deg
            action = list(action)
            if len(state) != len(action):
                raise DataError(f"state/action dimension mismatch in {episode.path}")
            if state_sums is None:
                state_sums = [0.0] * len(state)
                state_squares = [0.0] * len(state)
                action_sums = [0.0] * len(action)
                action_squares = [0.0] * len(action)
            if len(state) != len(state_sums) or len(action) != len(action_sums or []):
                raise DataError("inconsistent joint dimension across episodes")
            for i, value in enumerate(state):
                state_sums[i] += value  # type: ignore[index]
                state_squares[i] += value * value  # type: ignore[index]
            for i, value in enumerate(action):
                action_sums[i] += value  # type: ignore[index]
                action_squares[i] += value * value  # type: ignore[index]
            count += 1
    if state_sums is None or action_sums is None:
        raise DataError("cannot compute normalisation on an empty dataset")
    return NormStats(
        state=_finalise(state_sums, state_squares or [], count),
        action=_finalise(action_sums, action_squares or [], count),
    )
