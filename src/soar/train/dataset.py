"""Policy-agnostic dataset windows for behaviour cloning.

The framework does not ship a model, a loss or a training loop: it ships the *data contract*.
Windowing lives here so every policy (ACT, Diffusion, VLA, MLP) sees the same shapes; the
actual training loop implements :class:`TrainablePolicy.fit` (usually with PyTorch).

No numpy/torch import happens here, so the dataset layer stays importable in a minimal env.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soar.data.reader import EpisodeDataset, EpisodeReader
from soar.errors import DataError
from soar.train.spec import DatasetSpec

__all__ = ["EpisodeWindowDataset", "Window", "split_episodes"]


@dataclass(frozen=True, slots=True)
class Window:
    """One training sample.

    Args:
        observations: ``observation_horizon`` state vectors (oldest first).
        actions: ``action_horizon`` guarded action vectors.
        episode_id: Source episode.
        start_index: Frame index of the oldest observation.
        t_host_ns: Wall-clock timestamp of the first observation.
        instruction: Language task label (VLA/WAM prompt).
        images: Per-observation-step camera paths, ``name -> relative path``.
    """

    observations: tuple[tuple[float, ...], ...]
    actions: tuple[tuple[float, ...], ...]
    episode_id: str
    start_index: int
    t_host_ns: int
    instruction: str = ""
    images: tuple[Mapping[str, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """LeRobot-like flat view for generic PyTorch loaders."""
        payload: dict[str, Any] = {
            "observation.state": [list(state) for state in self.observations],
            "action": [list(action) for action in self.actions],
            "task": self.instruction,
            "episode_id": self.episode_id,
            "frame_index": self.start_index,
            "timestamp_ns": self.t_host_ns,
        }
        for name in {key for step in self.images for key in step}:
            payload[f"observation.images.{name}"] = [step.get(name, "") for step in self.images]
        return payload


class EpisodeWindowDataset:
    """Sliding windows over episodes with explicit episode-level filtering.

    Args:
        root: Dataset root.
        task: Optional task filter.
        success_only: Keep only episodes labelled ``success=True``.
        observation_horizon: Number of past states in ``Window.observations``.
        action_horizon: Number of future actions in ``Window.actions``.
        stride: Frame stride between windows.
        use_raw_actions: Train on the policy's raw action instead of the guarded one.
        include_images: Attach camera paths to every observation step (needed by VLA/WAM).
        image_keys: Restrict images to these cameras (``None`` = all recorded cameras).
    """

    def __init__(
        self,
        root: str | Path,
        *,
        task: str | None = None,
        success_only: bool = False,
        observation_horizon: int = 2,
        action_horizon: int = 1,
        stride: int = 1,
        use_raw_actions: bool = False,
        include_images: bool = True,
        image_keys: Sequence[str] | None = None,
    ) -> None:
        if observation_horizon < 1 or action_horizon < 1:
            raise DataError("horizons must be >= 1")
        if stride < 1:
            raise DataError("stride must be >= 1")
        self.dataset = EpisodeDataset(root, task=task, success=True if success_only else None)
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.stride = stride
        self.use_raw_actions = use_raw_actions
        self.include_images = include_images
        self.image_keys = tuple(image_keys) if image_keys is not None else None
        self._episodes = [EpisodeReader(path) for path in self.dataset.paths]

    def __len__(self) -> int:
        total = 0
        for episode in self._episodes:
            n = episode.meta.frame_count
            needed = self.observation_horizon - 1 + self.action_horizon
            span = max(0, n - needed + 1)
            total += (span + self.stride - 1) // self.stride if span else 0
        return total

    def __iter__(self) -> Iterator[Window]:
        for episode in self._episodes:
            frames = list(episode)
            states = [tuple(frame.q_deg) for frame in frames]
            actions = [
                tuple((frame.q_raw_deg if self.use_raw_actions else frame.q_cmd_deg) or frame.q_deg)
                for frame in frames
            ]
            last = min(len(states), len(actions))
            needed = self.observation_horizon - 1 + self.action_horizon
            span = max(0, last - needed + 1)
            for start in range(0, span, self.stride):
                action_start = start + self.observation_horizon - 1
                obs_slice = frames[start : start + self.observation_horizon]
                images: tuple[Mapping[str, str], ...] = ()
                if self.include_images:
                    keys = self.image_keys or tuple(
                        sorted({name for frame in obs_slice for name in frame.images})
                    )
                    images = tuple(
                        {name: frame.images.get(name, "") for name in keys} for frame in obs_slice
                    )
                yield Window(
                    observations=tuple(states[start : start + self.observation_horizon]),
                    actions=tuple(actions[action_start : action_start + self.action_horizon]),
                    episode_id=episode.episode_id,
                    start_index=start,
                    t_host_ns=frames[start].t_host_ns,
                    instruction=episode.meta.task,
                    images=images,
                )

    @property
    def episode_ids(self) -> Sequence[str]:
        return [episode.episode_id for episode in self._episodes]

    def spec(self) -> DatasetSpec:
        """Dataset capability descriptor (joints, horizons, cameras, size)."""
        image_keys: set[str] = set(self.image_keys or ())
        frames = 0
        fps = 0.0
        for episode in self._episodes:
            frames += episode.meta.frame_count
            fps = fps or episode.meta.fps
            image_keys.update(episode.meta.camera_names)
        joints = tuple(self._episodes[0].meta.joints) if self._episodes else ()
        return DatasetSpec(
            joints=joints,
            observation_horizon=self.observation_horizon,
            action_horizon=self.action_horizon,
            image_keys=tuple(sorted(image_keys)) if self.include_images else (),
            language=True,
            fps=fps or 30.0,
            episodes=len(self._episodes),
            frames=frames,
            use_raw_actions=self.use_raw_actions,
        )


def split_episodes(
    dataset: EpisodeDataset, *, validation_fraction: float = 0.2
) -> tuple[list[str], list[str]]:
    """Episode-level train/validation split (never splits frames of one episode)."""
    if not 0.0 < validation_fraction < 1.0:
        raise DataError("validation_fraction must be in (0, 1)")
    ids = list(dataset.ids)
    cut = max(1, int(len(ids) * (1 - validation_fraction)))
    return ids[:cut], ids[cut:]
