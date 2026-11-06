"""Training data contract: windows, language/images, specs and splits."""

from __future__ import annotations

from pathlib import Path

import pytest

from soar.data.reader import EpisodeDataset
from soar.train.dataset import EpisodeWindowDataset, split_episodes


def test_windows_respect_horizons(dataset_root: Path) -> None:
    dataset = EpisodeWindowDataset(dataset_root, observation_horizon=2, action_horizon=1)
    assert len(dataset) == 12  # 3 episodes * (5 frames - 2 + 1)
    window = next(iter(dataset))
    assert len(window.observations) == 2
    assert len(window.actions) == 1
    assert window.observations[1][0] == window.actions[0][0]
    assert window.start_index == 0


def test_windows_can_filter_success_and_task(dataset_root: Path) -> None:
    dataset = EpisodeWindowDataset(dataset_root, task="pick", success_only=True)
    assert dataset.episode_ids == ["ep-0001"]
    assert len(dataset) == 4


def test_stride_changes_window_count(dataset_root: Path) -> None:
    dataset = EpisodeWindowDataset(dataset_root, stride=2)
    assert len(dataset) == 6


def test_split_is_episode_level(dataset_root: Path) -> None:
    dataset = EpisodeDataset(dataset_root)
    train, validation = split_episodes(dataset, validation_fraction=0.34)
    assert len(train) + len(validation) == 3
    assert set(train).isdisjoint(validation)


def test_training_pipeline_protocol(tmp_path: Path) -> None:
    from soar.policy import PolicySpec
    from soar.train import (
        DatasetSpec,
        JointStats,
        NormStats,
        TrainablePolicy,
        TrainConfig,
        TrainingPipeline,
        TrainStats,
        Window,
    )

    class DummyTrainable:
        name = "dummy"

        def __init__(self) -> None:
            self.seen = 0

        def fit(self, dataset, config: TrainConfig) -> TrainStats:
            self.seen = len(list(dataset))
            return TrainStats(epochs=config.epochs, steps=self.seen, loss=0.5)

        def save(self, path) -> Path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("dummy")
            return target

        spec = PolicySpec(name="dummy", action_dim=6, action_horizon=4, language=True)

        def reset(self) -> None: ...
        def select_action(self, observation): ...
        def close(self) -> None: ...

    policy = DummyTrainable()
    assert isinstance(policy, TrainablePolicy)
    dataset = [Window((), (), "ep", 0, 0)]
    spec = DatasetSpec(
        joints=("j0", "j1"),
        observation_horizon=2,
        action_horizon=4,
        image_keys=("wrist",),
        language=True,
        episodes=1,
        frames=1,
    )
    stats = NormStats(
        state=JointStats((0.0, 0.0), (1.0, 1.0), 1),
        action=JointStats((0.0, 0.0), (1.0, 1.0), 1),
    )
    report = TrainingPipeline(
        policy,
        dataset,
        TrainConfig(epochs=3, checkpoint_dir=str(tmp_path)),
        dataset_spec=spec,
        norm_stats=stats,
    ).run()
    assert report["steps"] == 1
    assert (tmp_path / "train.json").is_file()
    assert (tmp_path / "dataset_spec.json").is_file()
    assert (tmp_path / "norm_stats.json").is_file()
    assert (tmp_path / "policy_spec.json").is_file()
    assert (tmp_path / "policy").read_text() == "dummy"


def test_windows_carry_language_and_camera_paths(tmp_path: Path) -> None:
    from helpers import make_episode

    from soar.train.dataset import EpisodeWindowDataset

    make_episode(
        tmp_path, episode_id="vla", task="put the cube in the bowl", frames=4, camera="wrist"
    )
    dataset = EpisodeWindowDataset(tmp_path, observation_horizon=2, action_horizon=1)
    window = next(iter(dataset))
    assert window.instruction == "put the cube in the bowl"
    assert len(window.images) == 2
    assert window.images[0]["wrist"].endswith(".jpg")
    payload = window.as_dict()
    assert payload["task"] == window.instruction
    assert payload["observation.state"]
    assert payload["observation.images.wrist"]

    spec = dataset.spec()
    assert spec.language is True
    assert spec.image_keys == ("wrist",)
    assert spec.state_dim == 6


def test_normalisation_stats_roundtrip(dataset_root: Path, tmp_path: Path) -> None:
    from soar.train.stats import NormStats, compute_norm_stats

    stats = compute_norm_stats(dataset_root)
    assert stats.state.count == 15
    assert len(stats.state.mean) == 6
    normalized = stats.state.normalize(stats.state.mean)
    assert normalized == pytest.approx((0.0,) * 6)
    assert stats.state.denormalize(normalized) == pytest.approx(stats.state.mean)

    path = stats.save(tmp_path / "norm.json")
    loaded = NormStats.load(path)
    assert loaded.state.mean == stats.state.mean
    assert loaded.action.std == stats.action.std
