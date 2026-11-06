"""Training interfaces (model-agnostic). Actual algorithms live in user code."""

from soar.train.dataset import EpisodeWindowDataset, Window, split_episodes
from soar.train.spec import DatasetSpec
from soar.train.stats import JointStats, NormStats, compute_norm_stats
from soar.train.trainer import (
    CheckpointError,
    TrainablePolicy,
    TrainConfig,
    TrainingPipeline,
    TrainStats,
)

__all__ = [
    "CheckpointError",
    "DatasetSpec",
    "EpisodeWindowDataset",
    "JointStats",
    "NormStats",
    "TrainConfig",
    "TrainStats",
    "TrainablePolicy",
    "TrainingPipeline",
    "Window",
    "compute_norm_stats",
    "split_episodes",
]
