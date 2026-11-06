"""Training protocol: the framework owns checkpoints and reporting, not the model.

A trainable policy implements ``fit(dataset, config) -> TrainStats`` (and optionally
``save(path)``). That keeps ACT/Diffusion/VLA/RL code out of the core package while giving
experiments one place for config, dataset selection and metrics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from soar.errors import SoarError

__all__ = [
    "CheckpointError",
    "TrainConfig",
    "TrainStats",
    "TrainablePolicy",
    "TrainingPipeline",
]


class CheckpointError(SoarError):
    """A policy checkpoint could not be saved or loaded."""


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Minimal, model-agnostic training configuration.

    Args:
        epochs: Number of passes over the dataset (interpreted by the policy).
        batch_size: Batch size hint.
        learning_rate: Optimizer learning rate hint.
        checkpoint_dir: Where the policy and ``train.json`` are written.
        seed: Reproducibility seed.
        observation_horizon: Frames per observation window (VLA/ACT convention).
        action_horizon: Actions predicted per sample.
        device: Training device hint (``cuda``/``mps``/``cpu``).
        dtype: Parameter dtype hint (``float32``/``bfloat16``).
        extra: Model-specific knobs (kept out of the core schema).
    """

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    checkpoint_dir: str = "runs/train"
    seed: int = 0
    observation_horizon: int = 2
    action_horizon: int = 1
    device: str = "cpu"
    dtype: str = "float32"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainStats:
    """What a policy reports after ``fit``."""

    epochs: int
    steps: int
    loss: float
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "steps": self.steps,
            "loss": self.loss,
            "metrics": dict(self.metrics),
        }


@runtime_checkable
class TrainablePolicy(Protocol):
    """A policy that can be trained on :class:`soar.train.dataset.Window` samples."""

    name: str

    def fit(self, dataset: Any, config: TrainConfig) -> TrainStats: ...

    def save(self, path: str | Path) -> Path: ...


class TrainingPipeline:
    """Run ``policy.fit`` and persist the outcome.

    Args:
        policy: A :class:`TrainablePolicy` (not just any Policy).
        dataset: Anything iterable of windows; passed through untouched.
        config: Training configuration.
        dataset_spec: Optional :class:`soar.train.spec.DatasetSpec`; written next to the
            checkpoint for inference-side validation.
        norm_stats: Optional :class:`soar.train.stats.NormStats`; written next to the
            checkpoint so serving uses the training statistics.
    """

    def __init__(
        self,
        policy: TrainablePolicy,
        dataset: Any,
        config: TrainConfig,
        *,
        dataset_spec: Any = None,
        norm_stats: Any = None,
    ) -> None:
        if not isinstance(policy, TrainablePolicy):
            raise CheckpointError(
                f"{type(policy).__name__} does not implement fit/save; see TrainablePolicy"
            )
        self.policy = policy
        self.dataset = dataset
        self.config = config
        self.dataset_spec = dataset_spec
        self.norm_stats = norm_stats

    def run(self) -> dict[str, Any]:
        started = time.time_ns()
        stats = self.policy.fit(self.dataset, self.config)
        out = Path(self.config.checkpoint_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = self.policy.save(out / "policy")
        if self.dataset_spec is not None:
            (out / "dataset_spec.json").write_text(
                json.dumps(self.dataset_spec.as_dict(), indent=2), encoding="utf-8"
            )
        if self.norm_stats is not None:
            self.norm_stats.save(out / "norm_stats.json")
        spec = getattr(self.policy, "spec", None)
        if spec is not None and hasattr(spec, "as_dict"):
            (out / "policy_spec.json").write_text(
                json.dumps(spec.as_dict(), indent=2), encoding="utf-8"
            )
        report = {
            **stats.as_dict(),
            "policy": self.policy.name,
            "checkpoint": str(checkpoint),
            "wall_time_s": round((time.time_ns() - started) / 1e9, 3),
            "policy_spec": spec.as_dict()
            if spec is not None and hasattr(spec, "as_dict")
            else None,
            "dataset_spec": self.dataset_spec.as_dict() if self.dataset_spec is not None else None,
            "config": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "seed": self.config.seed,
                "observation_horizon": self.config.observation_horizon,
                "action_horizon": self.config.action_horizon,
                "device": self.config.device,
                "dtype": self.config.dtype,
                **self.config.extra,
            },
        }
        (out / "train.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
