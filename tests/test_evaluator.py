"""Evaluation harness: success predicate, per-episode failures, recording."""

from __future__ import annotations

from pathlib import Path

from soar.config import EvalConfig, RunConfig
from soar.data.cameras import CameraRig, FakeCamera
from soar.data.recorder import Recorder
from soar.data.schema import new_episode_meta
from soar.evaluation.evaluator import EvalReport, Evaluator
from soar.policy import CallablePolicy, ConstantPolicy


def always_true(result) -> bool:
    return True


def always_false(result) -> bool:
    return False


def test_evaluator_runs_episodes_and_scores_them(fake_arm) -> None:
    evaluator = Evaluator(
        ConstantPolicy((0.0,) * 6),
        lambda: fake_arm()[0],
        config=EvalConfig(episodes=2, max_duration_s=0.1, success_fn=f"{__name__}:always_true"),
        run=RunConfig(hz=100),
    )
    report = evaluator.run()
    assert isinstance(report, EvalReport)
    assert report.episodes == 2
    assert report.success_rate == 1.0
    assert report.as_dict()["successes"] == 2


def test_evaluator_survives_a_failing_policy(fake_arm) -> None:
    state = {"calls": 0}

    def boom(observation, step):
        state["calls"] += 1
        raise RuntimeError("model exploded")

    evaluator = Evaluator(
        CallablePolicy(boom),
        lambda: fake_arm()[0],
        config=EvalConfig(episodes=2, max_duration_s=0.1, success_fn=f"{__name__}:always_false"),
        run=RunConfig(hz=50, on_error="abort"),
    )
    report = evaluator.run()
    assert report.episodes == 2
    assert report.success_rate == 0.0
    assert all(outcome.error and "model exploded" in outcome.error for outcome in report.outcomes)


def test_evaluator_records_episodes(tmp_path: Path, fake_arm) -> None:
    def recorder_factory(arm, cameras: CameraRig) -> Recorder:
        meta = new_episode_meta(
            episode_id=f"eval-{id(arm)}",
            task="eval",
            joints=arm.model.joints,
            robot={"model": arm.model.name},
            fps=30.0,
            camera_names=cameras.names,
        )
        return Recorder(tmp_path, meta, queue_size=64)

    evaluator = Evaluator(
        ConstantPolicy((0.0,) * 6),
        lambda: fake_arm()[0],
        config=EvalConfig(episodes=1, max_duration_s=0.1),
        run=RunConfig(hz=50),
        recorder_factory=recorder_factory,
        camera_factory=lambda: CameraRig([FakeCamera("wrist")]),
    )
    report = evaluator.run()
    assert report.outcomes[0].episode_path is not None
    assert (Path(report.outcomes[0].episode_path) / "meta.json").is_file()
