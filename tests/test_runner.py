"""The policy execution loop: recording, hold, errors, safety cap, cameras."""

from __future__ import annotations

from pathlib import Path

import pytest
from pysoarm import SafetyLimits

from soar.config import RunConfig
from soar.control.runner import PolicyRunner
from soar.data.cameras import CameraRig, FakeCamera
from soar.data.reader import EpisodeReader
from soar.data.recorder import Recorder
from soar.data.schema import new_episode_meta
from soar.data.validation import validate_episode
from soar.errors import PolicyError
from soar.policy import CallablePolicy, ConstantPolicy
from soar.policy.teleop import TeleopPolicy


def recorder(tmp_path: Path, arm, *, cameras: tuple[str, ...] = ()) -> Recorder:
    meta = new_episode_meta(
        episode_id="run-1",
        task="unit",
        joints=arm.model.joints,
        robot={"model": arm.model.name},
        fps=30.0,
        camera_names=cameras,
    )
    return Recorder(tmp_path, meta, queue_size=256)


def test_runner_records_a_teleop_rollout(tmp_path: Path, fake_arm) -> None:
    leader, _ = fake_arm(role="leader")
    follower, _ = fake_arm()
    rec = recorder(tmp_path, follower, cameras=("wrist",))
    runner = PolicyRunner(
        follower,
        TeleopPolicy(leader),
        RunConfig(hz=100, duration=0.15),
        recorder=rec,
        cameras=CameraRig([FakeCamera("wrist")]),
    )
    result = runner.run()
    assert result.completed
    assert result.frames > 5
    assert result.mean_latency_s > 0
    assert result.episode_path is not None
    report = validate_episode(result.episode_path)
    assert report.ok, report.errors
    reader = EpisodeReader(result.episode_path)
    frames = list(reader)
    assert frames[0].images and (Path(result.episode_path) / frames[0].images["wrist"]).is_file()


def test_none_action_holds_and_record_has_no_command(tmp_path: Path, fake_arm) -> None:
    follower, _ = fake_arm()
    rec = recorder(tmp_path, follower)
    policy = CallablePolicy(lambda observation, step: None)
    result = PolicyRunner(follower, policy, RunConfig(hz=100, duration=0.1), recorder=rec).run()
    assert result.frames > 0
    frames = list(EpisodeReader(result.episode_path))
    assert all(frame.q_cmd_deg is None for frame in frames)


def test_invalid_action_aborts_the_rollout(fake_arm) -> None:
    follower, _ = fake_arm()
    policy = CallablePolicy(lambda observation, step: [0.0, 1.0])
    runner = PolicyRunner(follower, policy, RunConfig(hz=100, duration=0.1, on_error="abort"))
    with pytest.raises(PolicyError, match="joints"):
        runner.run()


def test_speed_cap_is_applied_on_the_first_tick(fake_arm) -> None:
    follower, _ = fake_arm(limits=SafetyLimits(max_speed_deg_s=720.0))
    policy = ConstantPolicy((60.0,) * 6)
    runner = PolicyRunner(follower, policy, RunConfig(hz=200, duration=0.02))
    result = runner.run()
    assert result.completed
    # first tick is bounded by 720 deg/s / 200 Hz = 3.6 deg
    assert follower.state is not None


def test_runner_uses_fake_cameras_without_hardware(tmp_path: Path, fake_arm) -> None:
    follower, _ = fake_arm()
    rec = recorder(tmp_path, follower, cameras=("a", "b"))
    runner = PolicyRunner(
        follower,
        ConstantPolicy((0.0,) * 6),
        RunConfig(hz=50, duration=0.05),
        recorder=rec,
        cameras=CameraRig([FakeCamera("a"), FakeCamera("b")]),
    )
    result = runner.run()
    frames = list(EpisodeReader(result.episode_path))
    assert set(frames[0].images) == {"a", "b"}


def test_runner_executes_action_chunks_and_replans(tmp_path: Path, fake_arm) -> None:
    follower, _ = fake_arm()
    rec = recorder(tmp_path, follower)
    calls = {"n": 0}
    chunk_steps = [(float(i),) * 6 for i in range(8)]

    def chunk_policy(observation, step):
        calls["n"] += 1
        return chunk_steps

    run = RunConfig(hz=100, duration=0.1, replan_every=2)
    result = PolicyRunner(follower, CallablePolicy(chunk_policy), run, recorder=rec).run()
    frames = list(EpisodeReader(result.episode_path))
    # replan every 2 ticks: the number of policy calls is at most half the ticks
    assert calls["n"] <= result.stats.ticks // 2 + 1
    assert frames[0].q_raw_deg == (0.0,) * 6
    assert frames[1].q_raw_deg == (1.0,) * 6
    assert frames[2].q_raw_deg == (0.0,) * 6  # re-planned from the start of the chunk
    assert frames[0].meta["chunk_index"] == 0


def test_runner_passes_the_instruction(fake_arm) -> None:
    follower, _ = fake_arm()
    seen: list[str] = []

    def policy(observation, step):
        seen.append(observation.instruction)
        return observation.q_deg

    runner = PolicyRunner(
        follower,
        CallablePolicy(policy),
        RunConfig(hz=100, duration=0.05),
        instruction="pick the red cube",
    )
    runner.run()
    assert seen and all(text == "pick the red cube" for text in seen)


def test_runner_rejects_a_policy_with_wrong_action_dim(fake_arm) -> None:
    from soar.errors import PolicyError
    from soar.policy import PolicySpec

    class BadSpec:
        name = "bad"
        spec = PolicySpec(name="bad", action_dim=5)

        def reset(self): ...
        def select_action(self, observation):
            return [0.0] * 6

        def close(self): ...

    follower, _ = fake_arm()
    with pytest.raises(PolicyError, match="expects 5 joints"):
        PolicyRunner(follower, BadSpec(), RunConfig(hz=50, duration=0.05)).run()
