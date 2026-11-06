"""Policy protocol, registry, teleop and replay."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import make_episode
from pysoarm.exceptions import WatchdogTimeout

from soar.errors import PolicyError
from soar.policy import (
    ConstantPolicy,
    HoldPolicy,
    RandomPolicy,
    ReplayPolicy,
    ScriptedPolicy,
    available_policies,
    build_policy,
    normalize_action,
    register_policy,
)
from soar.policy.teleop import TeleopPolicy
from soar.types import Action, Observation


def obs(q=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0), t_ns=1_000_000_000) -> Observation:
    return Observation(q_deg=tuple(q), t_host_ns=t_ns, t_mono_s=0.0)


def test_constant_and_hold() -> None:
    constant = ConstantPolicy((1, 2, 3, 4, 5, 6))
    assert constant.select_action(obs()).q_deg == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    hold = HoldPolicy()
    assert hold.select_action(obs()).q_deg == obs().q_deg


def test_random_is_reproducible_after_reset() -> None:
    policy = RandomPolicy((-1,) * 6, (1,) * 6, seed=7)
    first = policy.select_action(obs()).q_deg
    policy.reset()
    assert policy.select_action(obs()).q_deg == first


def test_scripted_holds_last_action() -> None:
    policy = ScriptedPolicy([[0] * 6, [1] * 6])
    assert policy.select_action(obs()).q_deg == (0.0,) * 6
    assert policy.select_action(obs()).q_deg == (1.0,) * 6
    assert policy.select_action(obs()).q_deg == (1.0,) * 6


def test_normalize_action_validation() -> None:
    assert normalize_action([0] * 6, joints=6) == Action((0.0,) * 6)
    assert normalize_action(None, joints=6) is None
    with pytest.raises(PolicyError):
        normalize_action([0] * 5, joints=6)
    with pytest.raises(PolicyError):
        normalize_action([float("nan")] * 6, joints=6)


def test_registry_contains_baselines_and_rejects_unknown() -> None:
    for name in ("constant", "hold", "random", "scripted", "sinusoid"):
        assert name in available_policies()
    assert isinstance(build_policy("constant", q_deg=[0] * 6), ConstantPolicy)
    with pytest.raises(PolicyError, match="unknown policy"):
        build_policy("nope")


def test_register_duplicate_is_an_error() -> None:
    with pytest.raises(PolicyError, match="already registered"):
        register_policy("hold")(ConstantPolicy)


def test_teleop_policy_mirrors_the_leader(fake_arm) -> None:
    leader, _ = fake_arm(role="leader")
    policy = TeleopPolicy(leader)
    action = policy.select_action(obs())
    assert action is not None
    assert len(action.q_deg) == 6


def test_teleop_policy_holds_on_transient_failure_and_aborts(fake_arm) -> None:
    leader, bus = fake_arm(role="leader")
    policy = TeleopPolicy(leader, max_failures=2)
    bus.fail_next_reads = 1
    assert policy.select_action(obs()) is None
    assert policy.select_action(obs()) is not None
    # the policy cutoff (2) fires before the arm watchdog (3)
    bus.fail_next_reads = 10
    with pytest.raises(PolicyError, match="leader read failed"):
        for _ in range(2):
            policy.select_action(obs())


def test_leader_watchdog_precedes_the_policy_cutoff(fake_arm) -> None:
    leader, bus = fake_arm(role="leader")
    policy = TeleopPolicy(leader, max_failures=10)
    bus.fail_next_reads = 10
    with pytest.raises(WatchdogTimeout, match="consecutive bus failures"):
        for _ in range(3):
            policy.select_action(obs())


def test_replay_policy_interpolates_by_time(tmp_path: Path) -> None:
    path = make_episode(tmp_path, episode_id="rep", frames=3, joints=1, step_ns=100_000_000)
    policy = ReplayPolicy(path, speed=1.0)
    assert policy.duration_s == pytest.approx(0.2)
    start = 1_000_000_000
    assert policy.select_action(Observation((0.0,), start, 0.0)).q_deg == pytest.approx((0.0,))
    half = policy.select_action(Observation((0.0,), start + 50_000_000, 0.05))
    assert half.q_deg[0] == pytest.approx(0.05)  # frames are 0, 0.1, 0.2
    end = policy.select_action(Observation((0.0,), start + 1_000_000_000, 1.0))
    assert end.q_deg[0] == pytest.approx(0.2)


def test_replay_policy_uses_raw_actions_on_request(tmp_path: Path) -> None:
    path = make_episode(tmp_path, episode_id="raw", frames=2, joints=1)
    guarded = ReplayPolicy(path)
    raw = ReplayPolicy(path, use_raw_actions=True)
    guarded.reset()
    raw.reset()
    assert (
        raw.select_action(Observation((0.0,), 1_000_000_000, 0.0)).q_deg
        != guarded.select_action(Observation((0.0,), 1_000_000_000, 0.0)).q_deg
    )


# --------------------------------------------------------------------------- #
# Action chunks, policy specs and remote inference
# --------------------------------------------------------------------------- #
def test_normalize_policy_output_accepts_chunks_and_mappings() -> None:
    from soar.policy import normalize_policy_output
    from soar.types import Action, ActionChunk

    single = normalize_policy_output([0] * 6, joints=6)
    assert single is not None and single.horizon == 1

    chunk = normalize_policy_output([[0] * 6, [1] * 6], joints=6)
    assert chunk is not None and chunk.horizon == 2

    explicit = normalize_policy_output(ActionChunk(steps=((0.0,) * 6,)), joints=6)
    assert explicit is not None and explicit.horizon == 1

    mapped = normalize_policy_output(
        {"actions": [[0] * 6, [2] * 6], "meta": {"chunk": 7}}, joints=6
    )
    assert mapped is not None and mapped.horizon == 2
    assert mapped.meta["chunk"] == 7

    assert normalize_policy_output(Action((0.0,) * 6), joints=6).horizon  # type: ignore[union-attr]
    with pytest.raises(PolicyError):
        normalize_policy_output([[0] * 5], joints=6)
    with pytest.raises(PolicyError):
        normalize_policy_output([], joints=6)


def test_policy_spec_is_read_from_the_policy() -> None:
    from soar.policy import PolicySpec, get_policy_spec

    class WithSpec:
        name = "vla"
        spec = PolicySpec(name="vla", action_dim=6, action_horizon=16, language=True)

    spec = get_policy_spec(WithSpec(), action_dim=6)
    assert spec.action_horizon == 16 and spec.language
    fallback = get_policy_spec(ConstantPolicy((0.0,) * 6), action_dim=6)
    assert fallback.name == "constant" and fallback.action_dim == 6


class _FakeClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def infer(self, request):
        self.requests.append(dict(request))
        return {"actions": [[0.0] * 6, [1.0] * 6], "meta": {"server": "fake"}}


def test_remote_policy_encodes_observation_and_returns_a_chunk() -> None:
    import base64

    from soar.policy import RemotePolicy
    from soar.types import ImageFrame

    client = _FakeClient()
    policy = RemotePolicy(client, name="vla")
    observation = Observation(
        (0.0,) * 6,
        123,
        0.0,
        images={"wrist": ImageFrame("wrist", b"jpeg", ".jpg", 123)},
        instruction="pick the cube",
    )
    chunk = policy.select_action(observation)
    assert chunk is not None and chunk.horizon == 2
    assert policy.last_meta["server"] == "fake"
    request = client.requests[0]
    assert request["instruction"] == "pick the cube"
    assert request["images"]["wrist"]["data_b64"] == base64.b64encode(b"jpeg").decode()
