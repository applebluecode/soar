"""Policy layer: protocol, registry and built-in teleop/replay/baseline policies."""

from soar.policy.base import (
    Policy,
    PolicyOutput,
    PolicySpec,
    get_policy_spec,
    normalize_action,
    normalize_policy_output,
)
from soar.policy.builtin import (
    CallablePolicy,
    ChunkedHoldPolicy,
    ConstantPolicy,
    HoldPolicy,
    RandomPolicy,
    ScriptedPolicy,
    sinusoid_policy,
)
from soar.policy.registry import available_policies, build_policy, register_policy
from soar.policy.remote import (
    JsonHttpPolicyClient,
    PolicyClient,
    RemotePolicy,
    encode_observation,
)
from soar.policy.replay import ReplayPolicy
from soar.policy.teleop import TeleopPolicy

# Built-in baselines are registered by name for configs/CLI. Trainable policies belong in
# user code and register themselves with the same decorator.
register_policy("constant")(ConstantPolicy)
register_policy("hold")(HoldPolicy)
register_policy("random")(RandomPolicy)
register_policy("scripted")(ScriptedPolicy)
register_policy("sinusoid")(sinusoid_policy)
register_policy("chunked")(ChunkedHoldPolicy)


@register_policy("remote")
def _build_remote(url: str, name: str = "remote") -> RemotePolicy:
    """Build a remote policy from an inference URL (JSON over HTTP)."""
    return RemotePolicy(JsonHttpPolicyClient(url), name=name)


__all__ = [
    "CallablePolicy",
    "ChunkedHoldPolicy",
    "ConstantPolicy",
    "HoldPolicy",
    "JsonHttpPolicyClient",
    "Policy",
    "PolicyClient",
    "PolicyOutput",
    "PolicySpec",
    "RandomPolicy",
    "RemotePolicy",
    "ReplayPolicy",
    "ScriptedPolicy",
    "TeleopPolicy",
    "available_policies",
    "build_policy",
    "encode_observation",
    "get_policy_spec",
    "normalize_action",
    "normalize_policy_output",
    "register_policy",
    "sinusoid_policy",
]
