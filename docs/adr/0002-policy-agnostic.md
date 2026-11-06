# ADR 0002: Policies are a protocol, not a class hierarchy

Status: accepted (2026-11-05)

## Context

Teleoperation, replay, scripted baselines and learned models all map an observation to joint
targets. A class hierarchy would force every model to inherit framework code and couple the
core to an ML stack.

## Decision

Define:

```python
class Policy(Protocol):
    name: str
    def reset(self) -> None: ...
    def select_action(self, observation: Observation) -> Action | Sequence[float] | None: ...
    def close(self) -> None: ...
```

- `None` means "hold the last goal".
- `Action` carries joint targets plus free-form metadata.
- Policies are built by factories registered under a name (`soar.policy.registry`).
- `PolicyRunner` is the only place that calls `select_action`.
- Training is a separate protocol (`TrainablePolicy.fit/save`), not part of `Policy`.

## Consequences

- The same evaluator runs teleop, replay, a constant baseline and an ACT checkpoint.
- The core has no torch/numpy dependency; models can use any framework.
- Replay is implemented as a policy, so the runner has one execution path.
- Policy authors must validate their own model-specific inputs; `normalize_action` only
  checks the joint contract.

## References

Stable-Baselines3 policy protocol, Gymnasium reset/step discipline, ADR 0001.
