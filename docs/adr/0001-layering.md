# ADR 0001: Layered architecture with a one-way dependency rule

Status: accepted (2026-11-05)

## Context

The project started as a validation workspace for `pysoarm`: calibration scripts,
teleoperation experiments, hardware probes and reports. The next goal is a research
framework for SO-ARM where policies (teleop, replay, ACT, VLA, RL) are interchangeable and
the device layer stays small.

## Decision

Use six layers with a strict downward dependency rule:

```
runtime/ -> evaluation/ -> train/ -> policy/ -> control/ -> data/ -> hardware/ -> pysoarm
```

- `hardware/` is the only layer importing `pysoarm.SOArm`.
- `data/` has no knowledge of policies or hardware.
- `control/` composes hardware, policy and data.
- `runtime/` is the composition root (CLI); it is allowed to know everything.
- All layers may depend on `types.py`, `config.py`, `errors.py`.

Public extension points are protocols (`Policy`, `TrainablePolicy`, `Camera`,
`EpisodeWriter`, `Arm`), not base classes.

## Consequences

- The whole pipeline runs on `FakeBus`; hardware is only needed at the boundary.
- A policy cannot reach the bus, so safety guards cannot be bypassed accidentally.
- A new storage backend or camera does not touch the loop.
- Slight indirection: `runtime/factory.py` repeats some constructor wiring; accepted for a
  single, explicit composition root.

## References

LeRobot (robot/teleoperator/dataset/policy split), robomimic (env/dataset/algo), ADR 0002.
