# ADR 0003: Store both the requested and the applied action

Status: accepted (2026-11-05)

## Context

`pysoarm.write_goal` applies soft limits, speed/step caps, a deadband and health guards.
Training on the applied action is what the robot actually did; training on the requested
action is what the policy asked for. They differ exactly when safety matters, and that
difference is research data, not noise.

## Decision

Every frame stores:

- `q_deg` — measured state;
- `q_raw_deg` — policy request (may be null when holding);
- `q_cmd_deg` — action actually written;
- `clipped` — whether they differ;
- `t_host_ns` and `t_mono_s` — wall clock and device clock;
- health snapshot, latency, camera paths.

Episodes are written to a hidden temp directory and renamed on close. Metadata records schema
version, joint order, calibration hash and drop counters. Validation rejects non-monotonic
time, shape mismatches, missing images and high drop ratios.

## Consequences

- Behaviour-cloning can choose raw or guarded targets (`EpisodeWindowDataset(use_raw_actions=)`).
- Replay defaults to guarded actions (reproducible robot behaviour) with `--raw` for analysis.
- The dataset is slightly larger; accepted.
- Guard reason strings are reserved (`reasons`) because pysoarm 0.1.3 does not return
  `SafetyReport` from `write_goal` yet.

## References

`docs/data_format.md`, `soar/data/schema.py`, LeRobot Dataset v3 semantics (as a reference).
