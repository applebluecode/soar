# ADR 0004: Action chunks and a capability descriptor for VLA/WAM policies

Status: accepted (2026-11-05)

## Context

The first policy contract returned one joint vector per tick. Mainstream imitation/VLA/WAM
models (ACT, Diffusion Policy, π0, OpenVLA, world-action models) predict a short action
horizon, consume language and multiple cameras, and are often served remotely. Encoding those
assumptions in every experiment would leak model details into the control loop; ignoring them
would make the framework unusable for the target workloads.

## Decision

1. **`ActionChunk`** is a first-class policy output (`steps`, `meta`). `normalize_policy_output`
   accepts `Action`, flat/nested sequences, `ActionChunk` and JSON mappings.
2. The **runner**, not the policy, executes chunks: `run.replan_every` caps how many steps are
   executed before the next inference; an exhausted chunk re-plans early. The policy owns
   temporal ensembling or other chunk-assembly algorithms.
3. **`Observation.instruction`** carries the language prompt; cameras remain a named map.
4. **`PolicySpec`** is optional metadata (action dim/horizon, image keys, language, device)
   used to fail fast before the first tick and to document checkpoints.
5. Inference transports are a protocol: `PolicyClient.infer(request)`; the reference adapter
   is JSON-over-HTTP with base64 images. Model state stays inside the policy/server and is
   cleared by `reset()`.
6. Training metadata is written with checkpoints: `dataset_spec.json`, `norm_stats.json`,
   `policy_spec.json`.

## Consequences

- No model code enters the core; ACT/VLA/WAM packages implement the same protocol.
- Chunked execution is testable offline and recorded with the exact executed action.
- Re-planning latency is visible per tick (ticks that call the policy are more expensive).
- A policy that needs history must buffer it and reset per episode; the framework does not
  impose a window shape (the dataset helper provides one for training).
- Remote inference is synchronous in the runner; asynchronous prefetch is a policy-level
  concern (or a future `AsyncPolicyRunner`), keeping timing/recording deterministic.

## References

`docs/policies.md`, ADR 0002, ACT/π0/OpenVLA action-chunk interfaces.
