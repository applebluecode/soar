# Architecture

## 1. Goal

**SOAR** (SO-ARM Robot) provides a **policy-agnostic** research stack for one class of
hardware (SO-100/SO-101):

- the device layer (bus, safety, calibration, diagnostics) is `pysoarm`;
- this package owns the *research workflow*: teleoperation, episode data, replay,
  evaluation, and the interfaces a policy must implement;
- LeRobot is a reference for workflow and data semantics, **not a runtime dependency**.

The target user is a lab that wants to run controlled experiments on a real arm without
inheriting a full robot-learning framework: collect episodes, train a policy elsewhere,
evaluate it here, compare runs with the same metrics.

## 2. Layers and the dependency rule

```text
runtime/      CLI + composition root        (depends on everything, used by nobody)
evaluation/   evaluator, metrics            (policy, control, data, config)
train/        windows + protocols           (data only)
policy/       Policy protocol, registry     (types + hardware protocol only)
control/      PolicyRunner                  (hardware, policy, data, config)
data/         schema, recorder, reader      (types only)
hardware/     pysoarm wrappers              (pysoarm only)
```

Arrows point **down**. Concretely:

- `data/` never imports `policy/` or `pysoarm`;
- `policy/` never imports `data/` (replay is the exception: it reads an episode, and that
  is explicit in its import);
- `hardware/arm.py` is the only module that imports `pysoarm.SOArm`;
- `runtime/` is the only module that constructs concrete objects from configs;
- tests import the same public surface as users.

This is what makes the full pipeline testable with `FakeBus` and swappable policies.

## 3. One loop for everything

Teleoperation, replay, baselines and learned policies are all `Policy` objects:

```text
pysoarm loop tick
  └─ read state (SOArm.read_once)
      └─ Observation(q_deg, t_host_ns, t_mono_s, images, instruction, meta)
          └─ policy.select_action(obs)  ->  Action | ActionChunk | None
              └─ chunk buffer: execute up to run.replan_every steps
                  └─ arm.write_goal(q)  (pysoarm guards: limits/watchdog/health)
                      └─ Frame(...) -> Recorder queue (non-blocking)
```

`PolicyRunner` owns this sequence. Timing, deadline misses and the error policy remain in
pysoarm's loop; the runner owns data capture and latency measurement.

Consequences:

- adding a policy does not touch the loop;
- adding a recorder does not touch policies;
- replay is just another policy (time-indexed actions);
- evaluation is the runner repeated with `reset()` and a success predicate.

## 4. Data flow

```text
Recording:
  control loop ──append(Frame, images)──► bounded queue ──► writer thread ──► episode dir
                 (never blocks)              (drops counted)     (atomic publish)

Training:
  episode dir ──► EpisodeReader ──► EpisodeWindowDataset ──► Window(obs, act, t)

Evaluation:
  Policy ──► PolicyRunner ──► RunResult ──► success_fn ──► EvalReport (JSON)
```

The wall-clock (`t_host_ns`) is the dataset time base; the monotonic clock (`t_mono_s`) is
for control timing. Cameras run their own thread and hand over the latest encoded frame, so
camera jitter does not become arm jitter.

## 5. Failure semantics

| Failure | Behaviour |
|---|---|
| one leader read fails | teleop policy returns `None` (hold), counted; session continues |
| safe number of consecutive bus failures | pysoarm watchdog disables torque and raises |
| policy returns wrong length/NaN | `PolicyError`, rollout aborts (recording marked failed) |
| writer queue full | frame dropped and counted; validation rejects a high drop ratio |
| calibration interrupted | snapshot written before touching motors; registers restored on failure |
| process killed mid-episode | temp directory `.tmp-*` is never published; cleanup is explicit |

## 6. Key decisions

1. **Typed values, not arrays.** `Observation`/`Action` are frozen dataclasses with tuples:
   serialisable, hashable, ML-framework independent. Convert to numpy/torch inside policies.
2. **Action chunks.** `ActionChunk` + `RunConfig.replan_every` give ACT/Diffusion/π0/WAM
   policies a multi-step interface while the runner stays algorithm-free.
3. **PolicySpec.** Optional capability metadata (action dim/horizon, cameras, language)
   fails fast when a checkpoint does not match the arm.
4. **Async recording.** A 200 Hz loop cannot wait for JPEG/Parquet writes. Bounded queue +
   drop counters makes the trade-off explicit.
5. **Both actions are recorded.** `q_raw_deg` (policy request) and `q_cmd_deg` (guarded
   command) make safety/clipping experiments reproducible.
6. **Wall clock + monotonic clock.** Dataset alignment needs `time_ns`; control needs
   `perf_counter`/`ArmState.t`.
7. **Registry, not inheritance.** Any object with `reset/select_action/close` is a policy;
   factories are registered by name for configs and the CLI.
8. **No training loop in core.** `train/` owns the data contract and `TrainablePolicy`;
   ACT/Diffusion/VLA/RL stay in user code or separate packages.

## 7. Relationship to LeRobot

| Concern | LeRobot | This package |
|---|---|---|
| Arm control | `SOFollower/SOLeader` | `hardware/` over `pysoarm` |
| Teleop | `lerobot-teleoperate` | `TeleopPolicy` + `PolicyRunner` |
| Recording | `lerobot-record` + Dataset v3 | `data.Recorder` + schema 0.1 |
| Replay | `lerobot-replay` | `ReplayPolicy` |
| Training | `lerobot-train` | `train/` protocol only |
| Eval | `lerobot-eval` | `evaluation.Evaluator` |
| Calibration | `lerobot-calibrate` | `hardware.calibration` (same JSON) |

Calibration JSON is intentionally LeRobot-compatible, so files can be exchanged without
importing LeRobot. Dataset export to LeRobot v3 is a **converter** (planned), not a runtime
dependency.

## 8. Testing strategy

- `FakeBus` + `FakeCamera` exercise teleop, recording, replay, evaluation and chunked
  execution end to end.
- The recorder is tested with a slow writer to prove non-blocking drops.
- Validation has negative tests (non-monotonic time, wrong joints, drop ratio, missing
  images).
- Calibration is tested for both homing conventions and for rollback after a failure.
- Hardware is opt-in (`hardware` marker); the CLI is smoke-tested with `--fake`.

## 9. References

- LeRobot: robot/teleoperator/dataset/policy layering, Dataset v3 semantics.
- robomimic: environment/dataset/algorithm separation.
- Stable-Baselines3: policy protocol and registry.
- Gymnasium: typed observation/action boundary, `reset/step` discipline.
- libfranka / pyAgxArm: device-object ergonomics, structured device errors.
