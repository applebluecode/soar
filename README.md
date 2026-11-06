# SOAR — policy-agnostic research framework for SO-ARM

**SOAR** (SO-ARM Robot) is a small, layered framework for doing embodied-AI research on the
**SO-100 / SO-101** arm.
The hardware boundary is [`pysoarm`](https://pypi.org/project/pysoarm/) (serial bus, safety,
calibration); everything above it is ours and **does not depend on LeRobot at runtime**.

The framework answers four questions with one architecture:

1. **Calibrate** the arm (motor-side, LeRobot-compatible JSON).
2. **Teleoperate** with a leader arm (200 Hz, safety limits, optional recording).
3. **Record / replay** episodes (async writer, timestamps, validation, dataset windows).
4. **Evaluate policies** (teleop, replay, scripted baselines, ACT/VLA/RL checkpoints) with
   the same runner and the same dataset.

Policy training itself is deliberately *not* in the core: the framework defines the data
contract and the `Policy` protocol, so models stay replaceable. The same contract already
carries what VLA/WAM models need: language instructions, multi-camera observations, action
chunks, normalisation statistics, policy/dataset specs and a remote-inference adapter.

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│ runtime/        CLI + composition root (wires objects, no logic)  │
├──────────────────────────────────────────────────────────────────┤
│ evaluation/     rollouts, success predicate, metrics               │
│ train/          window extraction, TrainablePolicy protocol        │
├──────────────────────────────────────────────────────────────────┤
│ policy/         Policy protocol, registry, teleop/replay/baselines │
│ control/        PolicyRunner: read → policy → write → record       │
├──────────────────────────────────────────────────────────────────┤
│ data/           schema, async recorder, writers, reader, checks    │
│ hardware/       pysoarm: arms, calibration (only place importing)  │
├──────────────────────────────────────────────────────────────────┤
│ pysoarm         bus, safety, calibration primitives, diagnostics   │
└──────────────────────────────────────────────────────────────────┘
```

Dependency rule: **arrows point down only.** Policies see `Observation`/`Action`, never a
bus; the recorder never blocks the loop; the device layer never imports a policy.

Design references (shape, not code): LeRobot's robot/teleoperator/dataset/policy split,
robomimic's env/dataset/algo separation, Stable-Baselines3's policy protocol + registry,
Gymnasium's typed observation/action boundary, and libfranka's device-object ergonomics.

## What is implemented

| Area | Status |
|---|---|
| Calibration (`calibrate`, middle-pose convention, snapshot + rollback, LeRobot JSON) | ✅ |
| Teleoperation (`role="leader"`, 200 Hz, 720 °/s cap, transient-loss hold) | ✅ |
| Episode schema + async recorder + JSONL writer (+ optional Parquet) | ✅ |
| Reader, dataset discovery, validation, window extraction | ✅ |
| Replay (time-based, guarded or raw actions) | ✅ |
| Evaluation harness (N episodes, success predicate, metrics, optional recording) | ✅ |
| Policy protocol + registry + teleop/replay/scripted/sinusoid/chunked baselines | ✅ |
| VLA/WAM plumbing: language, multi-camera, `ActionChunk`, `PolicySpec`, `NormStats`, remote client | ✅ |
| Action-chunk execution (`replan_every`), chunk index recorded per frame | ✅ |
| Training pipeline protocol (`TrainablePolicy.fit/save`) | ✅ interface only |
| Cameras (fake + optional OpenCV latest-frame camera) | ✅ |
| Hardware-free CLI (`--fake`) and 57 offline tests | ✅ |

Not in this package: dataset video encoding, training loops/models, LeRobot Hub upload,
ROS 2 nodes, torque/impedance control. Those belong to `soar_policy`, `soar_ros`, or
LeRobot itself.

## Install

```bash
git clone git@github.com:applebluecode/soar.git
cd soar

# optional: isolated environment
python -m venv .venv && source .venv/bin/activate

pip install -e .

# optional extras
pip install -e ".[yaml,parquet,cameras,train,dev]"
```

Working from an existing checkout is the same, minus the clone:

```bash
cd soar
pip install -e ".[dev]"    # or: make install
```

Requires Python ≥ 3.10 and `pysoarm ≥ 0.1.2` (0.1.3 recommended for the middle-pose
calibration convention).

## Configs

Typed configs load from JSON/YAML and reject unknown keys:

```python
from soar.config import load_config
from soar.hardware.arm import make_arm

config = load_config("configs/hardware_teleop.json")
follower = make_arm(config.robot)
leader = make_arm(config.leader)
```

The CLI keeps explicit flags for experiments; `configs/` shows the same settings in a
serialisable form for programmatic runs and sweeps.

## Quickstart (no hardware)

```bash
# versions, registry, schema
soar info

# two FakeBus arms, record 1 second
soar teleop --fake --hz 200 --seconds 1 \
  --record data/demo --task fake_teleop --cameras wrist

# inspect + validate the episode
soar dataset info data/demo
soar dataset validate data/demo

# replay the episode on a fake arm
soar replay --episode data/demo/<episode_id> --fake --hz 200 --speed 1

# evaluate a baseline policy
soar eval --policy sinusoid --fake --episodes 3 --seconds 2
```

## Quickstart (hardware)

```bash
# 1. calibration (interactive: mid-travel, then full travel)
soar calibrate --port /dev/ttyACM0 --role follower --id hw_follower --yes

# 2. leader -> follower teleoperation, recording
soar teleop \
  --leader-port /dev/ttyACM1 --follower-port /dev/ttyACM0 \
  --hz 200 --seconds 20 --max-speed-deg-s 720 \
  --record data/pick --task pick_cube --cameras 0

# 3. read-only diagnosis
soar doctor --port /dev/ttyACM0
```

`--max-speed-deg-s` defaults to `720` for teleoperation. `--no-speed-cap` restores pysoarm's
0.1.2 default (no software ceiling) — use it knowingly.

## The policy contract

```python
from soar.policy import PolicySpec
from soar.types import ActionChunk, Observation

class MyPolicy:
    name = "my_policy"
    spec = PolicySpec(name=name, action_dim=6, action_horizon=16,
                      image_keys=("wrist",), language=True)

    def reset(self) -> None: ...

    def select_action(self, obs: Observation) -> ActionChunk:
        # obs.q_deg, obs.images, obs.instruction -> action chunk in degrees
        return ActionChunk(steps=..., meta={"tokens": ...})

    def close(self) -> None: ...
```

Run chunked policies with `--replan-every 8`. See
[`docs/policies.md`](docs/policies.md).

Register it so configs/CLI can find it:

```python
from soar.policy import register_policy

@register_policy("act")
def build_act(checkpoint: str, camera: str = "wrist") -> MyPolicy:
    return ACTPolicy(checkpoint, camera)
```

Then `soar eval --policy act --policy-kwargs '{"checkpoint": "runs/act"}' ...`.
Remote VLA/WAM servers plug in through `RemotePolicy(JsonHttpPolicyClient(url))`.

## Data format (schema 0.1)

```text
data/<task>/<episode_id>/
  meta.json          # schema, task, joints, robot, calibration hash, success, counters
  frames.jsonl       # or frames.parquet
  events.jsonl       # recording notes / errors / operator events
  images/<camera>/<frame_index>.jpg
```

Each frame stores:

- `t_host_ns` (wall clock, dataset time base) and `t_mono_s` (device clock);
- `q_deg` (measured), `q_raw_deg` (policy request), `q_cmd_deg` (what was sent);
- `clipped`, `latency_s`, health snapshot.

The **guarded** action is the default training target; raw actions are kept so clipping and
safety experiments remain reproducible.

## Repository layout

```
src/soar/
  runtime/      CLI, composition root
  hardware/     arm factory, calibration service
  control/      PolicyRunner
  data/         schema, recorder, writers, reader, validation, cameras
  policy/       protocol, registry, teleop/replay/baselines
  evaluation/   Evaluator, metrics
  train/        windows + TrainablePolicy protocol
tests/          57 offline tests (FakeBus, no hardware)
configs/        example JSON configs
docs/           architecture, data format, extension guide, ADRs
```

## Tests

```bash
python -m pytest -q          # 57 passed, 2 skipped (optional extras)
ruff check src tests
```

Hardware tests are marked and skipped by default. The CLI `--fake` path exercises teleop →
recording → validation → replay → evaluation end to end.

Measured on two real SO-101 arms (follower `/dev/ttyACM0`, leader `/dev/ttyACM1`):

```text
teleop  5 s @ 200 Hz : 1000 ticks, 0 overrun, latency 1.25 ms (p95 1.31), 0 dropped frames
replay  2x @ 200 Hz  :  540 ticks, 0 overrun, latency 0.23 ms
dataset validate     : 1 episode, 0 errors, 1000 frames
```

## VLA / WAM readiness

- `Observation.instruction` carries the task prompt; `--task` fills it for teleop/replay/eval.
- `Observation.images` is a named multi-camera map; `--cameras wrist front`.
- Policies may return an **`ActionChunk`** (ACT/Diffusion/π0/WAM style); the runner executes it
  at the control rate and re-plans every `run.replan_every` ticks.
- `PolicySpec` declares action dimension, horizon, cameras and language; the runner validates
  it before the first tick.
- `EpisodeWindowDataset` yields language + camera paths + state/action windows, with
  `DatasetSpec` and `NormStats` written next to checkpoints.
- `RemotePolicy` + `JsonHttpPolicyClient` provide the GPU/remote-inference seam; the core has
  no torch dependency.

Algorithm packages (`soar_policy_act`, `soar_policy_vla`, ...) implement `Policy` and
`TrainablePolicy`; nothing in the control loop changes.

## Documentation

- [`docs/policies.md`](docs/policies.md) — VLA/WAM contract, chunks, normalisation, remote serving.
- [`docs/architecture.md`](docs/architecture.md) — layers, dependency rule, data flow.
- [`docs/data_format.md`](docs/data_format.md) — episode schema and validation rules.
- [`docs/extending.md`](docs/extending.md) — add a policy, camera, writer or robot model.
- [`docs/adr/`](docs/adr/) — architecture decision records.
