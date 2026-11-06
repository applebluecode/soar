# Policies: VLA / WAM readiness

The framework does not implement ACT, Diffusion Policy, π0, OpenVLA, RT-2 or a world-action
model. It provides the **contract and data plumbing** those models need, so an algorithm can
be dropped in as a policy without touching the control loop or the recorder.

## 1. What "support" means here

| Requirement (VLA/WAM) | Provided by |
|---|---|
| Single-step or action-chunk output | `Action`, `ActionChunk`, `normalize_policy_output` |
| Joint-space interface (deg) | `Observation.q_deg` / `ActionChunk.steps` |
| Language instruction | `Observation.instruction` (CLI `--task`) |
| Multi-camera observation | `Observation.images: name -> ImageFrame` |
| Proprioception | `Observation.q_deg`, `t_mono_s`, `health_age_s` |
| History / episode state | policy-owned buffers cleared by `reset()` |
| Normalisation statistics | `NormStats` (state/action mean/std), saved with checkpoints |
| Dataset/serving contract | `DatasetSpec`, `Window.as_dict()`, `PolicySpec` |
| Remote/GPU inference | `RemotePolicy` + `PolicyClient` (JSON-over-HTTP reference) |
| Action chunks executed at control rate | `PolicyRunner` + `RunConfig.replan_every` |
| Capability validation | `PolicySpec` (action dim, horizon, cameras, language) |
| Checkpoint metadata | `TrainingPipeline`: `policy_spec.json`, `dataset_spec.json`, `norm_stats.json` |

Not provided: model architectures, tokenizers, losses, distributed training, GPU serving,
video decoding. Those belong in a separate `soar_policy_*` package or an inference server.

## 2. The contract

```python
from soar.policy import PolicySpec, register_policy
from soar.types import ActionChunk, Observation

class MyVLA:
    name = "my_vla"
    spec = PolicySpec(
        name=name,
        action_dim=6,
        action_horizon=16,
        image_keys=("wrist", "front"),
        language=True,
        device="cuda",
    )

    def reset(self) -> None:
        self.history.clear()      # episode-local state

    def select_action(self, obs: Observation) -> ActionChunk:
        # obs.q_deg, obs.images["wrist"].data, obs.instruction
        steps = self.model.predict(obs)          # your code
        return ActionChunk(steps=tuple(steps), meta={"tokens": ...})

    def close(self) -> None: ...

@register_policy("my_vla")
def build(checkpoint: str, device: str = "cuda") -> MyVLA:
    return MyVLA(checkpoint, device)
```

`select_action` may also return:

- `Action(q_deg=...)` — one step;
- a flat sequence — one step;
- `{"actions": [[...], ...], "meta": {...}}` — JSON-friendly remote output;
- `None` — hold the last goal.

`normalize_policy_output` validates shape and finiteness; the runner additionally checks
`PolicySpec.action_dim` and required cameras before the first tick.

## 3. Action chunks

```python
runner = PolicyRunner(
    follower,
    policy,
    RunConfig(hz=200, replan_every=8),   # execute 8 steps, then call the policy again
    recorder=recorder,
)
```

- The policy is called when the chunk is empty or `replan_every` steps were executed.
- Each executed step is recorded (`q_raw_deg`, `q_cmd_deg`, chunk index in `frame.meta`).
- `replan_every=1` is the old per-tick behaviour; a chunked policy with a short horizon is
  re-planned when it runs out even if `replan_every` is larger.
- Latency is measured per tick; ticks that re-plan include model inference time.

For temporal ensembling / receding-horizon tricks, implement them **inside the policy**
(return the assembled chunk); the runner stays algorithm-free.

## 4. Language and cameras

```python
soar teleop --leader-port /dev/ttyACM1 --follower-port /dev/ttyACM0 \
  --record data/pick --task "pick the red cube and put it in the bowl" --cameras wrist front
```

- `--task` is stored in `meta.json` and passed to policies as `Observation.instruction`.
- `--cameras wrist front` builds a camera rig; fake names use `FakeCamera`, real names use
  OpenCV (a digit name is a device index).
- Replay uses the episode's task label as the instruction.
- Evaluation uses `--task` (or `eval.task` in config).

## 5. Training preparation

### Windows

```python
from soar.train import EpisodeWindowDataset, compute_norm_stats

dataset = EpisodeWindowDataset(
    "data/pick", observation_horizon=2, action_horizon=16,
    image_keys=("wrist", "front"), use_raw_actions=False,
)
window = next(iter(dataset))
window.observations      # ((q_t-1), (q_t))
window.actions           # 16 future guarded actions
window.instruction       # task text
window.images            # per-step camera paths
window.as_dict()         # LeRobot-like flat dict for a PyTorch Dataset
```

### Spec and normalisation

```python
spec = dataset.spec()          # joints, horizons, cameras, language, size
stats = compute_norm_stats("data/pick")
stats.save("runs/act/norm_stats.json")
```

`NormStats` keeps the **training-time** mean/std, so deployment normalises exactly like
training. Use `stats.state.normalize(...)` / `stats.action.denormalize(...)` in the policy.

### Training pipeline

```python
from soar.train import TrainConfig, TrainingPipeline

report = TrainingPipeline(
    policy, dataset,
    TrainConfig(epochs=100, action_horizon=16, device="cuda", dtype="bfloat16"),
    dataset_spec=dataset.spec(),
    norm_stats=stats,
).run()
```

The pipeline writes:

```text
runs/train/
  policy                 # policy.save(...)
  policy_spec.json
  dataset_spec.json
  norm_stats.json
  train.json
```

The model/loss/optimizer live in `policy.fit`; the core never imports torch.

## 6. Inference modes

### Local model

Register a factory and evaluate:

```bash
soar eval --policy act --policy-kwargs '{"checkpoint": "runs/act/policy"}' \
  --port /dev/ttyACM0 --episodes 10 --seconds 60 --task "pick the cube"
```

### Remote VLA/WAM server

```python
from soar.policy import JsonHttpPolicyClient, RemotePolicy

policy = RemotePolicy(JsonHttpPolicyClient("http://gpu-host:8000/infer"), name="vla")
```

The request is JSON with base64 images, proprio and the instruction; the response is
`{"actions": [[...], ...], "meta": {...}}`. `RemotePolicy` normalises the response but owns
no model state; a stateful WAM keeps state on the server keyed by `obs.meta` (e.g. an
episode id) or sends history in `meta`.

### Asynchronous inference

The runner is synchronous by design: one tick never waits longer than the model call. For
pipelined inference, implement a policy whose `select_action` returns the latest available
chunk and refreshes it in a background thread; `reset()` clears the thread state. This keeps
timing/recording in the runner and the concurrency inside the policy (or a separate
`soar_policy_server` package).

## 7. Model-family mapping

| Model family | How to plug in |
|---|---|
| ACT / Diffusion Policy | `ActionChunk` with horizon `k`; `replan_every <= k` |
| π0 / OpenVLA / RT-2 | `Observation.instruction` + multi-camera; token decoding inside policy |
| WAM (world-action model) | keep latent/world state in the policy or remote server; `reset()` per episode |
| BC-MLP / regression | single `Action` per tick |
| RL policies | same protocol; success/reward computed by `Evaluator`/`success_fn` |
| Teleoperation / replay | already implemented |

## 8. What to add next (algorithm side, not core)

1. A `soar_policy_act` package implementing `fit` and `select_action`.
2. A `soar_policy_vla` package with a tokenizer/processor and a `RemotePolicy` client.
3. A serving package: request validation against `PolicySpec` + `DatasetSpec`, batching,
   GPU memory management.
4. Dataset export/import to LeRobot v3 (optional, file-level).
