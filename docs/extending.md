# Extending the framework

Everything is replaceable through a protocol + a factory. The examples below are the
intended extension points; none of them requires touching the control loop.

## 1. Add a policy

A policy is any object with `name`, `reset()`, `select_action(obs)` and `close()`.

```python
from soar.policy import register_policy
from soar.types import Action, Observation

class ACTPolicy:
    name = "act"

    def __init__(self, checkpoint: str, camera: str = "wrist") -> None:
        self.camera = camera
        self.model = load_act(checkpoint)  # your code

    def reset(self) -> None:
        self.model.reset_hidden_state()

    def select_action(self, obs: Observation) -> Action:
        image = obs.images[self.camera].data
        q = self.model.predict(image, obs.q_deg)
        return Action(q_deg=tuple(q), meta={"policy": self.name})

    def close(self) -> None:
        pass

@register_policy("act")
def build_act(checkpoint: str, camera: str = "wrist") -> ACTPolicy:
    return ACTPolicy(checkpoint, camera)
```

```bash
soar eval --policy act --policy-kwargs '{"checkpoint": "runs/act"}' \
  --port /dev/ttyACM0 --episodes 10 --seconds 60
```

For VLA/WAM models return an `ActionChunk` and declare capabilities:

```python
from soar.policy import PolicySpec
from soar.types import ActionChunk

class MyVLA:
    name = "vla"
    spec = PolicySpec(name=name, action_dim=6, action_horizon=16,
                      image_keys=("wrist", "front"), language=True)

    def select_action(self, obs):
        return ActionChunk(steps=tuple(self.model(obs)), meta={"tokens": ...})
```

Run with `--replan-every 8`; the runner executes the chunk and re-plans when needed.
Episode-local state (history, latent world state) belongs in `reset()`. For remote/GPU
inference implement `PolicyClient.infer` and wrap it in `RemotePolicy`; see
[`docs/policies.md`](policies.md).

## 2. Add a camera

Implement `read() -> ImageFrame | None` (non-blocking) and run your own capture thread.

```python
from soar.data.cameras import CameraRig
from soar.types import ImageFrame

class MyCamera:
    name = "side"

    def read(self) -> ImageFrame | None:
        return self._latest  # update in a background thread

    def close(self) -> None: ...
```

Then pass a `CameraRig([...])` to `PolicyRunner` / `Evaluator` (or extend
`runtime.factory.build_camera_rig`). Keep timestamps from `time.time_ns()` at capture time.

## 3. Add a storage backend

A writer needs `append(frame, images)`, `event(name, **fields)`, `finalize(meta)` and
`abort()`. Write to a temp directory and publish atomically like `JsonlEpisodeWriter`.

```python
from soar.data.writers import make_writer  # extend the factory branch
```

For large datasets, Parquet is already available; for video, store frames as JPEG and let a
training loader build clips (or add an mp4 writer outside the core).

## 4. Add a policy dataset/training loop

`soar.train` deliberately stops at the data contract:

```python
from soar.train import TrainConfig, TrainablePolicy, TrainingPipeline

class MyACTPolicy:                     # implements Policy and TrainablePolicy
    name = "act"
    def fit(self, dataset, config: TrainConfig): ...  # returns TrainStats
    def save(self, path): ...                          # writes checkpoint
    def load(self, path): ...
    def select_action(self, obs): ...

pipeline = TrainingPipeline(policy, EpisodeWindowDataset("data/pick"), config)
report = pipeline.run()
```

The core does not import torch. Put your model, loss and optimizer in a separate package
(`soar_policy_act`, `soar_policy_vla`, ...) and register the policy factory.

## 5. Add a robot model / bus

The device layer is `pysoarm`; new models or transports belong there (or in a pysoarm
plugin), not in this package. This framework only needs `Arm`:

```python
class Arm(Protocol):
    role: str
    def read_once(self, health: bool | None = False) -> ArmState: ...
    def write_goal(self, q_deg: Sequence[float]) -> tuple[float, ...]: ...
    def hold(self) -> ArmState: ...
    def stop(self) -> None: ...
    def set_speed_scale(self, scale: float) -> float: ...
```

If the interface is satisfied, `PolicyRunner`, `Evaluator` and the CLI work unchanged.

## 6. Add a config section

Config dataclasses live in `soar/config.py`; unknown keys are rejected. Add a field, a
dataclass, and (optionally) a CLI flag. Keep configs serialisable: no callables, use
`"module:function"` strings for hooks (see `eval.success_fn`).

## 7. Add an evaluator

The evaluator is a thin loop over `PolicyRunner`. For task-specific success, prefer a
`success_fn`:

```python
# my_tasks.py
def cube_in_bowl(result) -> bool:
    return read_final_state(result)["in_bowl"]
```

```bash
soar eval --episode data/pick/ep-1 --success my_tasks:cube_in_bowl
```

If you need a different observation (force sensors, multiple arms), implement a custom
runner rather than growing `PolicyRunner` with flags.
