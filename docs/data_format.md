# Episode format (schema 0.1)

One episode is one directory. Everything is written under a hidden temp directory and
renamed on close, so a published episode is complete by construction.

```text
<root>/<episode_id>/
  meta.json
  frames.jsonl          # default
  frames.parquet        # if writer="parquet"
  events.jsonl
  images/<camera>/<frame_index>.jpg
```

## meta.json

| Field | Meaning |
|---|---|
| `episode_id` | Directory name; unique under the dataset root |
| `schema_version` | `"0.1"`; validation rejects other versions |
| `created_utc` | ISO-8601 UTC |
| `task` | Natural-language task label (VLA/WAM instruction) |
| `joints` | Joint order for every vector in the episode |
| `robot` | `{model, joints, motor_ids, role, backend, pysoarm_version, ...}` |
| `fps` | Recording hint (not an enforced rate) |
| `camera_names` | Cameras present under `images/` |
| `success` | `true`/`false`/`null` (null = not evaluated) |
| `tags` | Free-form labels |
| `calibration_sha256` | Hash of the calibration JSON used at record time |
| `duration_s`, `frame_count`, `dropped_frames` | Written at close |
| `extra` | Experiment-specific fields |

## Frame fields

| Field | Meaning |
|---|---|
| `frame_index` | 0..N-1, contiguous |
| `t_host_ns` | `time.time_ns()`; dataset time base |
| `t_mono_s` | `ArmState.t`; control timing |
| `q_deg` | Measured joint positions |
| `q_raw_deg` | Policy request (before guards); `null` when the tick held |
| `q_cmd_deg` | Command actually written (after guards); `null` when held |
| `latency_s` | read + policy + write, measured by the runner |
| `clipped` | `q_raw_deg != q_cmd_deg` |
| `reasons` | Reserved for guard reasons; currently empty (pysoarm API gap) |
| `temperature_c`, `voltage_v` | Health snapshot (cached, not a fresh read each tick) |
| `flags` | Per-joint alarm names, when available |
| `images` | Camera name -> relative path |
| `meta` | Policy name, instruction, chunk index/meta, health age, extras |

## Validation rules

`soar dataset validate <root>` fails an episode when:

- `schema_version` is not the current one;
- `joints` differs from the expected order;
- `t_host_ns` is not strictly increasing;
- vector lengths differ from `joints`, or values are not finite;
- `meta.frame_count` disagrees with the frames actually read;
- dropped-frame ratio exceeds 5 % (configurable);
- a referenced image file is missing;
- no frames at all.

Warnings (not failures): empty task label, missing calibration hash.

## Language, camera paths and training views

`EpisodeMeta.task` is the episode-level instruction. `Window` (training) attaches it to every
sample, plus per-observation camera paths:

```python
window.instruction              # "pick the red cube"
window.images                   # ({"wrist": "images/wrist/000000.jpg"}, ...)
window.as_dict()                # {"observation.state": ..., "action": ..., "task": ...,
                                #  "observation.images.wrist": [...]}
```

`DatasetSpec` describes joints/horizons/cameras/language for a dataset; `NormStats` stores
state/action mean/std next to a checkpoint.

## Time and replay

Replay is scheduled by `t_host_ns`, not by tick count:

```text
target = first_frame.t_host_ns + (now - start) * speed
action = linear_interpolate(frames, target)
```

This makes replay independent of the control frequency and lets `--speed` change duration
without changing the trajectory. Actions are guarded goals by default; `--raw` replays the
policy's own output (useful to study clipping).

## LeRobot interoperability

The calibration JSON is LeRobot-compatible. The episode schema is **not** LeRobot v3; a
converter (`to_lerobot_v3`) is a planned, file-level adapter so datasets can be shared with
that ecosystem without importing it at runtime. Mapping:

| this schema | LeRobot v3 |
|---|---|
| `q_deg` | `observation.state` |
| `q_cmd_deg` | `action` |
| `t_host_ns` | `timestamp` (converted to seconds) |
| `frame_index` | `frame_index` |
| episode dir | `episode_index` + chunk files |

## Parquet

`writer="parquet"` stores the same columns in `frames.parquet` (requires
`pip install -e ".[parquet]"`). Nested fields (`flags`, `images`, `meta`) are JSON strings in
that mode and decoded by the reader.
