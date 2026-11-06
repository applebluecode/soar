# Changelog

All notable changes to **SOAR** (SO-ARM Robot).

## [0.2.0] - 2026-11-05

VLA/WAM readiness: the policy contract now carries language, multi-camera observations and
action chunks; training and inference metadata are first-class.

### Added

- `Observation.instruction` (task prompt) and `ActionChunk` (multi-step policy output).
- `PolicySpec` + `get_policy_spec`: action dimension, horizon, required cameras, language.
- Chunk execution in `PolicyRunner` via `RunConfig.replan_every`; chunk index/meta recorded
  per frame; `policy_spec` in `RunResult`.
- `normalize_policy_output` accepting `Action`, flat/nested sequences, `ActionChunk` and
  JSON mappings; `normalize_action` kept for compatibility.
- Remote inference: `PolicyClient`, `RemotePolicy`, `JsonHttpPolicyClient`, base64 image
  encoding; `remote` policy factory.
- Language + camera paths in `Window` (`instruction`, `images`, `as_dict`), `DatasetSpec`.
- `NormStats` / `JointStats` (state/action mean/std, save/load, normalise/denormalise).
- `TrainingPipeline` writes `policy_spec.json`, `dataset_spec.json`, `norm_stats.json`;
  `TrainConfig` gains observation/action horizons, device and dtype hints.
- `chunked` baseline policy; CLI `--replan-every`; `train` prints the dataset spec.
- Tests for chunks, replanning, instructions, remote policy, specs, normalisation, metadata.

### Changed

- Test count 48 → 57 (59 collected including optional-dependency skips).


## [0.1.0] - 2026-11-05

First framework release, extracted from the 0.1.x pysoarm validation workspace.

### Added

- Layered, policy-agnostic architecture (`hardware/`, `control/`, `data/`, `policy/`,
  `evaluation/`, `train/`, `runtime/`) with a one-way dependency rule and ADRs.
- Calibration service: middle-pose + half-turn convention, snapshot/rollback, LeRobot JSON.
- `PolicyRunner`: one loop for teleop, replay, baselines and learned policies; async
  recording with non-blocking drop accounting.
- Episode schema 0.1 with both guarded and raw actions, host/monotonic timestamps,  async
  JSONL writer (optional Parquet), reader, dataset discovery and validation.
- `TeleopPolicy`, `ReplayPolicy` (time-based interpolation), scripted/constant/random
  baselines and a policy registry.
- Evaluation harness with success predicates, per-episode failure isolation and JSON reports.
- Policy-agnostic training contract (`TrainablePolicy`, `TrainingPipeline`) and windowed
  dataset extraction.
- Cameras: fake camera for tests, latest-frame OpenCV camera for real use.
- CLI: `info`, `scan`, `doctor`, `calibrate`, `teleop`, `replay`, `eval`, `dataset`, `train`,
  with `--fake` end-to-end runs.
- 48 offline tests, ruff formatting/linting, strict typing config.

### Changed

- Project restructured from ad-hoc scripts into `src/soar`; legacy scripts and 0.1.x
  validation reports are not part of the published tree and are no longer maintained.
