"""Command line interface: one place where configs, hardware and policies are wired.

Commands are intentionally thin. Each one builds objects, calls a service and prints a
report; no learning logic and no register-level code lives here. ``--fake`` runs the same
pipeline on ``pysoarm.FakeBus``, which is also how the CLI is smoke-tested.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pysoarm
from pysoarm import get_model, make_bus

from soar import __version__
from soar.config import EvalConfig, RecordConfig, RobotConfig, RunConfig
from soar.control.runner import PolicyRunner
from soar.data.cameras import CameraRig
from soar.data.reader import EpisodeDataset, EpisodeReader
from soar.data.schema import SCHEMA_VERSION
from soar.data.validation import validate_dataset
from soar.errors import ConfigError, SoarError
from soar.evaluation.evaluator import Evaluator
from soar.hardware.arm import make_arm
from soar.hardware.calibration import (
    calibrate_with_rollback,
    calibration_table,
)
from soar.policy import ReplayPolicy, TeleopPolicy, available_policies, build_policy
from soar.runtime.factory import build_camera_rig, build_recorder, calibration_hash
from soar.train.dataset import EpisodeWindowDataset

__all__ = ["build_parser", "main"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _print(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
    else:
        print(payload)


def _enter_listener(prompt: str) -> Any:
    """Non-blocking "Enter pressed" predicate (keeps a progress view alive)."""
    if not sys.stdin.isatty():
        raise ConfigError("calibration needs an interactive terminal")
    pressed = threading.Event()

    def listen() -> None:
        with contextlib.suppress(EOFError, KeyboardInterrupt):
            input(prompt)
        pressed.set()

    threading.Thread(target=listen, name="soar-enter", daemon=True).start()
    return pressed.is_set


def _run_config(args: argparse.Namespace, *, duration: float | None = None) -> RunConfig:
    return RunConfig(
        hz=args.hz,
        duration=args.seconds if duration is None else duration,
        on_error=args.on_error,
        speed_scale=args.speed_scale,
        instruction=args.task,
        replan_every=args.replan_every,
    )


def _record_config(args: argparse.Namespace) -> RecordConfig | None:
    if not args.record:
        return None
    return RecordConfig(
        root=args.record,
        task=args.task,
        cameras=tuple(args.cameras or ()),
        fps=args.fps,
        writer=args.writer,
    )


def _follower_config(
    args: argparse.Namespace, *, fake: bool, port: str | None = None
) -> RobotConfig:
    speed = None if args.no_speed_cap else args.max_speed_deg_s
    return RobotConfig(
        port=port if port is not None else args.follower_port,
        role="follower",
        backend=args.backend,
        calibration=args.follower_calibration,
        fake=fake,
        max_speed_deg_s=speed,
        max_step_deg=args.max_step_deg,
    )


def _leader_config(args: argparse.Namespace, *, fake: bool) -> RobotConfig:
    return RobotConfig(
        port=args.leader_port,
        role="leader",
        backend=args.backend,
        calibration=args.leader_calibration,
        fake=fake,
    )


def _camera_rig(config: RecordConfig | None, *, fake: bool) -> CameraRig:
    return build_camera_rig(config.cameras if config else (), fake=fake)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_info(args: argparse.Namespace) -> int:
    payload = {
        "soar": __version__,
        "pysoarm": pysoarm.__version__,
        "python": sys.version.split()[0],
        "dataset_schema": SCHEMA_VERSION,
        "policies": sorted(available_policies()),
        "models": ["so101", "so100"],
    }
    _print(payload, as_json=args.json)
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    bus = make_bus(args.port, model.motor_ids, backend=args.backend)
    bus.open()
    try:
        found = bus.scan(model.motor_ids)
    finally:
        bus.close()
    _print({"port": args.port, "motors": found}, as_json=args.json)
    return 0 if found else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    path = Path(args.port)
    model = get_model(args.model)
    report: dict[str, Any] = {
        "port": str(path),
        "exists": path.exists(),
        "readable": os.access(path, os.R_OK | os.W_OK),
        "backend": args.backend,
    }
    if path.exists() and report["readable"]:
        try:
            bus = make_bus(args.port, model.motor_ids, backend=args.backend)
            bus.open()
            try:
                report["motors"] = bus.scan(model.motor_ids)
                ids = model.motor_ids
                voltage = bus.sync_read(ids, "present_voltage")
                temperature = bus.sync_read(ids, "present_temperature")
                torque = bus.sync_read(ids, "torque_enable")
                report["voltage_v"] = {model.joint_of(i): v / 10 for i, v in voltage.items()}
                report["temperature_c"] = {model.joint_of(i): v for i, v in temperature.items()}
                report["torque_enable"] = {model.joint_of(i): bool(v) for i, v in torque.items()}
            finally:
                bus.close()
        except Exception as exc:  # report, don't crash: doctor is a diagnostic tool
            report["error"] = f"{type(exc).__name__}: {exc}"
    _print(report, as_json=args.json)
    return 0 if report.get("motors") else 1


def cmd_calibrate(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    bus = make_bus(args.port, model.motor_ids, backend=args.backend)
    bus.open()
    try:
        if not args.yes:
            print("This resets homing offsets and overwrites the motor-side calibration.")
            if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
                print("aborted")
                return 1
        wait_middle = None
        if args.middle:
            if not sys.stdin.isatty():
                raise ConfigError("calibration needs an interactive terminal")
            wait_middle = lambda: input(  # noqa: E731
                "Put every joint near mid-travel, then press Enter... "
            )
        should_stop = _enter_listener(
            "Move every joint through its full travel, then press Enter... "
        )
        last = [0.0]

        def progress(
            stage: str, positions: dict[int, int], mins: dict[str, int], maxs: dict[str, int]
        ) -> None:
            if stage != "record":
                return
            now = time.monotonic()
            if now - last[0] < 0.5:
                return
            last[0] = now
            cols = " ".join(
                f"{joint}:{positions[mid]}[{mins[joint]}..{maxs[joint]}]"
                for joint, mid in zip(model.joints, model.motor_ids, strict=True)
            )
            print(f"\r{cols}", end="", flush=True)

        result = calibrate_with_rollback(
            bus,
            model,
            role=args.role,
            middle=args.middle,
            wait_middle=wait_middle,
            should_stop=should_stop,
            full_turn=args.full_turn,
            poll_s=args.poll,
            progress=progress,
            output=args.output,
            cal_id=args.id,
            lerobot_layout=args.lerobot,
            backup=not args.no_backup,
        )
        print()
    finally:
        bus.close()
    rows = calibration_table(result.calibration, model)
    if args.json:
        _print(
            {
                "path": str(result.json_path),
                "sha256": result.sha256,
                "changed_joints": list(result.changed_joints),
                "rows": rows,
            },
            as_json=True,
        )
    else:
        print(f"{'joint':<15}{'id':>3}{'home':>7}{'min':>6}{'max':>6}{'span':>6}{'deg':>8}")
        for row in rows:
            print(
                f"{row['joint']:<15}{row['id']:>3}{row['homing_offset']:>7}"
                f"{row['range_min']:>6}{row['range_max']:>6}{row['span_ticks']:>6}"
                f"{row['span_deg']:>8}"
            )
        print(f"saved: {result.json_path}")
        print(f"changed joints: {list(result.changed_joints)}")
    return 0


def cmd_teleop(args: argparse.Namespace) -> int:
    fake = args.fake
    leader_cfg = _leader_config(args, fake=fake)
    follower_cfg = _follower_config(args, fake=fake)
    record_cfg = _record_config(args)
    cameras = _camera_rig(record_cfg, fake=fake)
    leader = make_arm(leader_cfg)
    follower = make_arm(follower_cfg)
    recorder = None
    try:
        if record_cfg is not None:
            recorder = build_recorder(
                record_cfg,
                follower,
                cameras=cameras,
                calibration_sha256=calibration_hash(follower_cfg),
            )
        runner = PolicyRunner(
            follower,
            TeleopPolicy(leader),
            _run_config(args),
            recorder=recorder,
            cameras=cameras,
        )
        result = runner.run()
        payload = result.as_dict()
        _print(payload, as_json=args.json)
        return 0 if result.completed else 1
    finally:
        cameras.close()
        follower.stop()
        leader.stop()


def cmd_replay(args: argparse.Namespace) -> int:
    policy = ReplayPolicy(args.episode, speed=args.speed, use_raw_actions=args.raw)
    follower_cfg = _follower_config(args, fake=args.fake)
    record_cfg = _record_config(args)
    cameras = _camera_rig(record_cfg, fake=args.fake)
    follower = make_arm(follower_cfg)
    recorder = None
    try:
        if record_cfg is not None:
            recorder = build_recorder(
                record_cfg,
                follower,
                cameras=cameras,
                calibration_sha256=calibration_hash(follower_cfg),
            )
        runner = PolicyRunner(
            follower, policy, _run_config(args), recorder=recorder, cameras=cameras
        )
        result = runner.run(duration=policy.duration_s + 0.2)
        _print(result.as_dict(), as_json=args.json)
        return 0 if result.completed else 1
    finally:
        cameras.close()
        follower.stop()


def cmd_eval(args: argparse.Namespace) -> int:
    if args.episode:
        policy = ReplayPolicy(args.episode, speed=args.speed, use_raw_actions=args.raw)
    else:
        kwargs = json.loads(args.policy_kwargs) if args.policy_kwargs else {}
        policy = build_policy(args.policy, **kwargs)
    follower_cfg = _follower_config(args, fake=args.fake)
    record_cfg = _record_config(args)
    config = EvalConfig(
        episodes=args.episodes,
        max_duration_s=args.seconds,
        policy=policy.name,
        success_fn=args.success,
        task=args.task or "eval",
        instruction=args.task,
    )
    camera_factory = (
        (lambda: _camera_rig(record_cfg, fake=args.fake))
        if args.fake or record_cfg is not None
        else None
    )

    def recorder_factory(arm: Any, cameras: CameraRig) -> Any:
        assert record_cfg is not None
        return build_recorder(
            record_cfg,
            arm,
            cameras=cameras,
            calibration_sha256=calibration_hash(follower_cfg),
        )

    evaluator = Evaluator(
        policy,
        lambda: make_arm(follower_cfg),
        config=config,
        run=_run_config(args),
        recorder_factory=recorder_factory if record_cfg is not None else None,
        camera_factory=camera_factory,
    )
    report = evaluator.run()
    if args.report:
        report.save(args.report)
    _print(report.as_dict(), as_json=True)
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    if args.action == "validate":
        reports = validate_dataset(args.root)
        bad = [report for report in reports if not report.ok]
        payload = {
            "root": args.root,
            "episodes": len(reports),
            "invalid": len(bad),
            "reports": [report.as_dict() for report in reports],
        }
        _print(payload, as_json=True)
        return 0 if not bad else 1
    if args.action == "info" and (Path(args.root) / "meta.json").is_file():
        _print(EpisodeReader(args.root).summary(), as_json=True)
        return 0
    dataset = EpisodeDataset(args.root, task=args.task)
    payload = dataset.summary()
    if args.action == "list":
        payload["episode_list"] = [EpisodeReader(path).summary() for path in dataset.paths]
    _print(payload, as_json=True)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Inspect the dataset windows a policy would train on (no model in core)."""
    dataset = EpisodeWindowDataset(
        args.root,
        task=args.task,
        success_only=args.success,
        observation_horizon=args.obs_horizon,
        action_horizon=args.act_horizon,
        stride=args.stride,
    )
    sample = next(iter(dataset), None)
    payload = {
        "root": args.root,
        "spec": dataset.spec().as_dict(),
        "episode_ids": list(dataset.episode_ids),
        "windows": len(dataset),
        "observations_per_window": args.obs_horizon,
        "actions_per_window": args.act_horizon,
        "first_window": None
        if sample is None
        else {
            "observations": sample.observations,
            "actions": sample.actions,
            "episode_id": sample.episode_id,
            "start_index": sample.start_index,
        },
    }
    _print(payload, as_json=True)
    return 0 if sample is not None else 1


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def _add_common(parser: argparse.ArgumentParser, *, model: bool = False) -> None:
    parser.add_argument("--backend", choices=("auto", "scs", "rustypot"), default="auto")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    if model:
        parser.add_argument("--model", default="so101")


def _add_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hz", type=float, default=200.0)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--on-error", choices=("abort", "hold", "skip"), default="hold")
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument(
        "--replan-every",
        type=int,
        default=1,
        help="execute N steps of an action chunk before re-planning",
    )
    parser.add_argument("--max-speed-deg-s", type=float, default=720.0)
    parser.add_argument("--max-step-deg", type=float, default=None)
    parser.add_argument(
        "--no-speed-cap", action="store_true", help="do not configure a software velocity ceiling"
    )


def _add_record(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--record", default=None, metavar="DIR", help="write an episode dataset under DIR"
    )
    parser.add_argument("--task", default="", help="task label stored in the episode")
    parser.add_argument(
        "--cameras", nargs="*", default=None, help="camera names (device index/path or fake names)"
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--writer", choices=("jsonl", "parquet"), default="jsonl")


def _add_robot(parser: argparse.ArgumentParser, *, leader: bool) -> None:
    if leader:
        parser.add_argument("--leader-port", default=None)
        parser.add_argument("--leader-calibration", default=None)
    else:
        parser.add_argument("--follower-port", "--port", dest="follower_port", default=None)
        parser.add_argument(
            "--follower-calibration", "--calibration", dest="follower_calibration", default=None
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soar",
        description="Policy-agnostic SO-ARM research framework (device layer: pysoarm).",
    )
    parser.add_argument("--version", action="version", version=f"soar {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="versions, registry and schema")
    info.add_argument("--json", action="store_true")
    info.set_defaults(handler=cmd_info)

    scan = sub.add_parser("scan", help="list servos on a bus")
    scan.add_argument("--port", required=True)
    _add_common(scan, model=True)
    scan.set_defaults(handler=cmd_scan)

    doctor = sub.add_parser("doctor", help="check a port and read voltage/temperature")
    doctor.add_argument("--port", required=True)
    _add_common(doctor, model=True)
    doctor.set_defaults(handler=cmd_doctor)

    cal = sub.add_parser("calibrate", help="run the motor-side calibration flow")
    cal.add_argument("--port", required=True)
    _add_common(cal, model=True)
    cal.add_argument("--role", choices=("follower", "leader"), default="follower")
    cal.add_argument("--id", default="default")
    cal.add_argument("--output", default=None)
    cal.add_argument("--lerobot", action="store_true", help="write to LeRobot's layout")
    cal.add_argument(
        "--middle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="wait for mid-travel before half-turn homing",
    )
    cal.add_argument("--full-turn", action="store_true")
    cal.add_argument("--poll", type=float, default=0.05)
    cal.add_argument("--no-backup", action="store_true")
    cal.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    cal.set_defaults(handler=cmd_calibrate)

    teleop = sub.add_parser("teleop", help="leader -> follower, optional recording")
    _add_robot(teleop, leader=True)
    _add_robot(teleop, leader=False)
    _add_common(teleop)
    _add_run(teleop)
    _add_record(teleop)
    teleop.add_argument("--fake", action="store_true", help="two FakeBus arms")
    teleop.set_defaults(handler=cmd_teleop)

    replay = sub.add_parser("replay", help="replay one episode")
    replay.add_argument("--episode", required=True)
    replay.add_argument("--speed", type=float, default=1.0)
    replay.add_argument("--raw", action="store_true", help="replay raw policy actions")
    _add_robot(replay, leader=False)
    _add_common(replay)
    _add_run(replay)
    _add_record(replay)
    replay.add_argument("--fake", action="store_true")
    replay.set_defaults(handler=cmd_replay)

    ev = sub.add_parser("eval", help="evaluate a policy or replay an episode")
    group = ev.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode", default=None, help="replay this episode as the policy")
    group.add_argument("--policy", default=None, help="registered policy name")
    ev.add_argument("--policy-kwargs", default=None, help="JSON kwargs for the policy")
    ev.add_argument("--episodes", type=int, default=5)
    ev.add_argument("--success", default=None, help="module:function(RunResult)->bool")
    ev.add_argument("--report", default=None, help="write eval.json here")
    ev.add_argument("--speed", type=float, default=1.0)
    ev.add_argument("--raw", action="store_true")
    _add_robot(ev, leader=False)
    _add_common(ev)
    _add_run(ev)
    _add_record(ev)
    ev.add_argument("--fake", action="store_true")
    ev.set_defaults(handler=cmd_eval)

    ds = sub.add_parser("dataset", help="inspect/validate a dataset")
    ds.add_argument("action", choices=("list", "validate", "info"))
    ds.add_argument("root")
    ds.add_argument("--task", default=None)
    ds.set_defaults(handler=cmd_dataset)

    tr = sub.add_parser("train", help="inspect training windows")
    tr.add_argument("root")
    tr.add_argument("--task", default=None)
    tr.add_argument("--success", action="store_true")
    tr.add_argument("--obs-horizon", type=int, default=2)
    tr.add_argument("--act-horizon", type=int, default=1)
    tr.add_argument("--stride", type=int, default=1)
    tr.add_argument("--json", action="store_true")
    tr.set_defaults(handler=cmd_train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (SoarError, pysoarm.PysoarmError, json.JSONDecodeError) as exc:
        print(f"soar: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
