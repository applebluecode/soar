"""Test helpers (importable from test modules)."""

from __future__ import annotations

import json
from pathlib import Path

from soar.data.schema import EpisodeMeta, Frame, new_episode_meta
from soar.data.writers import JsonlEpisodeWriter

__all__ = ["dump", "make_episode"]


def make_episode(
    root: Path,
    *,
    episode_id: str = "ep-0001",
    task: str = "pick",
    frames: int = 5,
    joints: int = 6,
    success: bool = True,
    step_ns: int = 10_000_000,
    camera: str | None = None,
) -> Path:
    """Write a tiny valid episode and return its directory."""
    meta: EpisodeMeta = new_episode_meta(
        episode_id=episode_id,
        task=task,
        joints=[f"j{i}" for i in range(joints)],
        robot={"model": "fake", "joints": joints},
        fps=30.0,
        camera_names=(camera,) if camera else (),
    )
    writer = JsonlEpisodeWriter(root, episode_id)
    for index in range(frames):
        state = tuple(0.1 * index + j for j in range(joints))
        frame = Frame(
            frame_index=index,
            t_host_ns=1_000_000_000 + index * step_ns,
            t_mono_s=index * step_ns / 1e9,
            q_deg=state,
            q_raw_deg=tuple(v + 0.5 for v in state),
            q_cmd_deg=state,
            latency_s=0.001,
            temperature_c=(35,) * joints,
            voltage_v=(11.9,) * joints,
        )
        images = {}
        if camera:
            from soar.types import ImageFrame

            images = {camera: ImageFrame(camera, b"jpeg", ".jpg", frame.t_host_ns)}
        writer.append(frame, images)
    writer.event("episode_start", task=task)
    final = EpisodeMeta(
        episode_id=meta.episode_id,
        schema_version=meta.schema_version,
        created_utc=meta.created_utc,
        task=meta.task,
        joints=tuple(meta.joints),
        robot=dict(meta.robot),
        fps=meta.fps,
        camera_names=tuple(meta.camera_names),
        success=success,
        calibration_sha256=None,
        duration_s=frames * step_ns / 1e9,
        frame_count=frames,
        dropped_frames=0,
    )
    writer.finalize(final)
    return Path(root) / episode_id


def dump(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
