"""Episode writers, recorder, reader, validation and cameras."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from soar.data.cameras import CameraRig, FakeCamera
from soar.data.reader import EpisodeDataset, EpisodeReader
from soar.data.recorder import Recorder
from soar.data.schema import EpisodeMeta, Frame, new_episode_meta
from soar.data.validation import validate_dataset, validate_episode
from soar.data.writers import JsonlEpisodeWriter, ParquetEpisodeWriter, make_writer
from soar.errors import DataError
from soar.types import ImageFrame


def meta(episode_id: str = "ep", frames: int = 3) -> EpisodeMeta:
    return new_episode_meta(
        episode_id=episode_id,
        task="pick",
        joints=[f"j{i}" for i in range(6)],
        robot={"model": "fake"},
        fps=30.0,
    )


def frame(index: int, *, step_ns: int = 10_000_000) -> Frame:
    return Frame(
        frame_index=index,
        t_host_ns=1_000_000_000 + index * step_ns,
        t_mono_s=index * step_ns / 1e9,
        q_deg=tuple(float(index + j) for j in range(6)),
        q_raw_deg=tuple(float(index + j + 0.5) for j in range(6)),
        q_cmd_deg=tuple(float(index + j) for j in range(6)),
        latency_s=0.002,
        temperature_c=(35,) * 6,
        voltage_v=(11.9,) * 6,
    )


def test_jsonl_roundtrip_with_images_and_events(tmp_path: Path) -> None:
    writer = JsonlEpisodeWriter(tmp_path, "ep")
    writer.append(frame(0), {"wrist": ImageFrame("wrist", b"abc", ".jpg", 1)})
    writer.event("note", text="hello")
    path = writer.finalize(
        EpisodeMeta(
            **{
                **meta().as_dict(),
                "joints": tuple(meta().joints),
                "camera_names": ("wrist",),
                "frame_count": 1,
                "duration_s": 0.01,
            }
        )
    )
    reader = EpisodeReader(path)
    frames = list(reader)
    assert len(frames) == 1
    assert frames[0].images["wrist"].endswith(".jpg")
    assert (path / frames[0].images["wrist"]).read_bytes() == b"abc"
    assert (path / "events.jsonl").read_text().strip().endswith('"hello"}')
    assert reader.summary()["frames"] == 1


def test_writer_publishes_atomically_and_rejects_duplicates(tmp_path: Path) -> None:
    writer = JsonlEpisodeWriter(tmp_path, "ep")
    writer.append(frame(0), {})
    path = writer.finalize(meta("ep"))
    assert path.is_dir() and not list(tmp_path.glob(".ep.tmp-*"))
    with pytest.raises(DataError, match="already exists"):
        JsonlEpisodeWriter(tmp_path, "ep")


def test_recorder_thread_writes_and_finalizes(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path, meta("rec"), queue_size=128)
    recorder.start()
    for index in range(20):
        assert recorder.append(frame(index))
    path = recorder.close()
    reader = EpisodeReader(path)
    frames = list(reader)
    assert len(frames) == 20
    assert reader.meta.frame_count == 20
    assert recorder.dropped_frames == 0


def test_recorder_drops_instead_of_blocking(tmp_path: Path, monkeypatch) -> None:
    class SlowWriter(JsonlEpisodeWriter):
        def append(self, frame, images):
            time.sleep(0.02)
            super().append(frame, images)

    monkeypatch.setattr(
        "soar.data.recorder.make_writer",
        lambda kind, root, episode_id, **kw: SlowWriter(root, episode_id),
    )
    recorder = Recorder(tmp_path, meta("slow"), queue_size=1)
    recorder.start()
    for index in range(10):
        recorder.append(frame(index))
    recorder.close()
    assert recorder.dropped_frames > 0


def test_validation_reports_good_episode(episode_dir: Path) -> None:
    report = validate_episode(episode_dir)
    assert report.ok, report.errors
    assert report.stats["frames"] == 5
    assert report.stats["dropped_ratio"] == 0.0


def test_validation_catches_non_monotonic_and_wrong_joints(tmp_path: Path) -> None:
    writer = JsonlEpisodeWriter(tmp_path, "bad")
    writer.append(frame(0, step_ns=50_000_000), {})
    writer.append(frame(1, step_ns=0), {})  # same timestamp: non-monotonic
    path = writer.finalize(meta("bad"))
    report = validate_episode(path, expected_joints=[f"j{i}" for i in range(5)])
    assert not report.ok
    assert any("monotonic" in e for e in report.errors)
    assert any("joints" in e for e in report.errors)


def test_validation_flags_a_high_drop_ratio(tmp_path: Path) -> None:
    writer = JsonlEpisodeWriter(tmp_path, "drops")
    writer.append(frame(0), {})
    base = meta("drops")
    path = writer.finalize(
        EpisodeMeta(
            **{
                **base.as_dict(),
                "joints": tuple(base.joints),
                "frame_count": 1,
                "dropped_frames": 9,
            }
        )
    )
    report = validate_episode(path)
    assert not report.ok
    assert any("dropped" in e for e in report.errors)


def test_dataset_filtering_and_summary(dataset_root: Path) -> None:
    all_eps = EpisodeDataset(dataset_root)
    assert len(all_eps) == 3
    assert all_eps.tasks() == ["pick", "place"]
    pick_success = EpisodeDataset(dataset_root, task="pick", success=True)
    assert len(pick_success) == 1
    assert pick_success.summary()["success_rate"] == 1.0
    reports = validate_dataset(dataset_root)
    assert len(reports) == 3 and all(report.ok for report in reports)


def test_parquet_writer_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    writer = ParquetEpisodeWriter(tmp_path, "pq")
    writer.append(frame(0), {})
    path = writer.finalize(meta("pq"))
    assert (path / "frames.parquet").is_file()
    assert len(list(EpisodeReader(path))) == 1


def test_make_writer_unknown() -> None:
    with pytest.raises(DataError):
        make_writer("hdf5", "/tmp", "x")


def test_camera_rig_and_fake_camera() -> None:
    rig = CameraRig([FakeCamera("wrist"), FakeCamera("front")])
    frames = rig.read()
    assert set(frames) == {"wrist", "front"}
    rig.close()
