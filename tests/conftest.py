"""Shared fixtures: fake arms and synthetic episodes (no hardware, no LeRobot)."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import make_episode
from pysoarm import FakeBus, SOArm


@pytest.fixture
def fake_arm():
    created: list[SOArm] = []

    def factory(*, role: str = "follower", backend: str = "fake", **kwargs):
        bus = FakeBus()
        arm = SOArm(bus=bus, role=role, **kwargs)  # type: ignore[arg-type]
        arm.connect()
        created.append(arm)
        return arm, bus

    yield factory
    for arm in created:
        arm.stop()


@pytest.fixture
def episode_dir(tmp_path: Path) -> Path:
    return make_episode(tmp_path / "data")


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    make_episode(root, episode_id="ep-0001", task="pick", success=True)
    make_episode(root, episode_id="ep-0002", task="pick", success=False)
    make_episode(root, episode_id="ep-0003", task="place", success=True)
    return root
