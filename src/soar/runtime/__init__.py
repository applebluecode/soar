"""Composition root: factories and the command line interface."""

from soar.runtime.cli import build_parser, main
from soar.runtime.factory import (
    build_camera_rig,
    build_recorder,
    build_runner,
    build_teleop,
    calibration_hash,
)

__all__ = [
    "build_camera_rig",
    "build_parser",
    "build_recorder",
    "build_runner",
    "build_teleop",
    "calibration_hash",
    "main",
]
