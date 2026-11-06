"""Policy-agnostic research framework for SO-ARM robots.

The package is layered so that research code never has to know about serial ports and the
device layer never has to know about policies:

    runtime/     composition root + CLI (wires everything, owns no logic)
    evaluation/  policy rollouts, success metrics
    train/       policy-agnostic dataset windows and training protocol
    policy/      Policy protocol + built-in teleop/replay/scripted policies
    control/     the single policy execution loop (read -> policy -> write -> record)
    data/        episode schema, async recorder, readers, validation
    hardware/    pysoarm wrappers: arms, calibration
    config.py    typed configuration objects
    types.py     Observation / Action / ImageFrame

The only hard dependency is :mod:`pysoarm`; cameras, Parquet, YAML and training have
optional extras. Everything above ``hardware/`` runs against fakes, so the full pipeline is
testable on a laptop.

Design references: LeRobot (robot/teleop/dataset/policy separation), robomimic
(env/dataset/algo split), Stable-Baselines3 (policy protocol + registry), Gymnasium (typed
observation/action boundary) and libfranka's device-object ergonomics.
"""

from soar.types import Action, ActionChunk, ImageFrame, Observation
from soar.version import __version__

__all__ = ["Action", "ActionChunk", "ImageFrame", "Observation", "__version__"]
