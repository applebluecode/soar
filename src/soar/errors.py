"""Framework-wide exception hierarchy.

Device errors keep their :mod:`pysoarm` classes; this hierarchy only marks errors raised by
the research layer (configuration, data, policies, recorders). That lets callers write
``except pysoarm.PysoarmError`` for hardware and ``except SoarError`` for everything above
it without collapsing the two domains.
"""

from __future__ import annotations

__all__ = ["ConfigError", "DataError", "PolicyError", "RecorderError", "SoarError"]


class SoarError(Exception):
    """Base class for every error raised by the research framework."""


class ConfigError(SoarError):
    """Invalid configuration or a missing required argument."""


class DataError(SoarError):
    """An episode is missing, corrupt or fails validation."""


class PolicyError(SoarError):
    """A policy returned something unusable or failed during a rollout."""


class RecorderError(SoarError):
    """The recorder could not write or lost its background writer."""
