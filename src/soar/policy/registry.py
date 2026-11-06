"""Policy registry: name -> factory, used by configs and the CLI.

This mirrors Stable-Baselines3's ``get_policy_from_string``/Gymnasium's registry idea without
the coupling: registration is a dictionary, factories are plain callables and third-party
packages can register themselves at import time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from soar.errors import PolicyError
from soar.policy.base import Policy

__all__ = ["available_policies", "build_policy", "register_policy"]

_REGISTRY: dict[str, Callable[..., Policy]] = {}


def register_policy(name: str) -> Callable[[Callable[..., Policy]], Callable[..., Policy]]:
    """Decorator: ``@register_policy("act")`` on a policy factory."""

    def decorator(factory: Callable[..., Policy]) -> Callable[..., Policy]:
        if name in _REGISTRY:
            raise PolicyError(f"policy {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return decorator


def build_policy(name: str, **kwargs: Any) -> Policy:
    """Instantiate a registered policy.

    Raises:
        PolicyError: Unknown name or factory failure.
    """
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise PolicyError(f"unknown policy {name!r}; available: {sorted(_REGISTRY)}") from exc
    try:
        return factory(**kwargs)
    except TypeError as exc:
        raise PolicyError(f"policy {name!r}: {exc}") from exc


def available_policies() -> dict[str, Callable[..., Policy]]:
    """Snapshot of the registry (name -> factory)."""
    return dict(_REGISTRY)
