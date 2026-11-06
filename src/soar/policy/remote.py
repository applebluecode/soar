"""Remote inference adapter for VLA/WAM servers.

Training and serving usually live in another process/GPU host. The on-robot policy only
needs a synchronous ``client.infer(request) -> response`` call; this module defines that
boundary, encodes observations as JSON (images base64), and normalises the returned action
chunk. No model code lives here, and no network dependency is required by the core.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from soar.errors import PolicyError
from soar.policy.base import PolicySpec, normalize_policy_output
from soar.types import ActionChunk, Observation

__all__ = ["JsonHttpPolicyClient", "PolicyClient", "RemotePolicy", "encode_observation"]


class PolicyClient(Protocol):
    """Transport used by :class:`RemotePolicy` (HTTP, gRPC, local IPC, ...)."""

    def infer(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


def encode_observation(observation: Observation) -> dict[str, Any]:
    """JSON-serialisable observation for a remote model server.

    Images are base64-encoded with their timestamps; the server is responsible for
    resizing/normalisation. Keeping the wire format explicit makes it reproducible.
    """
    return {
        "q_deg": list(observation.q_deg),
        "t_host_ns": observation.t_host_ns,
        "t_mono_s": observation.t_mono_s,
        "instruction": observation.instruction,
        "meta": dict(observation.meta),
        "images": {
            name: {
                "data_b64": base64.b64encode(frame.data).decode("ascii"),
                "ext": frame.ext,
                "t_host_ns": frame.t_host_ns,
            }
            for name, frame in observation.images.items()
        },
    }


@dataclass(frozen=True, slots=True)
class JsonHttpPolicyClient:
    """Minimal JSON-over-HTTP client (stdlib only).

    The server receives ``POST <url>`` with an observation JSON and must answer with
    ``{"actions": [[...], ...], "meta": {...}}`` or ``{"action": [...]}``.

    Args:
        url: Inference endpoint.
        timeout_s: Per-request timeout.
        headers: Extra HTTP headers (auth tokens, content type overrides).
    """

    url: str
    timeout_s: float = 10.0
    headers: Mapping[str, str] | None = None

    def infer(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(request).encode("utf-8")
        headers = {"Content-Type": "application/json", **(self.headers or {})}
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise PolicyError("remote policy response must be a JSON object")
        return payload


class RemotePolicy:
    """Policy backed by a remote VLA/WAM inference client.

    Args:
        client: Transport implementing :class:`PolicyClient`.
        name: Display name (used in reports/recordings).
        spec: Optional capability descriptor; inferred fields are filled when omitted.
        max_horizon: Maximum accepted action-chunk length.

    The policy is stateless on the robot side: episode state is owned by the server (or
    sent in ``Observation.meta``), which keeps the adapter usable for stateful WAMs.
    """

    def __init__(
        self,
        client: PolicyClient,
        *,
        name: str = "remote",
        spec: PolicySpec | None = None,
        max_horizon: int = 64,
    ) -> None:
        self._client = client
        self.name = name
        self.spec = spec or PolicySpec(
            name=name, action_horizon=max_horizon, language=True, device="remote"
        )
        self._max_horizon = max_horizon
        self._last_meta: Mapping[str, Any] = {}

    @property
    def last_meta(self) -> Mapping[str, Any]:
        """Metadata returned by the server for the most recent inference."""
        return self._last_meta

    def reset(self) -> None:
        self._last_meta = {}

    def select_action(self, observation: Observation) -> ActionChunk | None:
        response = self._client.infer(encode_observation(observation))
        chunk = normalize_policy_output(
            response, joints=len(observation.q_deg), max_horizon=self._max_horizon
        )
        if chunk is not None:
            self._last_meta = chunk.meta
        return chunk

    def close(self) -> None:
        return None
