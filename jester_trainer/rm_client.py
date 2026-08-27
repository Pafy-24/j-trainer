from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .errors import TrainerError


class RMScopedClient:
    def __init__(self, topology: dict[str, Any]):
        controller = topology.get("rmController", {})
        grant = topology.get("serviceGrant") or {}
        self.base_url = str(controller.get("url", "")).rstrip("/")
        self.credential = str(grant.get("credential", ""))
        self.topology_id = str(topology.get("topologyId", ""))

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.credential and self.topology_id)

    def get(self, path: str) -> dict[str, Any]:
        if not self.available:
            raise TrainerError(
                "RM service grant is unavailable in the topology context",
                code="rm_service_grant_unavailable",
                status=409,
            )
        request = urllib.request.Request(
            self.base_url + path,
            method="GET",
            headers={
                "Authorization": f"Bearer {self.credential}",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                problem = json.loads(raw)
            except json.JSONDecodeError:
                problem = {"code": "rm_http_error", "detail": raw}
            raise TrainerError(
                str(problem.get("detail", exc.reason)),
                code=str(problem.get("code", "rm_http_error")),
                status=exc.code,
            ) from exc
        except (OSError, urllib.error.URLError) as exc:
            raise TrainerError(
                f"RM topology channel is unavailable: {exc}",
                code="rm_unavailable",
                status=503,
            ) from exc
        if not isinstance(value, dict):
            raise TrainerError(
                "RM returned a non-object response",
                code="invalid_rm_response",
                status=502,
            )
        return value

    def metrics(self) -> dict[str, Any]:
        return self.get(f"/v1/topologies/{self.topology_id}/metrics")

    def events(self) -> dict[str, Any]:
        return self.get(f"/v1/topologies/{self.topology_id}/events")

    def effective_config(self) -> dict[str, Any]:
        return self.get(
            f"/v1/topologies/{self.topology_id}/effective-config"
        )
