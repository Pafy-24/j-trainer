"""Fail-closed validation for RM-approved controller egress sessions."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen


class ChannelSessionDenied(PermissionError):
    pass


@dataclass(frozen=True)
class ManagedChannelContext:
    """Trusted launch-time binding; never populated from request headers."""
    session_id: str
    project_id: str
    deployment_id: str
    deployment_generation: int
    engine_instance_id: str
    engine_instance_generation: int
    engine_lease_id: str
    channel_id: str
    destination_member_id: str
    destination_instance_id: str
    destination_instance_generation: int
    destination_lease_id: str
    allowed_campaign_ids: frozenset[str] = frozenset()

    def expectation(self, operation: str, method: str) -> dict[str, Any]:
        return {
            "sessionId": self.session_id, "projectId": self.project_id,
            "deploymentId": self.deployment_id, "deploymentGeneration": self.deployment_generation,
            "engineInstanceId": self.engine_instance_id, "engineInstanceGeneration": self.engine_instance_generation,
            "engineLeaseId": self.engine_lease_id, "channelId": self.channel_id,
            "destinationMemberId": self.destination_member_id, "destinationInstanceId": self.destination_instance_id,
            "destinationInstanceGeneration": self.destination_instance_generation,
            "destinationLeaseId": self.destination_lease_id, "operation": operation, "method": method,
        }


class ManagedChannelGuard:
    def __init__(self, context: ManagedChannelContext, *, introspect: Callable[[str, dict[str, Any]], Mapping[str, Any]]):
        self._context = context
        self.allowed_campaign_ids = context.allowed_campaign_ids
        self._introspect = introspect

    def authorize(self, headers: Mapping[str, str], *, operation: str, method: str) -> None:
        credential = headers.get("X-Jester-Component-Session")
        if not credential:
            raise ChannelSessionDenied("managed channel session is required")
        expectation_data = self._context.expectation(operation, method)
        try:
            from jester_core.channel_sessions import (
                ChannelSessionExpectation, ChannelSessionValidationError, validate_channel_session_claims,
            )
            claims = self._introspect(credential, expectation_data)
            validate_channel_session_claims(claims, ChannelSessionExpectation.model_validate(expectation_data))
        except (ChannelSessionValidationError, ValueError, OSError, KeyError, TypeError) as exc:
            raise ChannelSessionDenied("managed channel session is not authorized") from exc


def rm_introspector(rm_base_url: str) -> Callable[[str, dict[str, Any]], Mapping[str, Any]]:
    endpoint = rm_base_url.rstrip("/") + "/v1/component-sessions/introspect"
    def call(credential: str, expectation: dict[str, Any]) -> Mapping[str, Any]:
        request = Request(endpoint, data=json.dumps(expectation).encode("utf-8"), method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("X-Jester-Component-Session", credential)
        with urlopen(request, timeout=5) as response:  # no credential is logged
            return json.loads(response.read().decode("utf-8"))
    return call
