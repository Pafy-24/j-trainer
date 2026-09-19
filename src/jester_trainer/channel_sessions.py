"""Fail-closed validation for RM-approved controller egress sessions."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen


class ChannelSessionDenied(PermissionError):
    pass


@dataclass(frozen=True)
class ManagedChannelContext:
    """Trusted launch configuration plus a current effective destination lease."""
    rm_url: str

    @staticmethod
    def from_environment() -> "ManagedChannelContext":
        rm_url = os.environ.get("JESTER_RM_URL", "").strip()
        if not rm_url:
            raise ChannelSessionDenied("managed channel context is incomplete")
        return ManagedChannelContext(rm_url=rm_url)

    def expectation(self, operation: str, method: str) -> dict[str, Any]:
        try:
            from jester_rm_client import require_lease
            lease = require_lease()
        except Exception as exc:
            raise ChannelSessionDenied("managed destination lease is unavailable") from exc
        instance_id = str(lease.instance.get("metadata", {}).get("id", ""))
        generation = int(lease.instance.get("generation", 0))
        if not instance_id or generation < 1 or not lease.lease_id:
            raise ChannelSessionDenied("managed destination lease is incomplete")
        return {
            "destinationInstanceId": instance_id, "destinationInstanceGeneration": generation,
            "destinationLeaseId": lease.lease_id, "operation": operation, "method": method,
        }


class ManagedChannelGuard:
    def __init__(self, context: ManagedChannelContext, *, introspect: Callable[[str, dict[str, Any]], Mapping[str, Any]] | None = None, expectation_provider: Callable[[str, str], dict[str, Any]] | None = None):
        self._context = context
        self._introspect = introspect or rm_introspector(context.rm_url)
        self._expectation_provider = expectation_provider or context.expectation

    def authorize(self, headers: Mapping[str, str], *, operation: str, method: str) -> None:
        credential = headers.get("X-Jester-Component-Session")
        if not credential:
            raise ChannelSessionDenied("managed channel session is required")
        local = self._expectation_provider(operation, method)
        try:
            from jester_core.channel_sessions import (
                DestinationChannelExpectation, ChannelSessionValidationError,
                validate_destination_channel_claims,
            )
        except ImportError as exc:
            raise ChannelSessionDenied("managed channel validator is unavailable") from exc
        try:
            claims = dict(self._introspect(credential, local))
            expected = DestinationChannelExpectation.model_validate(local)
            return validate_destination_channel_claims(claims, expected)
        except (ChannelSessionValidationError, ValueError, OSError, KeyError, TypeError) as exc:
            raise ChannelSessionDenied("managed channel session is not authorized") from exc


def rm_introspector(rm_base_url: str) -> Callable[[str, dict[str, Any]], Mapping[str, Any]]:
    endpoint = rm_base_url.rstrip("/") + "/v1/component-sessions/introspect"
    def call(credential: str, expectation: dict[str, Any]) -> Mapping[str, Any]:
        request = Request(endpoint, data=json.dumps(expectation).encode("utf-8"), method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("X-Jester-Component-Session", credential)
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    return call


def guard_from_managed_environment() -> ManagedChannelGuard:
    return ManagedChannelGuard(ManagedChannelContext.from_environment())
