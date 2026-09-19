from __future__ import annotations

import pytest

from jester_trainer.channel_sessions import ChannelSessionDenied, ManagedChannelContext, ManagedChannelGuard


def _context() -> ManagedChannelContext:
    return ManagedChannelContext(
        session_id="session-one", project_id="project-one", deployment_id="deployment-one", deployment_generation=2,
        engine_instance_id="engine-one", engine_instance_generation=3, engine_lease_id="engine-lease",
        channel_id="channel-one", destination_member_id="destination", destination_instance_id="destination-one",
        destination_instance_generation=4, destination_lease_id="destination-lease",
    )


def _claims(expectation: dict[str, object]) -> dict[str, object]:
    return {
        "schemaVersion": "jester-channel-session-claims-v1", "active": True,
        "sessionId": expectation["sessionId"], "projectId": expectation["projectId"],
        "deploymentId": expectation["deploymentId"], "deploymentGeneration": expectation["deploymentGeneration"],
        "engineInstanceId": expectation["engineInstanceId"], "engineInstanceGeneration": expectation["engineInstanceGeneration"],
        "engineLeaseId": expectation["engineLeaseId"], "channelId": expectation["channelId"],
        "destinationMemberId": expectation["destinationMemberId"], "destinationInstanceId": expectation["destinationInstanceId"],
        "destinationInstanceGeneration": expectation["destinationInstanceGeneration"], "destinationLeaseId": expectation["destinationLeaseId"],
        "operation": expectation["operation"], "method": expectation["method"],
        "expiresAt": "2030-01-01T00:00:00+00:00",
    }


def test_guard_validates_rm_claims_and_does_not_accept_missing_credential() -> None:
    guard = ManagedChannelGuard(_context(), introspect=lambda _credential, expected: _claims(expected))
    guard.authorize({"X-Jester-Component-Session": "opaque-session-token-at-least-24-chars"}, operation="trainer.campaign:create", method="POST")
    with pytest.raises(ChannelSessionDenied):
        guard.authorize({}, operation="trainer.campaign:create", method="POST")


def test_guard_rejects_mismatched_introspection_claims() -> None:
    def invalid(expected: dict[str, object]) -> dict[str, object]:
        claims = _claims(expected)
        claims["destinationLeaseId"] = "wrong-lease"
        return claims
    guard = ManagedChannelGuard(_context(), introspect=lambda _credential, expected: invalid(expected))
    with pytest.raises(ChannelSessionDenied):
        guard.authorize({"X-Jester-Component-Session": "opaque-session-token-at-least-24-chars"}, operation="trainer.campaign:create", method="POST")
