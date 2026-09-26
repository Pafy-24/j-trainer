from __future__ import annotations

import copy
import json

import pytest
from fastapi.testclient import TestClient

from jester_trainer.api import create_app
from jester_trainer.channel_sessions import ManagedChannelContext, ManagedChannelGuard
from jester_trainer.errors import TrainerError

from .test_managed_execution_audit import OWNER, _seed_start, _service


READBACK_OPERATION = "trainer.campaign:publication-readback"


def _seed(service, *, campaign: str = "campaign-a", operation: str = "start-a") -> tuple[str, str]:
    attempt = f"{campaign}-attempt-a1"
    result = f"{attempt}-result"
    _seed_start(service, campaign=campaign, operation=operation, attempt=attempt, result=result)
    return result, f"{attempt}-publication"


def _deny(call) -> None:
    with pytest.raises(TrainerError) as raised:
        call()
    assert raised.value.code == "publication_readback_denied"


def _managed_guard() -> ManagedChannelGuard:
    context = ManagedChannelContext(rm_url="http://rm.invalid")

    def expectation(operation: str, method: str) -> dict[str, object]:
        return {
            "destinationInstanceId": "trainer-one", "destinationInstanceGeneration": 1,
            "destinationLeaseId": "trainer-lease", "operation": operation, "method": method,
        }

    def introspect(credential: str, expected: dict[str, object]) -> dict[str, object]:
        project, deployment = {
            "owner": ("project-a", "deployment-a"),
            "wrong-project": ("project-b", "deployment-b"),
            "wrong-deployment": ("project-a", "deployment-b"),
        }.get(credential, ("project-a", "deployment-a"))
        operation = "trainer.campaign:execution" if credential == "wrong-operation" else expected["operation"]
        return {
            "schemaVersion": "jester-channel-session-claims-v1", "active": True,
            "sessionId": f"session-{credential}", "projectId": project,
            "deploymentId": deployment, "deploymentGeneration": 1,
            "engineInstanceId": "engine-one", "engineInstanceGeneration": 1,
            "engineLeaseId": "engine-lease", "channelId": "engine-to-trainer",
            "destinationMemberId": "trainer", "destinationInstanceId": expected["destinationInstanceId"],
            "destinationInstanceGeneration": expected["destinationInstanceGeneration"],
            "destinationLeaseId": expected["destinationLeaseId"], "operation": operation,
            "method": expected["method"], "expiresAt": "2030-01-01T00:00:00+00:00",
        }

    return ManagedChannelGuard(context, introspect=introspect, expectation_provider=expectation)


def test_publication_readback_returns_only_one_verified_pair_and_keeps_audit_redacted(tmp_path) -> None:
    service = _service(tmp_path, "campaign-a")
    result_id, publication_id = _seed(service)
    try:
        readback = service.publication_readback(
            "campaign-a", result_id, publication_id, operation_id="start-a",
            expected_project_id=OWNER["projectId"], expected_deployment_id=OWNER["deploymentId"],
        )
        assert readback["schemaVersion"] == "jester-trainer-publication-readback-v1"
        assert readback["identity"] == {
            "projectId": "project-a", "deploymentId": "deployment-a",
            "campaignId": "campaign-a", "startOperationId": "start-a",
        }
        assert readback["result"]["facetReference"] == "facet/private/path"
        assert readback["result"]["artifactSha256"] == "5" * 64
        assert readback["result"]["artifactDigest"] == "sha256:" + "6" * 64
        assert readback["publication"]["state"] == "acknowledged"
        assert set(readback) == {"schemaVersion", "identity", "result", "publication"}
        encoded = json.dumps(readback, sort_keys=True)
        for forbidden in ("artifactPath", "C:/secret/model", "metrics", "rawWeights", "token"):
            assert forbidden not in encoded
        audit = service.campaign_execution(
            "campaign-a", operation_id="start-a", expected_project_id=OWNER["projectId"],
            expected_deployment_id=OWNER["deploymentId"],
        )["audit"]
        assert "facet/private/path" not in json.dumps(audit, sort_keys=True)
        assert audit["results"][0]["facetReferenceSha256"]
    finally:
        service.close()


def test_publication_readback_denies_missing_stale_failed_cancelled_and_digest_mismatch(tmp_path) -> None:
    service = _service(tmp_path, "campaign-a")
    result_id, publication_id = _seed(service)
    arguments = {"operation_id": "start-a", "expected_project_id": OWNER["projectId"], "expected_deployment_id": OWNER["deploymentId"]}
    try:
        _deny(lambda: service.publication_readback("campaign-a", "missing", publication_id, **arguments))
        _deny(lambda: service.publication_readback("campaign-a", result_id, publication_id, operation_id="stale-start", expected_project_id=OWNER["projectId"], expected_deployment_id=OWNER["deploymentId"]))

        failed = _service(tmp_path / "failed", "campaign-failed")
        failed_result, failed_publication = _seed(failed, campaign="campaign-failed", operation="start-failed")
        failed_operation = failed.store.get_operation("start-failed")
        failed.store.put_operation({**failed_operation, "state": "failed"})
        _deny(lambda: failed.publication_readback("campaign-failed", failed_result, failed_publication, operation_id="start-failed", expected_project_id=OWNER["projectId"], expected_deployment_id=OWNER["deploymentId"]))
        failed.close()

        cancelled = _service(tmp_path / "cancelled", "campaign-cancelled")
        cancelled_result, cancelled_publication = _seed(cancelled, campaign="campaign-cancelled", operation="start-cancelled")
        campaign = cancelled.store.get_entity("campaign", "campaign-cancelled")
        cancelled.store.put_entity("campaign", "campaign-cancelled", {**campaign, "state": "cancelled"}, expected_generation=campaign["generation"])
        _deny(lambda: cancelled.publication_readback("campaign-cancelled", cancelled_result, cancelled_publication, operation_id="start-cancelled", expected_project_id=OWNER["projectId"], expected_deployment_id=OWNER["deploymentId"]))
        cancelled.close()

        publication = service.store.get_entity("publication", publication_id)
        service.store.put_entity("publication", publication_id, {**publication, "artifactDigest": "sha256:" + "7" * 64}, expected_generation=publication["generation"])
        _deny(lambda: service.publication_readback("campaign-a", result_id, publication_id, **arguments))
    finally:
        service.close()


def test_publication_readback_denies_ambiguous_audit_pair(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, "campaign-a")
    result_id, publication_id = _seed(service)
    original = service.campaign_execution

    def duplicate(*args, **kwargs):
        execution = copy.deepcopy(original(*args, **kwargs))
        execution["audit"]["results"].append(copy.deepcopy(execution["audit"]["results"][0]))
        return execution

    monkeypatch.setattr(service, "campaign_execution", duplicate)
    try:
        _deny(lambda: service.publication_readback("campaign-a", result_id, publication_id, operation_id="start-a", expected_project_id=OWNER["projectId"], expected_deployment_id=OWNER["deploymentId"]))
    finally:
        service.close()


def test_publication_readback_refuses_malformed_lineage_and_raw_identity_mismatch(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, "campaign-a")
    result_id, publication_id = _seed(service)
    original_execution = service.campaign_execution
    original_result = service.get_result

    def malformed_lineage(*args, **kwargs):
        execution = copy.deepcopy(original_execution(*args, **kwargs))
        execution["audit"]["results"][0]["lineage"] = ["not-a-lineage"]
        return execution

    monkeypatch.setattr(service, "campaign_execution", malformed_lineage)
    try:
        _deny(lambda: service.publication_readback(
            "campaign-a", result_id, publication_id, operation_id="start-a",
            expected_project_id=OWNER["projectId"], expected_deployment_id=OWNER["deploymentId"],
        ))
    finally:
        monkeypatch.setattr(service, "campaign_execution", original_execution)

    def wrong_raw_result(requested_id: str) -> dict:
        result = original_result(requested_id)
        return {**result, "resultId": "foreign-result"}

    monkeypatch.setattr(service, "get_result", wrong_raw_result)
    try:
        _deny(lambda: service.publication_readback(
            "campaign-a", result_id, publication_id, operation_id="start-a",
            expected_project_id=OWNER["projectId"], expected_deployment_id=OWNER["deploymentId"],
        ))
    finally:
        service.close()


def test_publication_readback_route_requires_exact_managed_session_and_owner(tmp_path) -> None:
    service = _service(tmp_path, "campaign-a")
    result_id, publication_id = _seed(service)
    path = f"/v1/campaigns/campaign-a/results/{result_id}/publications/{publication_id}/readback?operationId=start-a"
    try:
        with TestClient(create_app(service, managed_channel_guard=_managed_guard(), require_managed_session=True)) as client:
            owner = client.get(path, headers={"X-Jester-Component-Session": "owner"})
            foreign_project = client.get(path, headers={"X-Jester-Component-Session": "wrong-project"})
            foreign_deployment = client.get(path, headers={"X-Jester-Component-Session": "wrong-deployment"})
            wrong_operation = client.get(path, headers={"X-Jester-Component-Session": "wrong-operation"})
            bearer = client.get(path, headers={"Authorization": f"Bearer {service.token}"})
            unmanaged = client.get(f"/v1/results/{result_id}", headers={"X-Jester-Component-Session": "owner"})
        assert owner.status_code == 200
        assert owner.json()["result"]["resultId"] == result_id
        for response in (foreign_project, foreign_deployment, wrong_operation, bearer, unmanaged):
            assert response.status_code == 401
            assert response.json()["code"] == "channel_session_denied"
    finally:
        service.close()
