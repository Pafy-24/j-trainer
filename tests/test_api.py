from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from jester_trainer.api import create_app
from jester_trainer.config import Settings
from jester_trainer.channel_sessions import ManagedChannelContext, ManagedChannelGuard
from jester_trainer.service import TrainerService


def test_api_auth_and_dataset_idempotency(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    data.write_text(
        json.dumps({"id": "one", "semanticCaseId": "case-one", "text": "hello"})
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    service = TrainerService(Settings.for_test(tmp_path / "trainer"))
    headers = {
        "Authorization": f"Bearer {service.token}",
        "Idempotency-Key": "register-dataset-once",
    }
    with TestClient(create_app(service)) as client:
        assert client.get("/v1/health").status_code == 200
        assert client.get("/v1/datasets").status_code == 401
        payload = {
            "apiVersion": "jester.trainer/v1",
            "kind": "DatasetManifest",
            "metadata": {"id": "tiny-dataset"},
            "files": [
                {
                    "split": "train",
                    "digest": f"sha256:{digest}",
                    "path": str(data),
                    "records": 1,
                }
            ],
        }
        first = client.post("/v1/datasets", headers=headers, json=payload)
        second = client.post("/v1/datasets", headers=headers, json=payload)
        assert first.status_code == 201
        assert second.status_code == 201
        assert first.json() == second.json()
        changed = {**payload, "seed": 99}
        conflict = client.post("/v1/datasets", headers=headers, json=changed)
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "idempotency_conflict"
        listed = client.get(
            "/v1/datasets", headers={"Authorization": f"Bearer {service.token}"}
        )
        assert listed.status_code == 200
        assert [item["metadata"]["id"] for item in listed.json()["items"]] == [
            "tiny-dataset"
        ]


def test_managed_api_fails_closed_without_a_session_and_never_uses_bearer(tmp_path: Path) -> None:
    service = TrainerService(Settings.for_test(tmp_path / "trainer"))
    context = ManagedChannelContext(rm_url="http://rm.invalid")
    def local(operation: str, method: str) -> dict[str, object]:
        return {"destinationInstanceId": "trainer-one", "destinationInstanceGeneration": 1, "destinationLeaseId": "trainer-lease", "operation": operation, "method": method}
    def introspect(_credential: str, expected: dict[str, object]) -> dict[str, object]:
        return {"schemaVersion":"jester-channel-session-claims-v1", "active":True, "sessionId":"session", "projectId":"project-one", "deploymentId":"deployment-one", "deploymentGeneration":1, "engineInstanceId":"engine-one", "engineInstanceGeneration":1, "engineLeaseId":"engine-lease", "channelId":"trainer-channel", "destinationMemberId":"trainer", "destinationInstanceId":expected["destinationInstanceId"], "destinationInstanceGeneration":expected["destinationInstanceGeneration"], "destinationLeaseId":expected["destinationLeaseId"], "operation":expected["operation"], "method":expected["method"], "expiresAt":"2030-01-01T00:00:00+00:00"}
    guard = ManagedChannelGuard(context, introspect=introspect, expectation_provider=local)
    with TestClient(create_app(service, managed_channel_guard=guard, require_managed_session=True)) as client:
        assert client.get("/v1/datasets", headers={"Authorization": f"Bearer {service.token}"}).status_code == 401
        assert client.get("/v1/campaigns/not-allowed", headers={"X-Jester-Component-Session": "opaque-session-token-at-least-24-chars"}).status_code == 401
        assert client.post("/v1/campaigns", headers={"X-Jester-Component-Session": "opaque-session-token-at-least-24-chars"}, json={}).status_code == 422


def test_managed_campaign_owner_rejects_cross_owner_replays_and_survives_restart(
    tmp_path: Path, monkeypatch,
) -> None:
    from tests.helpers import campaign
    data = tmp_path / "data.jsonl"
    data.write_text('{"id":"x","text":"x"}\n', encoding="utf-8")
    topology = {
        "topologyId": "test-training-topology",
        "topologyType": "training",
        "role": "trainer",
        "nodeId": "trainer",
        "channels": [],
        "nodes": [],
    }
    settings = Settings.for_test(tmp_path / "trainer", topology=topology)
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    artifacts = tmp_path / "artifacts.json"
    artifacts.write_text(json.dumps({"entries": [{
        "digest": f"sha256:{digest}", "kind": "file", "checkoutPath": str(data),
    }]}), encoding="utf-8")
    monkeypatch.setenv("JESTER_RM_ARTIFACT_MANIFEST", str(artifacts))
    service = TrainerService(settings)

    def local(operation: str, method: str) -> dict[str, object]:
        return {
            "destinationInstanceId": "trainer",
            "destinationInstanceGeneration": 1,
            "destinationLeaseId": "lease",
            "operation": operation,
            "method": method,
        }

    introspection_calls: list[tuple[str, str]] = []

    def introspect(credential: str, expected: dict[str, object]) -> dict[str, object]:
        introspection_calls.append((str(expected["operation"]), str(expected["method"])))
        project, deployment = (
            ("project-a", "deployment-a")
            if credential == "owner-a"
            else ("project-b", "deployment-b")
        )
        return {
            "schemaVersion": "jester-channel-session-claims-v1",
            "active": True,
            "sessionId": credential,
            "projectId": project,
            "deploymentId": deployment,
            "deploymentGeneration": 1,
            "engineInstanceId": "engine",
            "engineInstanceGeneration": 1,
            "engineLeaseId": "engine-lease",
            "channelId": "channel",
            "destinationMemberId": "trainer",
            "destinationInstanceId": "trainer",
            "destinationInstanceGeneration": 1,
            "destinationLeaseId": "lease",
            "operation": expected["operation"],
            "method": expected["method"],
            "expiresAt": "2030-01-01T00:00:00+00:00",
        }

    guard = ManagedChannelGuard(
        ManagedChannelContext(rm_url="http://unused"),
        introspect=introspect,
        expectation_provider=local,
    )
    owner_a_campaign = campaign(
        tmp_path, tmp_path / "model", campaign_id="campaign-one", version="v1", workers=[]
    ).api_dict()
    owner_b_campaign = campaign(
        tmp_path, tmp_path / "model", campaign_id="campaign-two", version="v2", workers=[]
    ).api_dict()
    owner_a_headers = {"X-Jester-Component-Session": "owner-a", "Idempotency-Key": "same"}
    owner_b_headers = {"X-Jester-Component-Session": "owner-b", "Idempotency-Key": "same"}

    with TestClient(
        create_app(service, managed_channel_guard=guard, require_managed_session=True)
    ) as client:
        registered = client.post("/v1/datasets", headers=owner_a_headers, json={
            "metadata": {"id": "tiny-dataset"},
            "files": [{
                "split": "train", "digest": f"sha256:{digest}",
                "artifactDigest": f"sha256:{digest}", "logicalPath": "data.jsonl",
            }],
        })
        assert registered.status_code == 201
        created = client.post("/v1/campaigns", headers=owner_a_headers, json=owner_a_campaign)
        assert created.status_code == 201
        assert created.json()["metadata"]["id"] == "campaign-one"
        assert introspection_calls == [
            ("trainer.dataset:register", "POST"),
            ("trainer.campaign:create", "POST"),
        ]

        def queued_start(
            campaign_id: str, *, operation_id: str | None = None
        ) -> dict[str, object]:
            operation = {
                "operationId": operation_id,
                "action": "campaign.start",
                "targetId": campaign_id,
                "state": "queued",
                "createdAt": "2026-09-20T00:00:00Z",
                "updatedAt": "2026-09-20T00:00:00Z",
            }
            service.store.put_operation(operation)
            return operation

        service.start_campaign = queued_start  # type: ignore[method-assign]
        started = client.post(
            "/v1/campaigns/campaign-one/actions/start",
            headers={**owner_a_headers, "Idempotency-Key": "start-one"},
            json={"operationId": "engine.trainer." + "a" * 32},
        )
        denied_start = client.post(
            "/v1/campaigns/campaign-one/actions/start",
            headers={**owner_b_headers, "Idempotency-Key": "start-two"},
            json={"operationId": "engine.trainer." + "b" * 32},
        )
        execution = client.get(
            "/v1/campaigns/campaign-one/execution?operationId=engine.trainer."
            + "a" * 32,
            headers={"X-Jester-Component-Session": "owner-a"},
        )
        denied_execution = client.get(
            "/v1/campaigns/campaign-one/execution?operationId=engine.trainer."
            + "a" * 32,
            headers={"X-Jester-Component-Session": "owner-b"},
        )
        assert started.status_code == 202
        assert started.json()["operationId"] == "engine.trainer." + "a" * 32
        assert denied_start.status_code == 401
        assert execution.status_code == 200
        assert execution.json()["operation"]["operationId"] == "engine.trainer." + "a" * 32
        assert execution.json()["schemaVersion"] == "jester-trainer-campaign-execution-v1"
        assert denied_execution.status_code == 401
        assert introspection_calls[-4:] == [
            ("trainer.campaign:start", "POST"),
            ("trainer.campaign:start", "POST"),
            ("trainer.campaign:execution", "GET"),
            ("trainer.campaign:execution", "GET"),
        ]

        # Both replays are rejected before the idempotency result can expose owner A's campaign.
        same_key = client.post("/v1/campaigns", headers=owner_b_headers, json=owner_a_campaign)
        new_key = client.post(
            "/v1/campaigns",
            headers={**owner_b_headers, "Idempotency-Key": "new"},
            json=owner_a_campaign,
        )
        assert same_key.status_code == 403
        assert same_key.json()["code"] == "channel_session_denied"
        assert new_key.status_code == 403
        assert new_key.json()["code"] == "channel_session_denied"

        # A global idempotency key must not return owner A's result to owner B.
        different_campaign = client.post(
            "/v1/campaigns", headers=owner_b_headers, json=owner_b_campaign
        )
        assert different_campaign.status_code == 403
        assert different_campaign.json()["code"] == "channel_session_denied"

    service.close()
    reopened = TrainerService(settings)
    with TestClient(
        create_app(reopened, managed_channel_guard=guard, require_managed_session=True)
    ) as client:
        calls_before_status = len(introspection_calls)
        owner_a_status = client.get(
            "/v1/campaigns/campaign-one",
            headers={"X-Jester-Component-Session": "owner-a"},
        )
        owner_b_status = client.get(
            "/v1/campaigns/campaign-one",
            headers={"X-Jester-Component-Session": "owner-b"},
        )
        assert owner_a_status.status_code == 200
        assert owner_a_status.json()["metadata"]["id"] == "campaign-one"
        assert owner_b_status.status_code == 401
        # The middleware validates status once and the handler reuses those claims.
        assert len(introspection_calls) == calls_before_status + 2
        nested = client.get(
            "/v1/campaigns/campaign-one/events",
            headers={"X-Jester-Component-Session": "owner-a"},
        )
        assert nested.status_code == 401
        assert len(introspection_calls) == calls_before_status + 2
    reopened.close()


def test_managed_create_cannot_claim_existing_unowned_legacy_campaign(tmp_path: Path) -> None:
    from tests.helpers import campaign, dataset_manifest

    data = tmp_path / "data.jsonl"
    data.write_text('{"id":"legacy","text":"x"}\n', encoding="utf-8")
    settings = Settings.for_test(
        tmp_path / "trainer",
        topology={
            "topologyId": "test-training-topology",
            "topologyType": "training",
            "role": "trainer",
            "nodeId": "trainer",
            "channels": [],
            "nodes": [],
        },
    )
    service = TrainerService(settings)
    service.put_dataset(dataset_manifest(data))
    legacy = campaign(
        tmp_path,
        tmp_path / "model",
        campaign_id="legacy-campaign",
        version="v1",
        workers=[],
    )
    original = service.put_campaign(legacy)

    def local(operation: str, method: str) -> dict[str, object]:
        return {
            "destinationInstanceId": "trainer",
            "destinationInstanceGeneration": 1,
            "destinationLeaseId": "lease",
            "operation": operation,
            "method": method,
        }

    def introspect(_credential: str, expected: dict[str, object]) -> dict[str, object]:
        return {
            "schemaVersion": "jester-channel-session-claims-v1",
            "active": True,
            "sessionId": "owner-a",
            "projectId": "project-a",
            "deploymentId": "deployment-a",
            "deploymentGeneration": 1,
            "engineInstanceId": "engine",
            "engineInstanceGeneration": 1,
            "engineLeaseId": "engine-lease",
            "channelId": "channel",
            "destinationMemberId": "trainer",
            "destinationInstanceId": "trainer",
            "destinationInstanceGeneration": 1,
            "destinationLeaseId": "lease",
            "operation": expected["operation"],
            "method": expected["method"],
            "expiresAt": "2030-01-01T00:00:00+00:00",
        }

    guard = ManagedChannelGuard(
        ManagedChannelContext(rm_url="http://unused"),
        introspect=introspect,
        expectation_provider=local,
    )
    with TestClient(
        create_app(service, managed_channel_guard=guard, require_managed_session=True)
    ) as client:
        denied = client.post(
            "/v1/campaigns",
            headers={"X-Jester-Component-Session": "owner-a", "Idempotency-Key": "claim"},
            json=legacy.api_dict(),
        )
        assert denied.status_code == 403
        assert denied.json()["code"] == "channel_session_denied"
        assert service.store.campaign_owner("legacy-campaign") is None
        assert service.get_campaign("legacy-campaign").api_dict() == original
