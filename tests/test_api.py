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
    context = ManagedChannelContext(
        session_id="session-one", project_id="project-one", deployment_id="deployment-one", deployment_generation=1,
        engine_instance_id="engine-one", engine_instance_generation=1, engine_lease_id="engine-lease",
        channel_id="channel-one", destination_member_id="trainer", destination_instance_id="trainer-one",
        destination_instance_generation=1, destination_lease_id="trainer-lease", allowed_campaign_ids=frozenset({"campaign-one"}),
    )
    def introspect(_credential: str, expected: dict[str, object]) -> dict[str, object]:
        return {
            "schemaVersion": "jester-channel-session-claims-v1", "active": True,
            "sessionId": expected["sessionId"], "projectId": expected["projectId"],
            "deploymentId": expected["deploymentId"], "deploymentGeneration": expected["deploymentGeneration"],
            "engineInstanceId": expected["engineInstanceId"], "engineInstanceGeneration": expected["engineInstanceGeneration"],
            "engineLeaseId": expected["engineLeaseId"], "channelId": expected["channelId"],
            "destinationMemberId": expected["destinationMemberId"], "destinationInstanceId": expected["destinationInstanceId"],
            "destinationInstanceGeneration": expected["destinationInstanceGeneration"], "destinationLeaseId": expected["destinationLeaseId"],
            "operation": expected["operation"], "method": expected["method"], "expiresAt": "2030-01-01T00:00:00+00:00",
        }
    guard = ManagedChannelGuard(context, introspect=introspect)
    with TestClient(create_app(service, managed_channel_guard=guard, require_managed_session=True)) as client:
        assert client.get("/v1/datasets", headers={"Authorization": f"Bearer {service.token}"}).status_code == 401
        assert client.get("/v1/campaigns/not-allowed", headers={"X-Jester-Component-Session": "opaque-session-token-at-least-24-chars"}).status_code == 401
        # Session is accepted at middleware; payload validation follows normally.
        assert client.post("/v1/campaigns", headers={"X-Jester-Component-Session": "opaque-session-token-at-least-24-chars"}, json={}).status_code == 422
