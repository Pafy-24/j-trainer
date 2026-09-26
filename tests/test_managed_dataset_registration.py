from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest
from fastapi.testclient import TestClient

from jester_trainer.api import create_app
from jester_trainer.channel_sessions import ManagedChannelContext, ManagedChannelGuard
from jester_trainer.config import Settings
from jester_trainer.errors import ConflictError, TrainerError
from jester_trainer.models import Campaign, DatasetManifest
from jester_trainer.service import TrainerService


def _guard() -> ManagedChannelGuard:
    def local(operation: str, method: str) -> dict[str, object]:
        return {
            "destinationInstanceId": "trainer",
            "destinationInstanceGeneration": 1,
            "destinationLeaseId": "lease.trainer",
            "operation": operation,
            "method": method,
        }

    def introspect(credential: str, expected: dict[str, object]) -> dict[str, object]:
        project, deployment = {
            "owner-a": ("project-a", "deployment-a"),
            "owner-a-other-deployment": ("project-a", "deployment-other"),
            "owner-b": ("project-b", "deployment-b"),
        }[credential]
        return {
            "schemaVersion": "jester-channel-session-claims-v1",
            "active": True,
            "sessionId": credential,
            "projectId": project,
            "deploymentId": deployment,
            "deploymentGeneration": 1,
            "engineInstanceId": "engine",
            "engineInstanceGeneration": 1,
            "engineLeaseId": "lease.engine",
            "channelId": "engine-trainer",
            "destinationMemberId": "trainer",
            **expected,
            "expiresAt": "2030-01-01T00:00:00+00:00",
        }

    return ManagedChannelGuard(
        ManagedChannelContext(rm_url="http://unused"),
        introspect=introspect,
        expectation_provider=local,
    )


def _service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TrainerService, dict[str, Any], dict[str, Any]]:
    data = tmp_path / "train.jsonl"
    data.write_text(
        '{"id":"one","semanticCaseId":"case-one","text":"hello"}\n',
        encoding="utf-8",
    )
    changed_data = tmp_path / "train-v2.jsonl"
    changed_data.write_text(
        '{"id":"one","semanticCaseId":"case-one","text":"changed"}\n',
        encoding="utf-8",
    )
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    changed_digest = hashlib.sha256(changed_data.read_bytes()).hexdigest()
    artifact_manifest = tmp_path / "rm-artifacts.json"
    artifact_manifest.write_text(
        json.dumps({
            "entries": [
                {
                    "digest": f"sha256:{digest}",
                    "kind": "file",
                    "checkoutPath": str(data),
                },
                {
                    "digest": f"sha256:{changed_digest}",
                    "kind": "file",
                    "checkoutPath": str(changed_data),
                },
            ]
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("JESTER_RM_ARTIFACT_MANIFEST", str(artifact_manifest))
    topology = {
        "topologyId": "test-training-topology",
        "topologyType": "training",
        "role": "trainer",
        "nodeId": "trainer",
        "channels": [],
        "nodes": [],
    }
    service = TrainerService(Settings.for_test(tmp_path / "trainer", topology=topology))
    manifest = {
        "metadata": {"id": "tiny-dataset"},
        "files": [{
            "split": "train",
            "digest": digest.upper(),
            "artifactDigest": f"SHA256:{digest.upper()}",
            "logicalPath": "train.jsonl",
        }],
    }
    changed_manifest = {
        "metadata": {"id": "tiny-dataset"},
        "files": [{
            "split": "train",
            "digest": changed_digest,
            "artifactDigest": f"sha256:{changed_digest}",
            "logicalPath": "train-v2.jsonl",
        }],
    }
    return service, manifest, changed_manifest


def _campaign(
    campaign_id: str,
    tmp_path: Path,
    *,
    global_dataset_ids: list[str],
    facet_dataset_ids: list[str] | None = None,
) -> Campaign:
    return Campaign.model_validate({
        "metadata": {"id": campaign_id},
        "spec": {
            "topologyId": "test-training-topology",
            "datasets": [
                {"datasetId": dataset_id, "split": "train"}
                for dataset_id in global_dataset_ids
            ],
            "facets": [{
                "id": "tiny",
                "miConfig": {},
                "datasets": [
                    {"datasetId": dataset_id, "split": "train"}
                    for dataset_id in (facet_dataset_ids or [])
                ],
            }],
            "training": {"maxSteps": 1},
            "checkpoint": {"outputDirectory": str(tmp_path / f"{campaign_id}-results")},
        },
    })


def test_registration_is_concurrently_idempotent_across_store_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, payload, _changed_payload = _service(tmp_path, monkeypatch)
    second = TrainerService(first.settings)
    gate = Barrier(2)
    manifest = DatasetManifest.model_validate(payload)

    def register(service: TrainerService) -> dict[str, Any]:
        gate.wait(timeout=5)
        return service.register_managed_dataset(
            manifest, project_id="project-a", deployment_id="deployment-a"
        )

    try:
        assert first.store is not second.store
        assert first.store.connection is not second.store.connection
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(register, service) for service in (first, second)]
            results = [future.result(timeout=10) for future in futures]
        assert results[0]["manifestSha256"] == results[1]["manifestSha256"]
        assert first.store.dataset_owner("tiny-dataset") == second.store.dataset_owner(
            "tiny-dataset"
        )
        assert first.store.connection.execute(
            "SELECT COUNT(*) FROM dataset_owners WHERE dataset_id=?", ("tiny-dataset",)
        ).fetchone()[0] == 1
        assert first.store.connection.execute(
            "SELECT COUNT(*) FROM entities WHERE kind='dataset' AND entity_id=?",
            ("tiny-dataset",),
        ).fetchone()[0] == 1
    finally:
        second.close()
        first.close()


def test_pristine_dataset_denies_foreign_project_and_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, payload, _changed_payload = _service(tmp_path, monkeypatch)
    service.register_managed_dataset(
        DatasetManifest.model_validate(payload),
        project_id="project-a",
        deployment_id="deployment-a",
    )
    with TestClient(
        create_app(service, managed_channel_guard=_guard(), require_managed_session=True)
    ) as client:
        denied_project = client.get(
            "/v1/datasets/tiny-dataset",
            headers={"X-Jester-Component-Session": "owner-b"},
        )
        denied_deployment = client.get(
            "/v1/datasets/tiny-dataset",
            headers={"X-Jester-Component-Session": "owner-a-other-deployment"},
        )
        assert denied_project.status_code == 401
        assert denied_project.json()["code"] == "channel_session_denied"
        assert denied_deployment.status_code == 401
        assert denied_deployment.json()["code"] == "channel_session_denied"


def test_same_owner_changed_manifest_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, payload, changed_payload = _service(tmp_path, monkeypatch)
    try:
        service.register_managed_dataset(
            DatasetManifest.model_validate(payload),
            project_id="project-a",
            deployment_id="deployment-a",
        )
        with pytest.raises(ConflictError) as changed:
            service.register_managed_dataset(
                DatasetManifest.model_validate(changed_payload),
                project_id="project-a",
                deployment_id="deployment-a",
            )
        assert changed.value.code == "dataset_manifest_conflict"
    finally:
        service.close()


def test_cross_deployment_registration_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, payload, _changed_payload = _service(tmp_path, monkeypatch)
    manifest = DatasetManifest.model_validate(payload)
    try:
        service.register_managed_dataset(
            manifest, project_id="project-a", deployment_id="deployment-a"
        )
        with pytest.raises(ConflictError) as foreign_deployment:
            service.register_managed_dataset(
                manifest,
                project_id="project-a",
                deployment_id="deployment-other",
            )
        assert foreign_deployment.value.code == "dataset_owner_conflict"
    finally:
        service.close()


def test_unmanaged_dataset_cannot_be_claimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, payload, _changed_payload = _service(tmp_path, monkeypatch)
    manifest = DatasetManifest.model_validate(payload)
    try:
        unmanaged = service.put_dataset(manifest)
        with pytest.raises(ConflictError) as claim:
            service.register_managed_dataset(
                manifest, project_id="project-a", deployment_id="deployment-a"
            )
        assert claim.value.code == "dataset_owner_missing"
        assert service.store.dataset_owner("tiny-dataset") is None
        assert service.get_dataset("tiny-dataset").api_dict() == unmanaged
    finally:
        service.close()


def test_foreign_facet_dataset_rejects_campaign_without_partial_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, payload, _changed_payload = _service(tmp_path, monkeypatch)
    manifest = DatasetManifest.model_validate(payload)
    try:
        service.register_managed_dataset(
            manifest, project_id="project-b", deployment_id="deployment-b"
        )
        campaign = _campaign(
            "campaign-foreign-facet",
            tmp_path,
            global_dataset_ids=[],
            facet_dataset_ids=["tiny-dataset"],
        )
        with pytest.raises(TrainerError) as denied:
            service.put_managed_campaign(
                campaign, project_id="project-a", deployment_id="deployment-a"
            )
        assert denied.value.code == "channel_session_denied"
        assert service.store.campaign_owner("campaign-foreign-facet") is None
        assert service.store.connection.execute(
            "SELECT COUNT(*) FROM entities WHERE kind='campaign' AND entity_id=?",
            ("campaign-foreign-facet",),
        ).fetchone()[0] == 0
        assert service.store.connection.execute(
            "SELECT COUNT(*) FROM campaign_dataset_bindings WHERE campaign_id=?",
            ("campaign-foreign-facet",),
        ).fetchone()[0] == 0
    finally:
        service.close()


def test_managed_start_refuses_missing_binding_and_entity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, payload, _changed_payload = _service(tmp_path, monkeypatch)
    try:
        service.register_managed_dataset(
            DatasetManifest.model_validate(payload),
            project_id="project-a",
            deployment_id="deployment-a",
        )
        old_managed = _campaign(
            "campaign-old-managed", tmp_path, global_dataset_ids=["tiny-dataset"]
        )
        service.put_campaign(old_managed)
        with service.store.connection:
            service.store.connection.execute(
                "INSERT INTO campaign_owners(campaign_id,project_id,deployment_id,created_at) VALUES(?,?,?,?)",
                (
                    "campaign-old-managed",
                    "project-a",
                    "deployment-a",
                    "2026-09-26T00:00:00Z",
                ),
            )
        with pytest.raises(ConflictError) as missing_binding:
            service.start_campaign(
                "campaign-old-managed", operation_id="engine.trainer." + "b" * 32
            )
        assert missing_binding.value.code == "campaign_dataset_binding_missing"
        assert service.store.active_campaign_start("campaign-old-managed") is None

        managed_campaign = _campaign(
            "campaign-one", tmp_path, global_dataset_ids=["tiny-dataset"]
        )
        service.put_managed_campaign(
            managed_campaign, project_id="project-a", deployment_id="deployment-a"
        )
        drifted = service.get_dataset("tiny-dataset").api_dict()
        drifted["seed"] = 99
        service.store.put_entity(
            "dataset", "tiny-dataset", drifted, expected_generation=1
        )
        with pytest.raises(ConflictError) as changed:
            service.start_campaign(
                "campaign-one", operation_id="engine.trainer." + "a" * 32
            )
        assert changed.value.code == "campaign_dataset_binding_changed"
        assert service.store.active_campaign_start("campaign-one") is None
    finally:
        service.close()
