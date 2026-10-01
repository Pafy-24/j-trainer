from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from jester_trainer.config import Settings
from jester_trainer.errors import ConflictError
from jester_trainer.models import DatasetManifest, TrainingJob
from jester_trainer.service import TrainerService

from .helpers import campaign as campaign_fixture
from .helpers import tiny_model, training_dataset, worker_topology


OWNER = {"projectId": "project-a", "deploymentId": "deployment-a"}


def _lineage(
    campaign: str, operation: str, job: str, attempt: str, *, owner: dict = OWNER
) -> dict:
    return {
        "schemaVersion": "jester-trainer-execution-lineage-v1",
        **owner,
        "campaignId": campaign,
        "startOperationId": operation,
        "jobId": job,
        "attemptId": attempt,
    }


def _seed_start(
    service: TrainerService, *, campaign: str, operation: str,
    attempt: str, result: str, state: str = "succeeded", owner: dict = OWNER,
) -> None:
    store = service.store
    job = f"{campaign}-job"
    lineage = _lineage(campaign, operation, job, attempt, owner=owner)
    store.put_operation({
        "operationId": operation, "action": "campaign.start", "targetId": campaign,
        "state": state, "createdAt": "2026-09-26T00:00:00Z",
        "updatedAt": "2026-09-26T00:00:01Z",
        "result": {"secret": "must-not-leak"},
    })
    if not any(item.get("jobId") == job for item in store.list_entities("job")):
        store.put_entity("job", job, {
            "jobId": job, "campaignId": campaign, "facetId": "facet-a",
            "candidateId": "candidate-a", "state": "succeeded", "workerIds": ["w1", "w2"],
            "attempt": 1, "resultId": result, "config": {"token": "secret"},
            "error": {"path": "C:/secret"},
        })
    store.put_entity("attempt", attempt, {
        "attemptId": attempt, "jobId": job, "number": 1,
        "cohort": ["w1", "w2"], "state": "succeeded", "lineage": lineage,
    })
    store.put_entity("round", f"{attempt}-r0", {
        "roundId": f"{attempt}-r0", "attemptId": attempt, "index": 0,
        "state": "committed", "weightVersion": "v1", "lineage": lineage,
        "contributions": [
            {"workerId": worker, "gradientSha256": character * 64,
             "supervisedTokens": 2, "examples": 2, "meanLoss": 0.5}
            for worker, character in (("w1", "1"), ("w2", "2"))
        ],
        "aggregateSha256": "3" * 64,
        "error": {"tensor": [1, 2, 3]},
    })
    for worker in ("w1", "w2"):
        session = f"{attempt}-{worker}"
        store.put_entity("worker-session", session, {
            "schemaVersion": "jester-trainer-worker-session-observation-v1",
            "id": session, "campaignId": campaign, "jobId": job,
            "attemptId": attempt, "workerId": worker, "sessionId": session,
            "createState": "acknowledged", "cleanupState": "acknowledged",
            "createAck": {"id": session, "state": "ready", "weightVersion": "v1", "weightSha256": "4" * 64},
            "cleanupAck": {"id": session, "deleted": True, "artifactsRetained": True},
            "cleanupError": {"secret": "must-not-leak"}, "lineage": lineage,
        })
    store.put_entity("result", result, {
        "resultId": result, "campaignId": campaign, "jobId": job, "lineage": lineage,
        "facetReference": "facet/private/path", "artifactPath": "C:/secret/model",
        "artifactSha256": "5" * 64, "artifactDigest": "sha256:" + "6" * 64,
        "metrics": {"rawWeights": [1, 2, 3], "token": "secret"},
    })
    store.put_entity("publication", f"{attempt}-publication", {
        "schemaVersion": "jester-trainer-artifact-publication-v1",
        "kind": "ArtifactPublication", "publicationId": f"{attempt}-publication",
        "resultId": result, "state": "acknowledged", "lineage": lineage,
        "artifactSha256": "5" * 64, "artifactDigest": "sha256:" + "6" * 64,
    })


def _service(tmp_path: Path, campaign: str, *, owner: dict = OWNER) -> TrainerService:
    service = TrainerService(Settings.for_test(tmp_path / "trainer"))
    service.store.bind_campaign_owner(
        campaign, project_id=owner["projectId"], deployment_id=owner["deploymentId"]
    )
    service.store.put_entity("campaign", campaign, {
        "metadata": {"id": campaign}, "state": "succeeded", "spec": {"secret": "redacted"}
    })
    return service


def test_two_starts_and_attempt_scoped_results_are_isolated_and_redacted(tmp_path: Path) -> None:
    service = _service(tmp_path, "campaign-a")
    _seed_start(
        service, campaign="campaign-a", operation="start-one",
        attempt="campaign-a-job-a1", result="campaign-a-job-a1-result",
    )
    _seed_start(
        service, campaign="campaign-a", operation="start-two",
        attempt="campaign-a-job-a2", result="campaign-a-job-a2-result",
    )

    first = service.campaign_execution("campaign-a", operation_id="start-one")
    second = service.campaign_execution("campaign-a", operation_id="start-two")
    assert first["audit"]["proofComplete"] is True
    assert [item["attemptId"] for item in first["audit"]["attempts"]] == ["campaign-a-job-a1"]
    assert [item["resultId"] for item in first["audit"]["results"]] == ["campaign-a-job-a1-result"]
    assert [item["attemptId"] for item in second["audit"]["attempts"]] == ["campaign-a-job-a2"]
    encoded = json.dumps(first, sort_keys=True)
    for forbidden in ("rawWeights", "C:/secret", '"metrics"', '"config"', "must-not-leak"):
        assert forbidden not in encoded
    assert len(first["audit"]["workerSessions"]) == 2
    assert len(first["audit"]["rounds"][0]["contributions"]) == 2


def test_two_campaigns_with_distinct_owners_never_cross_audit_boundaries(tmp_path: Path) -> None:
    service = _service(tmp_path, "campaign-a")
    owner_b = {"projectId": "project-b", "deploymentId": "deployment-b"}
    service.store.bind_campaign_owner(
        "campaign-b", project_id=owner_b["projectId"], deployment_id=owner_b["deploymentId"]
    )
    service.store.put_entity("campaign", "campaign-b", {
        "metadata": {"id": "campaign-b"}, "state": "succeeded"
    })
    _seed_start(
        service, campaign="campaign-a", operation="start-a",
        attempt="campaign-a-job-a1", result="campaign-a-job-a1-result",
    )
    _seed_start(
        service, campaign="campaign-b", operation="start-b",
        attempt="campaign-b-job-a1", result="campaign-b-job-a1-result",
        owner=owner_b,
    )
    first = service.campaign_execution(
        "campaign-a", operation_id="start-a",
        expected_project_id="project-a", expected_deployment_id="deployment-a",
    )
    second = service.campaign_execution(
        "campaign-b", operation_id="start-b",
        expected_project_id="project-b", expected_deployment_id="deployment-b",
    )
    assert first["audit"]["identity"]["projectId"] == "project-a"
    assert second["audit"]["identity"]["projectId"] == "project-b"
    assert {item["attemptId"] for item in first["audit"]["attempts"]} == {
        "campaign-a-job-a1"
    }
    assert {item["attemptId"] for item in second["audit"]["attempts"]} == {
        "campaign-b-job-a1"
    }


def test_collection_cap_is_explicitly_incomplete(tmp_path: Path) -> None:
    service = _service(tmp_path, "campaign-cap")
    _seed_start(
        service, campaign="campaign-cap", operation="start-cap",
        attempt="campaign-cap-job-a1", result="campaign-cap-job-a1-result",
    )
    service.store.EXECUTION_ENTITY_LIMIT = 1
    audit = service.campaign_execution("campaign-cap", operation_id="start-cap")["audit"]
    assert audit["proofComplete"] is False
    assert audit["auditState"] == "incomplete"
    assert "evidence-truncated:worker-session" in audit["reasons"]
    assert audit["counts"]["workerSessions"] == 1


def test_operation_history_cap_keeps_exact_requested_operation(tmp_path: Path) -> None:
    service = _service(tmp_path, "campaign-operations")
    service.store.put_operation({
        "operationId": "start-old", "action": "campaign.start",
        "targetId": "campaign-operations", "state": "failed",
        "createdAt": "2026-09-25T00:00:00Z",
    })
    _seed_start(
        service, campaign="campaign-operations", operation="start-requested",
        attempt="campaign-operations-job-a1",
        result="campaign-operations-job-a1-result",
    )
    service.store.EXECUTION_OPERATION_LIMIT = 1
    execution = service.campaign_execution(
        "campaign-operations", operation_id="start-requested"
    )
    assert execution["operation"]["operationId"] == "start-requested"
    assert execution["operation"]["state"] == "succeeded"
    assert execution["audit"]["proofComplete"] is False
    assert "evidence-truncated:operations" in execution["audit"]["reasons"]


def test_cross_attempt_result_lineage_forces_incomplete_audit(tmp_path: Path) -> None:
    service = _service(tmp_path, "campaign-cross-lineage")
    _seed_start(
        service, campaign="campaign-cross-lineage", operation="start-cross",
        attempt="campaign-cross-lineage-job-a1",
        result="campaign-cross-lineage-job-a1-result",
    )
    service.store.put_entity("job", "job-b", {
        "jobId": "job-b", "campaignId": "campaign-cross-lineage",
        "facetId": "f", "candidateId": "c", "state": "succeeded",
        "workerIds": [], "attempt": 1,
    })
    result = service.store.get_entity(
        "result", "campaign-cross-lineage-job-a1-result"
    )
    result["jobId"] = "job-b"
    result["lineage"] = {
        **result["lineage"], "jobId": "job-b"
    }
    with service.store._lock, service.store.connection:
        service.store.connection.execute(
            "UPDATE entities SET payload=? WHERE kind='result' AND entity_id=?",
            (json.dumps(result), result["resultId"]),
        )
    audit = service.campaign_execution(
        "campaign-cross-lineage", operation_id="start-cross"
    )["audit"]
    assert audit["proofComplete"] is False
    assert "result-attempt-lineage-mismatch" in audit["reasons"]


def test_lost_cleanup_ack_is_unknown_without_changing_natural_success(tmp_path: Path) -> None:
    service = _service(tmp_path, "campaign-cleanup")
    _seed_start(
        service, campaign="campaign-cleanup", operation="start-cleanup",
        attempt="campaign-cleanup-job-a1", result="campaign-cleanup-job-a1-result",
    )
    session = service.store.get_entity(
        "worker-session", "campaign-cleanup-job-a1-w1"
    )
    session["cleanupState"] = "outcome-unknown"
    session.pop("cleanupAck", None)
    service.store.put_entity(
        "worker-session", session["sessionId"], session,
        expected_generation=session["generation"],
    )
    audit = service.campaign_execution(
        "campaign-cleanup", operation_id="start-cleanup"
    )["audit"]
    assert audit["outcome"] == {
        "campaignState": "succeeded", "startOperationState": "succeeded"
    }
    assert audit["auditState"] == "unknown"
    assert audit["proofComplete"] is False
    assert "worker-session-cleanup-outcome-unknown" in audit["reasons"]


def test_succeeded_resumed_attempt_with_zero_new_rounds_stays_incomplete(tmp_path: Path) -> None:
    service = _service(tmp_path, "campaign-resumed")
    _seed_start(
        service, campaign="campaign-resumed", operation="start-resumed",
        attempt="campaign-resumed-job-a2", result="campaign-resumed-job-a2-result",
    )
    round_row = service.store.get_entity("round", "campaign-resumed-job-a2-r0")
    service.store.archive_entity(
        "round", "campaign-resumed-job-a2-r0",
        expected_generation=round_row["generation"],
    )
    audit = service.campaign_execution(
        "campaign-resumed", operation_id="start-resumed"
    )["audit"]
    assert audit["outcome"]["campaignState"] == "succeeded"
    assert [item["resultId"] for item in audit["results"]] == [
        "campaign-resumed-job-a2-result"
    ]
    assert audit["proofComplete"] is False
    assert "committed-round-unavailable" in audit["reasons"]


def test_publication_ack_before_result_is_explicitly_incomplete_after_reopen(tmp_path: Path) -> None:
    service = _service(tmp_path, "campaign-gap")
    operation = "start-gap"
    attempt = "campaign-gap-job-a1"
    job = "campaign-gap-job"
    lineage = _lineage("campaign-gap", operation, job, attempt)
    service.store.put_operation({
        "operationId": operation, "action": "campaign.start", "targetId": "campaign-gap",
        "state": "running",
    })
    service.store.put_entity("job", job, {
        "jobId": job, "campaignId": "campaign-gap", "facetId": "f", "candidateId": "c",
        "state": "running", "workerIds": ["w1"], "attempt": 1,
    })
    service.store.put_entity("attempt", attempt, {
        "attemptId": attempt, "jobId": job, "number": 1, "cohort": ["w1"],
        "state": "running", "lineage": lineage,
    })
    service.store.put_entity("publication", f"{attempt}-publication", {
        "publicationId": f"{attempt}-publication", "resultId": f"{attempt}-result",
        "state": "acknowledged", "lineage": lineage,
        "artifactSha256": "a" * 64, "artifactDigest": "sha256:" + "b" * 64,
    })
    before = service.campaign_execution("campaign-gap", operation_id=operation)
    assert before["audit"]["proofComplete"] is False
    assert "artifact-publication-result-pending" in before["audit"]["reasons"]
    service.close()

    reopened = TrainerService(Settings.for_test(tmp_path / "trainer"))
    after = reopened.campaign_execution("campaign-gap", operation_id=operation)
    assert after["audit"] == before["audit"]
    reopened.close()


def test_multiple_attempts_for_one_start_preserve_retry_lineage(tmp_path: Path) -> None:
    service = _service(tmp_path, "campaign-retry")
    operation = "start-retry"
    job = "campaign-retry-job"
    failed_attempt = "campaign-retry-job-a1"
    failed_lineage = _lineage("campaign-retry", operation, job, failed_attempt)
    service.store.put_entity("job", job, {
        "jobId": job, "campaignId": "campaign-retry", "facetId": "f", "candidateId": "c",
        "state": "succeeded", "workerIds": ["w1", "w2"], "attempt": 2,
        "resultId": "campaign-retry-job-a2-result",
    })
    service.store.put_entity("attempt", failed_attempt, {
        "attemptId": failed_attempt, "jobId": job, "number": 1,
        "cohort": ["w1", "w2"], "state": "failed", "lineage": failed_lineage,
    })
    for worker in ("w1", "w2"):
        session = f"{failed_attempt}-{worker}"
        service.store.put_entity("worker-session", session, {
            "id": session, "campaignId": "campaign-retry", "jobId": job,
            "attemptId": failed_attempt, "workerId": worker, "sessionId": session,
            "createState": "acknowledged", "cleanupState": "acknowledged",
            "createAck": {"id": session},
            "cleanupAck": {"id": session, "deleted": True, "artifactsRetained": True},
            "lineage": failed_lineage,
        })
    _seed_start(
        service, campaign="campaign-retry", operation=operation,
        attempt="campaign-retry-job-a2", result="campaign-retry-job-a2-result",
    )
    audit = service.campaign_execution("campaign-retry", operation_id=operation)["audit"]
    assert audit["proofComplete"] is True
    assert [(item["attemptId"], item["state"]) for item in audit["attempts"]] == [
        (failed_attempt, "failed"), ("campaign-retry-job-a2", "succeeded")
    ]
    assert audit["counts"]["workerSessions"] == 4


def test_lineage_cannot_be_rebound_and_historical_missing_lineage_is_unavailable(tmp_path: Path) -> None:
    service = _service(tmp_path, "campaign-lineage")
    service.store.put_operation({
        "operationId": "start-a", "action": "campaign.start",
        "targetId": "campaign-lineage", "state": "failed",
    })
    service.store.put_entity("job", "job-a", {
        "jobId": "job-a", "campaignId": "campaign-lineage", "facetId": "f",
        "candidateId": "c", "state": "failed", "workerIds": [], "attempt": 1,
    })
    historical = service.store.put_entity("attempt", "attempt-old", {
        "attemptId": "attempt-old", "jobId": "job-a", "number": 1,
        "cohort": ["w1"], "state": "failed",
    })
    with pytest.raises(ConflictError, match="no durable execution lineage"):
        service.store.put_entity(
            "attempt", "attempt-old",
            {**historical, "lineage": _lineage("campaign-lineage", "start-a", "job-a", "attempt-old")},
            expected_generation=historical["generation"],
        )
    audit = service.campaign_execution("campaign-lineage", operation_id="start-a")["audit"]
    assert audit["proofComplete"] is False
    assert audit["lineageUnavailable"] == [{"kind": "attempt", "id": "attempt-old"}]


def test_actual_parameter_server_path_captures_lineage_acks_and_retry_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "trainer"
    model = workspace / "base"
    data = workspace / "train.jsonl"
    tiny_model(model)
    training_dataset(data)
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    artifact_manifest = workspace / "rm-artifacts.json"
    artifact_manifest.write_text(json.dumps({"entries": [{
        "digest": f"sha256:{digest}", "kind": "file", "checkoutPath": str(data),
    }]}), encoding="utf-8")
    monkeypatch.setenv("JESTER_RM_ARTIFACT_MANIFEST", str(artifact_manifest))

    class RMStub:
        available = True
        topology_id = "test-training-topology"

        def register_trainer_publication(
            self,
            payload: dict,
            *,
            idempotency_key: str,
            timeout_seconds: float = 3600,
        ) -> dict:
            del idempotency_key, timeout_seconds
            suffix = "1" if "attempt-0001" in payload["relativePath"] else "2"
            source = service._publication_source_expectation().api_dict(
                exclude_none=False
            )
            return {
                "schemaVersion": "jester-rm-trainer-publication-authority-v1",
                "authorityId": f"authority-{suffix}",
                "identity": {
                    "projectId": OWNER["projectId"],
                    "deploymentId": OWNER["deploymentId"],
                    "deploymentGeneration": 1,
                    "topologyId": self.topology_id,
                    "topologyGeneration": 1,
                },
                "source": source,
                "result": {
                    "resultId": payload["resultId"],
                    "lineage": payload["lineage"],
                },
                "publication": {
                    "publicationId": payload["publicationId"],
                    "state": "acknowledged",
                    "artifactSha256": payload["artifactSha256"],
                    "artifactDigest": "sha256:" + suffix * 64,
                    "descriptorKind": "directory",
                    "descriptorSize": None,
                },
            }

    with worker_topology(tmp_path / "workers", count=2) as topology:
        settings = Settings.for_test(workspace, topology)
        settings.lease.update({
            "instanceGeneration": 1,
            "hostEpoch": "test-host-epoch",
        })
        settings.lease["paths"]["root"] = str(workspace)
        service = TrainerService(settings)
        try:
            service.register_managed_dataset(
                DatasetManifest.model_validate({
                    "metadata": {"id": "tiny-dataset"},
                    "files": [{
                        "split": "train", "digest": f"sha256:{digest}",
                        "artifactDigest": f"sha256:{digest}", "logicalPath": "train.jsonl",
                    }],
                }),
                project_id=OWNER["projectId"], deployment_id=OWNER["deploymentId"],
            )
            managed = campaign_fixture(
                workspace, model, campaign_id="campaign-production-path",
                version="v1", workers=["worker-1", "worker-2"],
            )
            service.put_managed_campaign(
                managed, project_id=OWNER["projectId"], deployment_id=OWNER["deploymentId"]
            )
            service.plan_campaign(managed.metadata.id)
            service.rm = RMStub()  # type: ignore[assignment]
            campaign_model = service.get_campaign(managed.metadata.id)
            job = TrainingJob.model_validate(service.campaign_jobs(managed.metadata.id)[0])
            workers = service.pool.select(["worker-1", "worker-2"], 2)
            operation_id = "start-production"
            service.store.reserve_campaign_start(
                managed.metadata.id,
                {
                    "operationId": operation_id,
                    "action": "campaign.start",
                    "targetId": managed.metadata.id,
                    "state": "accepted",
                },
                controller_id=service._controller_lease.instance_id,
            )
            service.store.mark_campaign_start_launch_committed(
                managed.metadata.id,
                operation_id,
                controller_id=service._controller_lease.instance_id,
            )
            assert service.store.mark_campaign_start_running(
                managed.metadata.id,
                operation_id,
                controller_id=service._controller_lease.instance_id,
            )
            observed_before_effect: list[str] = []
            for worker in workers:
                original = worker.json

                def checked(method, path, payload=None, *, _original=original):
                    if method == "POST" and path == "/v1/sessions":
                        session_id = str(payload["sessionId"])
                        attempt_id = session_id.rsplit("-worker-", 1)[0]
                        stored_attempt = service.store.get_entity("attempt", attempt_id)
                        observation = service.store.get_entity("worker-session", session_id)
                        assert stored_attempt["lineage"]["startOperationId"] == operation_id
                        assert observation["createState"] == "dispatched"
                        assert observation["lineage"] == stored_attempt["lineage"]
                        observed_before_effect.append(session_id)
                    return _original(method, path, payload)

                worker.json = checked  # type: ignore[method-assign]

            first = service._run_job(
                campaign_model, job, workers, attempt_number=1,
                start_operation_id=operation_id,
            )
            first_checkpoints = (
                workspace / "results" / managed.metadata.id / job.job_id
                / "attempt-0001" / "checkpoints"
            )
            for checkpoint in first_checkpoints.glob("round-*/*"):
                if checkpoint.is_file():
                    checkpoint.unlink()
            second = service._run_job(
                campaign_model, job, workers, attempt_number=2,
                start_operation_id=operation_id,
            )
            assert first.result_id == f"{job.job_id}-a1-result"
            assert second.result_id == f"{job.job_id}-a2-result"
            assert {item["resultId"] for item in service.store.list_entities("result")} == {
                first.result_id, second.result_id,
            }
            assert len(observed_before_effect) == 4

            current_campaign = service.get_campaign(managed.metadata.id)
            service.store.put_entity(
                "campaign", managed.metadata.id,
                current_campaign.model_copy(update={"state": "succeeded"}).api_dict(),
                expected_generation=current_campaign.generation,
            )
            service.store.put_operation({
                "operationId": operation_id, "action": "campaign.start",
                "targetId": managed.metadata.id, "state": "succeeded",
            })
            audit = service.campaign_execution(
                managed.metadata.id, operation_id=operation_id
            )["audit"]
            assert audit["reasons"] == [], {
                "attempts": audit["attempts"], "rounds": audit["rounds"]
            }
            assert audit["proofComplete"] is True
            assert audit["counts"] == {
                "attempts": 2, "rounds": 2, "results": 2,
                "publications": 2, "workerSessions": 4,
            }
            assert all(len(item["contributions"]) == 2 for item in audit["rounds"])
            assert all(item["aggregateSha256"] for item in audit["rounds"])
            assert {
                item["artifactDigest"] for item in audit["publications"]
            } == {"sha256:" + "1" * 64, "sha256:" + "2" * 64}
            assert all(
                session["cleanupAck"] == {
                    "id": session["sessionId"], "deleted": True,
                    "artifactsRetained": True,
                }
                for session in audit["workerSessions"]
            )
        finally:
            service.close()
