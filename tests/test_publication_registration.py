from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from jester_trainer.config import Settings
from jester_trainer.db import StateStore
from jester_trainer.errors import ConflictError, TrainerError
from jester_trainer.models import (
    ArtifactPublication,
    Campaign,
    ExecutionLineage,
    Operation,
    PublicationAuthorityArtifact,
    PublicationRegistration,
    PublicationRegistrationRequest,
    TrainingAttempt,
    TrainingJob,
    TrainingResult,
)
from jester_trainer.service import TrainerService
from jester_trainer.util import sha256_json


PROJECT = "project-a"
DEPLOYMENT = "deployment-a"
CAMPAIGN = "campaign-a"
JOB = "job-a"
ATTEMPT = "job-a-a1"
OPERATION = "start-a"
RESULT = "job-a-a1-result"
PUBLICATION = "job-a-a1-publication"


def _settings(root: Path) -> Settings:
    topology = {
        "topologyId": "topology-a",
        "topologyType": "training",
        "role": "trainer",
        "nodeId": "trainer",
        "channels": [],
        "nodes": [],
        "rmController": {"url": "http://127.0.0.1:1"},
        "serviceGrant": {"credential": "opaque"},
    }
    settings = Settings.for_test(root, topology=topology)
    settings.lease.update({
        "instanceId": "trainer-a",
        "instanceGeneration": 7,
        "leaseId": "lease-a",
        "hostId": "host-a",
        "hostEpoch": "epoch-a",
    })
    settings.lease["paths"]["root"] = str(root)
    return settings


def _lineage() -> ExecutionLineage:
    return ExecutionLineage(
        projectId=PROJECT,
        deploymentId=DEPLOYMENT,
        campaignId=CAMPAIGN,
        startOperationId=OPERATION,
        jobId=JOB,
        attemptId=ATTEMPT,
    )


def _campaign(service: TrainerService) -> Campaign:
    return Campaign.model_validate({
        "metadata": {"id": CAMPAIGN},
        "spec": {
            "topologyId": "topology-a",
            "datasets": [{"datasetId": "dataset-a", "split": "train"}],
            "facets": [{"id": "facet", "miConfig": {}}],
            "training": {"maxSteps": 1},
            "checkpoint": {
                "outputDirectory": str(service.settings.home / "results")
            },
        },
    })


def _seed_campaign_start(service: TrainerService) -> Campaign:
    service.store.bind_campaign_owner(
        CAMPAIGN, project_id=PROJECT, deployment_id=DEPLOYMENT
    )
    campaign = _campaign(service)
    service.store.put_entity("campaign", CAMPAIGN, campaign.api_dict())
    service.store.reserve_campaign_start(
        CAMPAIGN,
        Operation(
            operationId=OPERATION,
            action="campaign.start",
            targetId=CAMPAIGN,
        ).api_dict(),
        controller_id=service._controller_lease.instance_id,
    )
    service.store.mark_campaign_start_launch_committed(
        CAMPAIGN, OPERATION, controller_id=service._controller_lease.instance_id
    )
    assert service.store.mark_campaign_start_running(
        CAMPAIGN, OPERATION, controller_id=service._controller_lease.instance_id
    )
    return service.get_campaign(CAMPAIGN)


def _seed_execution(service: TrainerService) -> None:
    _seed_campaign_start(service)
    service.store.put_entity("job", JOB, TrainingJob(
        jobId=JOB, campaignId=CAMPAIGN, facetId="facet", candidateId="candidate",
        state="running", attempt=1,
    ).api_dict())
    service.store.put_entity("attempt", ATTEMPT, TrainingAttempt(
        attemptId=ATTEMPT, jobId=JOB, number=1, cohort=["worker-a"],
        lineage=_lineage(), state="running",
    ).api_dict())


def _draft(service: TrainerService) -> PublicationRegistration:
    artifact = service.settings.home / "artifacts" / "result-a"
    artifact.mkdir(parents=True, exist_ok=True)
    weights = artifact / "facet.safetensors"
    weights.write_bytes(b"tiny-weights")
    raw_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    lineage = _lineage()
    request = PublicationRegistrationRequest(
        relativePath="artifacts/result-a",
        publicationId=PUBLICATION,
        resultId=RESULT,
        lineage=lineage,
        artifactSha256=raw_sha,
    )
    registration = PublicationRegistration(
        registrationId=PUBLICATION,
        publicationId=PUBLICATION,
        resultId=RESULT,
        lineage=lineage,
        relativePath=request.relative_path,
        artifactSha256=raw_sha,
        idempotencyKey="k" * 48,
        requestSha256=sha256_json({
            "topologyId": service.rm.topology_id,
            **request.api_dict(exclude_none=False),
        }),
        request=request,
        sourceExpectation=service._publication_source_expectation(),
        startControllerId=service._controller_lease.instance_id,
    )
    result = TrainingResult(
        resultId=RESULT, campaignId=CAMPAIGN, jobId=JOB, lineage=lineage,
        facetReference="facet@v1", artifactPath=str(artifact),
        artifactSha256=raw_sha, metrics={"loss": 1.0},
    )
    publication = ArtifactPublication(
        publicationId=PUBLICATION, resultId=RESULT, lineage=lineage,
        artifactSha256=raw_sha,
    )
    _result, _publication, stored = service.store.reserve_publication_draft(
        result=result.api_dict(), publication=publication.api_dict(),
        registration=registration.api_dict(exclude_none=False),
    )
    return PublicationRegistration.model_validate(stored)


def _authority(registration: PublicationRegistration, **source_changes: object) -> dict:
    source = {
        "trainerInstanceId": "trainer-a",
        "trainerInstanceGeneration": 7,
        "trainerLeaseId": "lease-a",
        "hostId": "host-a",
        "hostEpoch": "epoch-a",
        **source_changes,
    }
    return {
        "schemaVersion": "jester-rm-trainer-publication-authority-v1",
        "authorityId": "authority-a",
        "identity": {
            "projectId": PROJECT,
            "deploymentId": DEPLOYMENT,
            "deploymentGeneration": 3,
            "topologyId": "topology-a",
            "topologyGeneration": 5,
        },
        "source": source,
        "result": {
            "resultId": RESULT,
            "lineage": registration.lineage.api_dict(exclude_none=False),
        },
        "publication": {
            "publicationId": PUBLICATION,
            "state": "acknowledged",
            "artifactSha256": registration.artifact_sha256,
            "artifactDigest": "sha256:" + "d" * 64,
            "descriptorKind": "directory",
            "descriptorSize": None,
        },
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifactSha256", "x" + "a" * 64 + "x"),
        ("artifactSha256", "a" * 64 + "\n"),
        ("artifactDigest", "xsha256:" + "b" * 64 + "x"),
        ("artifactDigest", "sha256:" + "b" * 64 + "\n"),
    ],
)
def test_publication_authority_artifact_requires_exact_canonical_hashes(
    field: str, value: str,
) -> None:
    payload = {
        "publicationId": PUBLICATION,
        "state": "acknowledged",
        "artifactSha256": "a" * 64,
        "artifactDigest": "sha256:" + "b" * 64,
        "descriptorKind": "directory",
        "descriptorSize": None,
    }
    payload[field] = value
    with pytest.raises(ValueError):
        PublicationAuthorityArtifact.model_validate(payload)


def test_publication_request_rejects_nested_lineage_extra_field() -> None:
    lineage = _lineage().api_dict(exclude_none=False)
    with pytest.raises(ValueError):
        PublicationRegistrationRequest.model_validate(
            {
                "relativePath": "artifacts/result-a",
                "publicationId": PUBLICATION,
                "resultId": RESULT,
                "lineage": {**lineage, "unexpected": "forbidden"},
                "artifactSha256": "a" * 64,
            }
        )
class RMReply:
    available = True
    topology_id = "topology-a"

    def __init__(self, response: dict, *, fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.calls: list[tuple[dict, str]] = []
        self.timeouts: list[float] = []

    def register_trainer_publication(
        self,
        payload: dict,
        *,
        idempotency_key: str,
        timeout_seconds: float = 3600,
    ) -> dict:
        self.calls.append((payload, idempotency_key))
        self.timeouts.append(timeout_seconds)
        if self.fail:
            raise TrainerError("response lost", code="rm_unavailable", status=503)
        return self.response


def test_intent_and_complete_result_are_durable_before_first_rm_call(tmp_path: Path) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        _seed_execution(service)
        registration = _draft(service)
        response = _authority(registration)

        class InspectingRM(RMReply):
            def register_trainer_publication(
                self,
                payload: dict,
                *,
                idempotency_key: str,
                timeout_seconds: float = 3600,
            ) -> dict:
                reopened = StateStore(service.settings.database)
                try:
                    durable = reopened.get_entity(
                        "publication-registration", PUBLICATION
                    )
                    assert durable["state"] == "pending"
                    assert durable["request"] == payload
                    assert durable["idempotencyKey"] == idempotency_key
                    assert reopened.get_entity("result", RESULT)["metrics"] == {"loss": 1.0}
                    assert reopened.get_entity("publication", PUBLICATION)["state"] == "dispatched"
                finally:
                    reopened.close()
                return super().register_trainer_publication(
                    payload,
                    idempotency_key=idempotency_key,
                    timeout_seconds=timeout_seconds,
                )

        service.rm = InspectingRM(response)  # type: ignore[assignment]
        acknowledged = service._register_reserved_publication(registration)
        assert acknowledged.state == "acknowledged"
        assert service.rm.timeouts == [3600]  # type: ignore[attr-defined]
        assert service.store.get_entity("result", RESULT)["artifactDigest"] == response["publication"]["artifactDigest"]
    finally:
        service.close()


@pytest.mark.parametrize(
    "source_change",
    [
        {"trainerInstanceId": "trainer-b"},
        {"trainerInstanceGeneration": 8},
        {"trainerLeaseId": "lease-b"},
        {"hostId": "host-b"},
        {"hostEpoch": "epoch-b"},
    ],
)
def test_source_authority_mismatch_remains_uncertain(
    tmp_path: Path, source_change: dict
) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        _seed_execution(service)
        registration = _draft(service)
        service.rm = RMReply(_authority(registration, **source_change))  # type: ignore[assignment]
        with pytest.raises(TrainerError) as error:
            service._register_reserved_publication(registration)
        assert error.value.code == "artifact_publication_outcome_unknown"
        stored = service.store.get_entity("publication-registration", PUBLICATION)
        assert stored["state"] == "reconciliation-required"
        assert service.store.get_entity("publication", PUBLICATION)["state"] == "dispatched"
    finally:
        service.close()


def test_lost_ack_restart_replays_same_request_and_key_without_training(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = TrainerService(settings)
    _seed_execution(service)
    registration = _draft(service)
    response = _authority(registration)
    lost = RMReply(response, fail=True)
    service.rm = lost  # type: ignore[assignment]
    with pytest.raises(TrainerError):
        service._register_reserved_publication(registration)
    service.close()

    replay = RMReply(response)
    with patch("jester_trainer.service.RMScopedClient", return_value=replay):
        reopened = TrainerService(settings)
    try:
        assert replay.calls == [(registration.request.api_dict(exclude_none=False), registration.idempotency_key)]
        assert replay.timeouts == [5]
        assert reopened.store.get_entity("publication-registration", PUBLICATION)["state"] == "acknowledged"
        assert reopened.store.get_entity("attempt", ATTEMPT)["state"] == "succeeded"
        assert reopened.store.get_entity("job", JOB)["resultId"] == RESULT
        assert reopened.store.get_entity("campaign", CAMPAIGN)["state"] == "succeeded"
        assert len(reopened.store.list_entities("attempt")) == 1
    finally:
        reopened.close()


def test_restart_after_local_ack_settles_without_rm_call(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = TrainerService(settings)
    _seed_execution(service)
    registration = _draft(service)
    response = _authority(registration)
    service.store.acknowledge_publication_authority(
        registration_id=PUBLICATION,
        expected_generation=registration.generation,
        authority=response,
    )
    service.close()

    no_call = RMReply(response, fail=True)
    with patch("jester_trainer.service.RMScopedClient", return_value=no_call):
        reopened = TrainerService(settings)
    try:
        assert no_call.calls == []
        assert reopened.store.get_entity("attempt", ATTEMPT)["state"] == "succeeded"
        assert reopened.store.get_entity("campaign", CAMPAIGN)["state"] == "succeeded"
    finally:
        reopened.close()


def test_second_reopen_reconciles_valid_ack_after_first_unknown_recovery(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    service = TrainerService(settings)
    _seed_execution(service)
    registration = _draft(service)
    response = _authority(registration)
    service.rm = RMReply(response, fail=True)  # type: ignore[assignment]
    with pytest.raises(TrainerError):
        service._register_reserved_publication(registration)
    service.close()

    first_unknown = RMReply(response, fail=True)
    with patch("jester_trainer.service.RMScopedClient", return_value=first_unknown):
        first_reopen = TrainerService(settings)
    try:
        assert first_unknown.timeouts == [5]
        start = first_reopen.store.connection.execute(
            "SELECT phase FROM campaign_start_reservations WHERE campaign_id=?",
            (CAMPAIGN,),
        ).fetchone()
        assert start["phase"] == "reconciliation-required"
        assert first_reopen.store.get_entity("attempt", ATTEMPT)["state"] == "running"
    finally:
        first_reopen.close()

    valid = RMReply(response)
    with patch("jester_trainer.service.RMScopedClient", return_value=valid):
        second_reopen = TrainerService(settings)
    try:
        assert valid.timeouts == [5]
        assert second_reopen.store.get_entity("attempt", ATTEMPT)["state"] == "succeeded"
        assert second_reopen.store.get_entity("job", JOB)["state"] == "succeeded"
        assert second_reopen.store.get_entity("campaign", CAMPAIGN)["state"] == "succeeded"
        assert len(second_reopen.store.list_entities("attempt")) == 1
        start = second_reopen.store.connection.execute(
            "SELECT phase FROM campaign_start_reservations WHERE campaign_id=?",
            (CAMPAIGN,),
        ).fetchone()
        assert start["phase"] == "terminal"
    finally:
        second_reopen.close()


def test_recovery_accepts_authority_values_for_unspecified_legacy_source_fields(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.lease.pop("instanceGeneration", None)
    settings.lease.pop("hostEpoch", None)
    service = TrainerService(settings)
    try:
        _seed_execution(service)
        registration = _draft(service)
        assert registration.source_expectation.trainer_instance_generation is None
        assert registration.source_expectation.host_epoch is None
        service.rm = RMReply(_authority(registration))  # type: ignore[assignment]
        acknowledged = service._register_reserved_publication(registration)
        assert service.store.settle_acknowledged_publication(PUBLICATION)
        service._settle_recovered_campaign(acknowledged)
        assert service.store.get_entity("campaign", CAMPAIGN)["state"] == "succeeded"
    finally:
        service.close()


def test_cancel_after_intent_preserves_ack_but_blocks_success(tmp_path: Path) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        _seed_execution(service)
        registration = _draft(service)
        response = _authority(registration)
        service.store.reserve_campaign_cancel(
            CAMPAIGN,
            OPERATION,
            Operation(
                operationId="cancel-a", action="campaign.cancel", targetId=CAMPAIGN
            ).api_dict(),
            controller_id=service._controller_lease.instance_id,
        )
        service.rm = RMReply(response)  # type: ignore[assignment]
        acknowledged = service._register_reserved_publication(registration)
        assert acknowledged.state == "acknowledged"
        assert service.store.settle_acknowledged_publication(PUBLICATION) is False
        assert service.store.get_entity("attempt", ATTEMPT)["state"] == "running"
        assert service.store.get_entity("publication", PUBLICATION)["state"] == "acknowledged"
    finally:
        service.close()


def test_atomic_reservation_rolls_back_all_three_rows(tmp_path: Path) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        _seed_execution(service)
        service.store.connection.execute(
            """CREATE TRIGGER fail_registration BEFORE INSERT ON entities
               WHEN NEW.kind='publication-registration'
               BEGIN SELECT RAISE(ABORT, 'injected'); END"""
        )
        with pytest.raises(Exception, match="injected"):
            _draft(service)
        with pytest.raises(Exception):
            service.store.get_entity("result", RESULT)
        with pytest.raises(Exception):
            service.store.get_entity("publication", PUBLICATION)
        with pytest.raises(Exception):
            service.store.get_entity("publication-registration", PUBLICATION)
    finally:
        service.close()


def test_atomic_ack_rolls_back_result_publication_and_registration(
    tmp_path: Path,
) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        _seed_execution(service)
        registration = _draft(service)
        service.store.connection.execute(
            """CREATE TRIGGER fail_publication_ack BEFORE UPDATE ON entities
               WHEN NEW.kind='publication'
               BEGIN SELECT RAISE(ABORT, 'ack-injected'); END"""
        )
        with pytest.raises(Exception, match="ack-injected"):
            service.store.acknowledge_publication_authority(
                registration_id=PUBLICATION,
                expected_generation=registration.generation,
                authority=_authority(registration),
            )
        assert service.store.get_entity("result", RESULT).get("artifactDigest") is None
        assert service.store.get_entity("publication", PUBLICATION)["state"] == "dispatched"
        stored = service.store.get_entity("publication-registration", PUBLICATION)
        assert stored["state"] == "pending"
        assert stored["authority"] is None
    finally:
        service.close()


def test_settlement_reports_missing_execution_row_without_partial_success(
    tmp_path: Path,
) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        _seed_execution(service)
        registration = _draft(service)
        service.store.acknowledge_publication_authority(
            registration_id=PUBLICATION,
            expected_generation=registration.generation,
            authority=_authority(registration),
        )
        service.store.connection.execute(
            "DELETE FROM entities WHERE kind='job' AND entity_id=?", (JOB,)
        )
        service.store.connection.commit()
        with pytest.raises(ConflictError) as error:
            service.store.settle_acknowledged_publication(PUBLICATION)
        assert error.value.code == "publication_draft_conflict"
        assert service.store.get_entity("attempt", ATTEMPT)["state"] == "running"
    finally:
        service.close()


def test_settlement_refuses_succeeded_job_with_different_result(
    tmp_path: Path,
) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        _seed_execution(service)
        registration = _draft(service)
        service.store.acknowledge_publication_authority(
            registration_id=PUBLICATION,
            expected_generation=registration.generation,
            authority=_authority(registration),
        )
        job = service.store.get_entity("job", JOB)
        service.store.put_entity(
            "job",
            JOB,
            {**job, "state": "succeeded", "resultId": "other-result"},
            expected_generation=job["generation"],
        )
        with pytest.raises(ConflictError) as error:
            service.store.settle_acknowledged_publication(PUBLICATION)
        assert error.value.code == "publication_execution_mismatch"
        assert service.store.get_entity("job", JOB)["resultId"] == "other-result"
    finally:
        service.close()


@pytest.mark.parametrize("stale_fence", ["owner", "start-controller"])
def test_settlement_refuses_stale_owner_or_start_fence(
    tmp_path: Path, stale_fence: str,
) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        _seed_execution(service)
        registration = _draft(service)
        service.store.acknowledge_publication_authority(
            registration_id=PUBLICATION,
            expected_generation=registration.generation,
            authority=_authority(registration),
        )
        if stale_fence == "owner":
            service.store.connection.execute(
                "UPDATE campaign_owners SET deployment_id='deployment-stale' "
                "WHERE campaign_id=?",
                (CAMPAIGN,),
            )
            expected_code = "publication_owner_mismatch"
        else:
            service.store.connection.execute(
                "UPDATE campaign_start_reservations SET controller_id='controller-stale' "
                "WHERE campaign_id=?",
                (CAMPAIGN,),
            )
            expected_code = "publication_start_fence_missing"
        service.store.connection.commit()
        with pytest.raises(ConflictError) as error:
            service.store.settle_acknowledged_publication(PUBLICATION)
        assert error.value.code == expected_code
        assert service.store.get_entity("attempt", ATTEMPT)["state"] == "running"
        assert service.store.get_entity("job", JOB)["state"] == "running"
    finally:
        service.close()


def test_recovery_refuses_changed_current_source_without_rm_or_settlement(
    tmp_path: Path,
) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        _seed_execution(service)
        registration = _draft(service)
        service.store.acknowledge_publication_authority(
            registration_id=PUBLICATION,
            expected_generation=registration.generation,
            authority=_authority(registration),
        )
        no_call = RMReply(_authority(registration), fail=True)
        service.rm = no_call  # type: ignore[assignment]
        service.settings.lease["leaseId"] = "lease-stale"
        service._recover_publications()
        assert no_call.calls == []
        assert service.store.get_entity("attempt", ATTEMPT)["state"] == "running"
        assert service.store.get_entity("job", JOB)["state"] == "running"
    finally:
        service.close()


def test_acknowledged_registration_without_authority_never_settles(
    tmp_path: Path,
) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        _seed_execution(service)
        _draft(service)
        stored = service.store.get_entity(
            "publication-registration", PUBLICATION
        )
        service.store.put_entity(
            "publication-registration",
            PUBLICATION,
            {**stored, "state": "acknowledged", "authority": None},
            expected_generation=stored["generation"],
        )
        service._recover_publications()
        assert service.store.get_entity("attempt", ATTEMPT)["state"] == "running"
        assert service.store.get_entity("job", JOB)["state"] == "running"
    finally:
        service.close()


def test_settled_receipts_do_not_starve_later_recoverable_intent(
    tmp_path: Path,
) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        _seed_execution(service)
        registration = _draft(service)
        authority = _authority(registration)
        _result, _publication, acknowledged_payload = (
            service.store.acknowledge_publication_authority(
                registration_id=PUBLICATION,
                expected_generation=registration.generation,
                authority=authority,
            )
        )
        acknowledged = PublicationRegistration.model_validate(acknowledged_payload)
        assert service.store.settle_acknowledged_publication(PUBLICATION)
        service._settle_recovered_campaign(acknowledged)
        settled_payload = acknowledged.api_dict(exclude_none=False)
        for index in range(256):
            clone_id = f"a-settled-{index:03d}"
            service.store.put_entity(
                "publication-registration",
                clone_id,
                {**settled_payload, "registrationId": clone_id},
            )
        pending_id = "z-pending"
        service.store.put_entity(
            "publication-registration",
            pending_id,
            {
                **settled_payload,
                "registrationId": pending_id,
                "state": "reconciliation-required",
                "authority": None,
            },
        )

        recoverable = service.store.recoverable_publication_registrations(limit=256)
        assert [item["registrationId"] for item in recoverable] == [pending_id]
    finally:
        service.close()


def test_historical_failed_attempt_does_not_block_reconciled_success(
    tmp_path: Path,
) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        _seed_execution(service)
        registration = _draft(service)
        _result, _publication, acknowledged_payload = (
            service.store.acknowledge_publication_authority(
                registration_id=PUBLICATION,
                expected_generation=registration.generation,
                authority=_authority(registration),
            )
        )
        failed_lineage = _lineage().model_copy(
            update={"attempt_id": "job-a-a0"}
        )
        service.store.put_entity(
            "attempt",
            "job-a-a0",
            TrainingAttempt(
                attemptId="job-a-a0",
                jobId=JOB,
                number=1,
                cohort=["worker-old"],
                lineage=failed_lineage,
                state="failed",
            ).api_dict(),
        )
        assert service.store.settle_acknowledged_publication(PUBLICATION)
        service.store.mark_campaign_start_reconciliation_required(
            CAMPAIGN,
            OPERATION,
            controller_id=service._controller_lease.instance_id,
            reason="simulated restart before campaign settlement",
        )
        service._settle_recovered_campaign(
            PublicationRegistration.model_validate(acknowledged_payload)
        )
        assert service.store.get_entity("campaign", CAMPAIGN)["state"] == "succeeded"
        assert service.store.get_entity("attempt", "job-a-a0")["state"] == "failed"
    finally:
        service.close()


def test_failed_recovery_rotates_behind_next_durable_intent(tmp_path: Path) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        _seed_execution(service)
        registration = _draft(service)
        payload = registration.api_dict(exclude_none=False)
        service.store.put_entity(
            "publication-registration",
            "a-recovery",
            {**payload, "registrationId": "a-recovery"},
        )
        service.store.put_entity(
            "publication-registration",
            "b-recovery",
            {**payload, "registrationId": "b-recovery"},
        )
        first = service.store.recoverable_publication_registrations(limit=1)
        assert first[0]["registrationId"] == PUBLICATION
        service.store.mark_publication_reconciliation_required(
            PUBLICATION, error={"code": "rm_unavailable", "detail": "unknown"}
        )
        second = service.store.recoverable_publication_registrations(limit=1)
        assert second[0]["registrationId"] == "a-recovery"
    finally:
        service.close()


def test_invalid_ack_page_rotates_so_pending_intent_progresses_next_pass(
    tmp_path: Path,
) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        _seed_execution(service)
        registration = _draft(service)
        payload = registration.api_dict(exclude_none=False)
        invalid_ack = {**payload, "state": "acknowledged", "authority": None}
        current = service.store.get_entity(
            "publication-registration", PUBLICATION
        )
        service.store.put_entity(
            "publication-registration",
            PUBLICATION,
            invalid_ack,
            expected_generation=current["generation"],
        )
        for index in range(255):
            clone_id = f"invalid-ack-{index:03d}"
            service.store.put_entity(
                "publication-registration",
                clone_id,
                {**invalid_ack, "registrationId": clone_id},
            )
        pending_id = "pending-after-invalid-page"
        service.store.put_entity(
            "publication-registration",
            pending_id,
            {**payload, "registrationId": pending_id},
        )
        rm = RMReply(_authority(registration), fail=True)
        service.rm = rm  # type: ignore[assignment]

        service._recover_publications()
        assert rm.calls == []
        service._recover_publications()
        assert len(rm.calls) == 1
        assert rm.timeouts == [5]
    finally:
        service.close()


def test_cancel_fenced_ack_page_rotates_and_next_reopen_reaches_pending(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    service = TrainerService(settings)
    try:
        _seed_execution(service)
        registration = _draft(service)
        _result, _publication, acknowledged_payload = (
            service.store.acknowledge_publication_authority(
                registration_id=PUBLICATION,
                expected_generation=registration.generation,
                authority=_authority(registration),
            )
        )
        service.store.reserve_campaign_cancel(
            CAMPAIGN,
            OPERATION,
            Operation(
                operationId="cancel-a", action="campaign.cancel", targetId=CAMPAIGN
            ).api_dict(),
            controller_id=service._controller_lease.instance_id,
        )
        for index in range(255):
            clone_id = f"cancel-fenced-{index:03d}"
            service.store.put_entity(
                "publication-registration",
                clone_id,
                {**acknowledged_payload, "registrationId": clone_id},
            )
        pending_id = "pending-after-cancel-page"
        pending = registration.api_dict(exclude_none=False)
        service.store.put_entity(
            "publication-registration",
            pending_id,
            {**pending, "registrationId": pending_id},
        )
    finally:
        service.close()

    rm = RMReply(_authority(registration), fail=True)
    with patch("jester_trainer.service.RMScopedClient", return_value=rm):
        first_reopen = TrainerService(settings)
        try:
            assert rm.calls == []
            assert first_reopen.store.get_entity("attempt", ATTEMPT)["state"] == "running"
        finally:
            first_reopen.close()

        second_reopen = TrainerService(settings)
        try:
            assert len(rm.calls) == 1
            assert rm.timeouts == [5]
            assert second_reopen.store.get_entity("attempt", ATTEMPT)["state"] == "running"
            assert second_reopen.store.get_entity(
                "publication-registration", PUBLICATION
            )["state"] == "acknowledged"
        finally:
            second_reopen.close()


def test_external_replay_budget_does_not_rotate_unattempted_pending_intent(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    service = TrainerService(settings)
    try:
        _seed_execution(service)
        first = _draft(service)
        second_id = "pending-b"
        second_key = "b" * 48
        service.store.put_entity(
            "publication-registration",
            second_id,
            {
                **first.api_dict(exclude_none=False),
                "registrationId": second_id,
                "idempotencyKey": second_key,
            },
        )
    finally:
        service.close()

    rm = RMReply(_authority(first), fail=True)
    with patch("jester_trainer.service.RMScopedClient", return_value=rm):
        first_reopen = TrainerService(settings)
        first_reopen.close()
        assert [key for _payload, key in rm.calls] == [first.idempotency_key]

        second_reopen = TrainerService(settings)
        second_reopen.close()
        assert [key for _payload, key in rm.calls] == [
            first.idempotency_key,
            second_key,
        ]
        assert rm.timeouts == [5, 5]


def test_run_job_uses_parameter_server_then_durable_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        campaign = _seed_campaign_start(service)
        job = TrainingJob(
            jobId=JOB,
            campaignId=CAMPAIGN,
            facetId="facet",
            candidateId="default",
        )
        job = TrainingJob.model_validate(
            service.store.put_entity("job", JOB, job.api_dict())
        )
        artifact = service.settings.home / "artifacts" / "parameter-server-result"
        artifact.mkdir(parents=True)
        weights = artifact / "facet.safetensors"
        weights.write_bytes(b"deterministic-parameter-server-output")
        artifact_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
        observed: dict[str, object] = {"runs": 0}

        class FakeParameterServer:
            def __init__(self, **kwargs: object) -> None:
                observed["attempt"] = kwargs["attempt"]
                observed["job"] = kwargs["job"]
                self.session_ids: dict[str, str] = {}

            def run(self) -> SimpleNamespace:
                observed["runs"] = int(observed["runs"]) + 1
                return SimpleNamespace(
                    facet_reference="facet@v1",
                    artifact_path=artifact,
                    artifact_sha256=artifact_sha256,
                    metrics={"loss": 0.25},
                )

        class DynamicRM:
            available = True
            topology_id = "topology-a"

            def __init__(self) -> None:
                self.calls: list[tuple[dict, str]] = []

            def register_trainer_publication(
                self,
                payload: dict,
                *,
                idempotency_key: str,
                timeout_seconds: float = 3600,
            ) -> dict:
                del timeout_seconds
                self.calls.append((payload, idempotency_key))
                stored = PublicationRegistration.model_validate(
                    service.store.get_entity(
                        "publication-registration", payload["publicationId"]
                    )
                )
                return _authority(stored)

        monkeypatch.setattr("jester_trainer.service.ParameterServer", FakeParameterServer)
        monkeypatch.setattr(service, "get_dataset", lambda _dataset_id: {})
        monkeypatch.setattr(service.datasets, "shard_plan", lambda *_args, **_kwargs: {})
        rm = DynamicRM()
        service.rm = rm  # type: ignore[assignment]

        result = service._run_job(
            campaign,
            job,
            [],
            attempt_number=1,
            start_operation_id=OPERATION,
        )

        assert observed["runs"] == 1
        assert result.result_id == RESULT
        assert result.artifact_digest == "sha256:" + "d" * 64
        assert len(rm.calls) == 1
        registration = service.store.get_entity(
            "publication-registration", PUBLICATION
        )
        assert registration["state"] == "acknowledged"
        assert service.store.get_entity("attempt", ATTEMPT)["state"] == "succeeded"
        assert service.store.get_entity("job", JOB)["state"] == "succeeded"
    finally:
        service.close()


def test_pending_recovery_reuses_key_and_never_settles(tmp_path: Path) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        _seed_execution(service)
        registration = _draft(service)
        pending = RMReply(_authority(registration), fail=True)
        service.rm = pending  # type: ignore[assignment]
        service._recover_publications()
        service._recover_publications()
        assert [call[1] for call in pending.calls] == [
            registration.idempotency_key, registration.idempotency_key
        ]
        assert service.store.get_entity("publication-registration", PUBLICATION)["state"] == "reconciliation-required"
        assert service.store.get_entity("attempt", ATTEMPT)["state"] == "running"
        assert len(service.store.list_entities("attempt")) == 1
    finally:
        service.close()


def test_same_key_changed_durable_request_is_refused_before_rm(tmp_path: Path) -> None:
    service = TrainerService(_settings(tmp_path))
    try:
        _seed_execution(service)
        registration = _draft(service)
        changed = registration.model_copy(update={"request_sha256": "0" * 64})
        service.rm = RMReply(_authority(registration))  # type: ignore[assignment]
        with pytest.raises(TrainerError) as error:
            service._register_reserved_publication(changed)
        assert error.value.code == "artifact_publication_intent_changed"
        assert service.rm.calls == []  # type: ignore[attr-defined]
    finally:
        service.close()
