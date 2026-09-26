from __future__ import annotations

import multiprocessing
import os
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from jester_trainer.api import create_app
from jester_trainer.channel_sessions import ManagedChannelContext, ManagedChannelGuard
from jester_trainer.config import Settings
from jester_trainer.errors import ConflictError, TrainerError
from jester_trainer.models import (
    CampaignState, DatasetManifest, Operation, TrainingAttempt, TrainingJob,
)
from jester_trainer.parameter_server import ParameterServer
from jester_trainer.service import TrainerService
from jester_trainer.workers import WorkerDescriptor

from .helpers import campaign, dataset_manifest


TOPOLOGY = {
    "topologyId": "test-training-topology",
    "topologyType": "training",
    "role": "trainer",
    "nodeId": "trainer",
    "channels": [],
    "nodes": [],
}


def _settings(root: Path | str) -> Settings:
    return Settings.for_test(root, topology=TOPOLOGY)


def _planned_campaign(service: TrainerService, root: Path, campaign_id: str) -> None:
    data = root / f"{campaign_id}.jsonl"
    data.write_text('{"id":"one","text":"hello"}\n', encoding="utf-8")
    if not service.store.list_entities("dataset"):
        service.put_dataset(dataset_manifest(data))
    model = campaign(
        root,
        root / "model",
        campaign_id=campaign_id,
        version="v1",
        workers=[],
    )
    service.put_campaign(model)
    current = service.get_campaign(campaign_id)
    planned = current.model_copy(update={"state": CampaignState.PLANNED})
    service.store.put_entity(
        "campaign",
        campaign_id,
        planned.api_dict(),
        expected_generation=current.generation,
    )


def _managed_artifact(root: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    data = root / "managed.jsonl"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text('{"id":"one","text":"hello"}\n', encoding="utf-8")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    artifact_manifest = root / "rm-artifacts.json"
    artifact_manifest.write_text(json.dumps({"entries": [{
        "digest": f"sha256:{digest}", "kind": "file", "checkoutPath": str(data),
    }]}), encoding="utf-8")
    monkeypatch.setenv("JESTER_RM_ARTIFACT_MANIFEST", str(artifact_manifest))
    return digest


def _planned_managed_campaign(
    service: TrainerService, root: Path, campaign_id: str, digest: str
) -> None:
    service.register_managed_dataset(
        DatasetManifest.model_validate({
            "metadata": {"id": "tiny-dataset"},
            "files": [{
                "split": "train", "digest": f"sha256:{digest}",
                "artifactDigest": f"sha256:{digest}", "logicalPath": "managed.jsonl",
            }],
        }),
        project_id="project-a",
        deployment_id="deployment-a",
    )
    service.put_managed_campaign(
        campaign(
            root, root / "model", campaign_id=campaign_id, version="v1", workers=[]
        ),
        project_id="project-a",
        deployment_id="deployment-a",
    )
    current = service.get_campaign(campaign_id)
    planned = current.model_copy(update={"state": CampaignState.PLANNED})
    service.store.put_entity(
        "campaign", campaign_id, planned.api_dict(), expected_generation=current.generation
    )


def _draft_campaign(service: TrainerService, root: Path, campaign_id: str) -> None:
    data = root / f"{campaign_id}.jsonl"
    data.write_text('{"id":"one","text":"hello"}\n', encoding="utf-8")
    if not service.store.list_entities("dataset"):
        service.put_dataset(dataset_manifest(data))
    service.put_campaign(
        campaign(
            root,
            root / "model",
            campaign_id=campaign_id,
            version="v1",
            workers=[],
        )
    )


def _process_start(
    root: str,
    campaign_id: str,
    operation_id: str,
    ready: Any,
    go: Any,
    output: Any,
) -> None:
    settings = _settings(root)
    with patch.object(
        Settings,
        "initialize",
        lambda current: current.token_path.read_text(encoding="utf-8").strip(),
    ):
        service = TrainerService(settings)
    service._spawn = lambda _name, _action: output.put(("spawn", operation_id))  # type: ignore[method-assign]
    ready.put(operation_id)
    go.wait(15)
    try:
        output.put(("result", service.start_campaign(campaign_id, operation_id=operation_id)))
    except BaseException as exc:
        output.put(("error", {"type": type(exc).__name__, "code": getattr(exc, "code", None)}))
    finally:
        service.close()


def _crash_during_start(
    root: str, campaign_id: str, operation_id: str, crash_point: str
) -> None:
    service = TrainerService(_settings(root))
    if crash_point == "before-launch-commit":
        service.plan_campaign = lambda *_args, **_kwargs: os._exit(70)  # type: ignore[method-assign]
    elif crash_point == "after-launch-commit":
        service._spawn = lambda *_args, **_kwargs: os._exit(71)  # type: ignore[method-assign]
    else:
        service._spawn = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    service.start_campaign(campaign_id, operation_id=operation_id)
    os._exit(72)


def test_cancel_intent_precedes_signal_and_terminal_waits_for_worker_unwind(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    _planned_campaign(service, root, "campaign-cancel")
    start_operation_id = "engine.trainer." + "1" * 32
    cancel_operation_id = "engine.trainer.cancel." + "2" * 32
    actions: list[Any] = []
    service._spawn = lambda _name, action: actions.append(action)  # type: ignore[method-assign]
    service.start_campaign("campaign-cancel", operation_id=start_operation_id)
    controls = service._controls("campaign-cancel")
    original_set = controls["cancel"].set
    observed: list[tuple[str, str]] = []

    def observe_set() -> None:
        reservation = service.store.campaign_cancel_reservation("campaign-cancel")
        observed.append(
            (
                str(reservation and reservation["phase"]),
                service.get_campaign("campaign-cancel").state,
            )
        )
        original_set()
        actions[0]()

    controls["cancel"].set = observe_set  # type: ignore[method-assign]
    requested = service.cancel_campaign(
        "campaign-cancel",
        start_operation_id=start_operation_id,
        cancel_operation_id=cancel_operation_id,
    )

    assert observed == [("signal-committed", CampaignState.CANCELLING)]
    assert requested["state"] == "succeeded"

    execution = service.campaign_execution(
        "campaign-cancel", operation_id=start_operation_id
    )
    assert execution["campaign"]["state"] == "cancelled"
    assert execution["operation"]["state"] == "cancelled"
    assert execution["cancellation"]["phase"] == "terminal"
    assert execution["cancellation"]["operation"]["state"] == "succeeded"
    assert all(item["state"] == "cancelled" for item in execution["jobs"])
    service.close()


def test_cancel_exact_replay_and_conflicting_operation_are_fenced(tmp_path: Path) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    _planned_campaign(service, root, "campaign-replay")
    start_operation_id = "engine.trainer." + "3" * 32
    cancel_operation_id = "engine.trainer.cancel." + "4" * 32
    service._spawn = lambda _name, _action: None  # type: ignore[method-assign]
    service.start_campaign("campaign-replay", operation_id=start_operation_id)

    first = service.cancel_campaign(
        "campaign-replay",
        start_operation_id=start_operation_id,
        cancel_operation_id=cancel_operation_id,
    )
    replay = service.cancel_campaign(
        "campaign-replay",
        start_operation_id=start_operation_id,
        cancel_operation_id=cancel_operation_id,
    )
    assert replay == first

    with pytest.raises(ConflictError) as conflict:
        service.cancel_campaign(
            "campaign-replay",
            start_operation_id=start_operation_id,
            cancel_operation_id="engine.trainer.cancel." + "5" * 32,
        )
    assert conflict.value.code == "campaign_cancel_reserved"
    service.close()


def test_natural_terminal_before_cancel_is_preserved(tmp_path: Path) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    _planned_campaign(service, root, "campaign-late")
    start_operation_id = "engine.trainer." + "6" * 32
    service._spawn = lambda _name, _action: None  # type: ignore[method-assign]
    service.start_campaign("campaign-late", operation_id=start_operation_id)
    assert service.store.settle_campaign_succeeded(
        "campaign-late",
        start_operation_id,
        {"campaignId": "campaign-late", "results": []},
        controller_id=service._controller_lease.instance_id,
    )

    with pytest.raises(ConflictError) as conflict:
        service.cancel_campaign(
            "campaign-late",
            start_operation_id=start_operation_id,
            cancel_operation_id="engine.trainer.cancel." + "7" * 32,
        )
    assert conflict.value.code == "campaign_cancel_too_late"
    assert service.get_campaign("campaign-late").state == CampaignState.SUCCEEDED
    assert service.get_operation(start_operation_id)["state"] == "succeeded"
    assert service.store.campaign_cancel_reservation("campaign-late") is None
    service.close()


def test_reopen_fences_ambiguous_cancel_without_resignal_or_terminal_overwrite(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    _planned_campaign(service, root, "campaign-unknown")
    start_operation_id = "engine.trainer." + "8" * 32
    cancel_operation_id = "engine.trainer.cancel." + "9" * 32
    service._spawn = lambda _name, _action: None  # type: ignore[method-assign]
    service.start_campaign("campaign-unknown", operation_id=start_operation_id)
    service.cancel_campaign(
        "campaign-unknown",
        start_operation_id=start_operation_id,
        cancel_operation_id=cancel_operation_id,
    )
    stale_controller = service._controller_lease.instance_id
    service.close()

    reopened = TrainerService(_settings(root))
    execution = reopened.campaign_execution(
        "campaign-unknown", operation_id=start_operation_id
    )
    assert execution["campaign"]["state"] == "cancelling"
    assert execution["operation"]["state"] == "reconciliation-required"
    assert execution["cancellation"]["phase"] == "reconciliation-required"
    assert execution["cancellation"]["operation"]["state"] == "reconciliation-required"
    assert not reopened.store.settle_campaign_cancelled(
        "campaign-unknown",
        start_operation_id,
        controller_id=stale_controller,
    )
    assert not reopened.store.settle_campaign_failed(
        "campaign-unknown",
        start_operation_id,
        {"code": "stale", "detail": "stale writer"},
        controller_id=stale_controller,
    )
    assert reopened.get_operation(start_operation_id)["state"] == "reconciliation-required"
    reopened.close()


def test_managed_cancel_requires_exact_ids_and_returns_operation_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "trainer"
    digest = _managed_artifact(root, monkeypatch)
    service = TrainerService(_settings(root))
    _planned_managed_campaign(service, root, "campaign-api-cancel", digest)
    service._spawn = lambda _name, _action: None  # type: ignore[method-assign]
    start_operation_id = "engine.trainer." + "a" * 32
    cancel_operation_id = "engine.trainer.cancel." + "b" * 32
    headers = {
        "X-Jester-Component-Session": "owner-a",
        "Idempotency-Key": "cancel-api",
    }
    with TestClient(
        create_app(
            service,
            managed_channel_guard=_managed_guard(),
            require_managed_session=True,
        )
    ) as client:
        started = client.post(
            "/v1/campaigns/campaign-api-cancel/actions/start",
            headers={**headers, "Idempotency-Key": "start-api"},
            json={"operationId": start_operation_id},
        )
        invalid = client.post(
            "/v1/campaigns/campaign-api-cancel/actions/cancel",
            headers=headers,
            json={"startOperationId": start_operation_id},
        )
        cancelled = client.post(
            "/v1/campaigns/campaign-api-cancel/actions/cancel",
            headers=headers,
            json={
                "startOperationId": start_operation_id,
                "cancelOperationId": cancel_operation_id,
            },
        )
        assert started.status_code == 202
        assert invalid.status_code == 422
        assert cancelled.status_code == 202
        assert cancelled.json()["operationId"] == cancel_operation_id
        assert cancelled.json()["action"] == "campaign.cancel"
        assert cancelled.json()["state"] == "running"


def test_cancel_rejects_wrong_controller_and_reconciliation_start(tmp_path: Path) -> None:
    root = tmp_path / "trainer"
    owner = TrainerService(_settings(root))
    peer = TrainerService(_settings(root))
    _planned_campaign(owner, root, "campaign-owner-fence")
    start_operation_id = "engine.trainer." + "c" * 32
    owner._spawn = lambda _name, _action: None  # type: ignore[method-assign]
    owner.start_campaign("campaign-owner-fence", operation_id=start_operation_id)

    with pytest.raises(ConflictError) as wrong_owner:
        peer.cancel_campaign(
            "campaign-owner-fence",
            start_operation_id=start_operation_id,
            cancel_operation_id="engine.trainer.cancel." + "d" * 32,
        )
    assert wrong_owner.value.code == "campaign_cancel_start_not_active"
    assert owner.store.campaign_cancel_reservation("campaign-owner-fence") is None

    assert owner.store.mark_campaign_start_reconciliation_required(
        "campaign-owner-fence",
        start_operation_id,
        controller_id=owner._controller_lease.instance_id,
        reason="test fence",
    )
    with pytest.raises(ConflictError) as reconciliation:
        owner.cancel_campaign(
            "campaign-owner-fence",
            start_operation_id=start_operation_id,
            cancel_operation_id="engine.trainer.cancel." + "e" * 32,
        )
    assert reconciliation.value.code == "campaign_cancel_start_not_active"
    assert owner.store.campaign_cancel_reservation("campaign-owner-fence") is None
    peer.close()
    owner.close()


def test_cancel_settlement_uses_exact_campaign_job_ownership(tmp_path: Path) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    _planned_campaign(service, root, "foo")
    own_job = TrainingJob(
        jobId="shared-looking-id", campaignId="foo", facetId="one", candidateId="default"
    )
    foreign_job = TrainingJob(
        jobId="foo-bar-job", campaignId="foo-bar", facetId="one", candidateId="default"
    )
    service.store.put_entity("job", own_job.job_id, own_job.api_dict())
    service.store.put_entity("job", foreign_job.job_id, foreign_job.api_dict())
    own_attempt = TrainingAttempt(
        attemptId="attempt-own", jobId=own_job.job_id, number=1, cohort=[]
    )
    foreign_attempt = TrainingAttempt(
        attemptId="attempt-foreign", jobId=foreign_job.job_id, number=1, cohort=[]
    )
    service.store.put_entity("attempt", own_attempt.attempt_id, own_attempt.api_dict())
    service.store.put_entity("attempt", foreign_attempt.attempt_id, foreign_attempt.api_dict())
    service._spawn = lambda _name, _action: None  # type: ignore[method-assign]
    start_operation_id = "engine.trainer." + "f" * 32
    service.start_campaign("foo", operation_id=start_operation_id)
    service.cancel_campaign(
        "foo",
        start_operation_id=start_operation_id,
        cancel_operation_id="engine.trainer.cancel." + "0" * 32,
    )
    assert service.store.settle_campaign_cancelled(
        "foo", start_operation_id, controller_id=service._controller_lease.instance_id
    )
    assert service.store.get_entity("attempt", "attempt-own")["state"] == "cancelled"
    assert service.store.get_entity("attempt", "attempt-foreign")["state"] == "running"
    service.close()


def test_terminal_success_cannot_be_replaced_by_late_failure(tmp_path: Path) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    _planned_campaign(service, root, "campaign-terminal")
    service._spawn = lambda _name, _action: None  # type: ignore[method-assign]
    start_operation_id = "engine.trainer." + "1" * 32
    service.start_campaign("campaign-terminal", operation_id=start_operation_id)
    assert service.store.settle_campaign_succeeded(
        "campaign-terminal",
        start_operation_id,
        {"campaignId": "campaign-terminal", "results": []},
        controller_id=service._controller_lease.instance_id,
    )
    assert not service.store.settle_campaign_failed(
        "campaign-terminal",
        start_operation_id,
        {"code": "late_event_failure", "detail": "event write failed"},
        controller_id=service._controller_lease.instance_id,
    )
    assert service.get_campaign("campaign-terminal").state == CampaignState.SUCCEEDED
    assert service.get_operation(start_operation_id)["state"] == "succeeded"
    service.close()


def test_worker_settles_cancel_between_reservation_and_signal_commit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    _planned_campaign(service, root, "campaign-reserve-signal")
    start_operation_id = "engine.trainer." + "4" * 32
    cancel_operation_id = "engine.trainer.cancel." + "5" * 32
    actions: list[Any] = []
    service._spawn = lambda _name, action: actions.append(action)  # type: ignore[method-assign]
    service.start_campaign(
        "campaign-reserve-signal", operation_id=start_operation_id
    )
    operation = Operation(
        operationId=cancel_operation_id,
        action="campaign.cancel",
        targetId="campaign-reserve-signal",
    )
    _, created = service.store.reserve_campaign_cancel(
        "campaign-reserve-signal",
        start_operation_id,
        operation.api_dict(),
        controller_id=service._controller_lease.instance_id,
    )
    assert created
    assert (
        service.store.campaign_cancel_reservation("campaign-reserve-signal")["phase"]
        == "requested"
    )

    actions[0]()

    canonical = service.store.mark_campaign_cancel_signal_committed(
        "campaign-reserve-signal",
        cancel_operation_id,
        controller_id=service._controller_lease.instance_id,
    )
    assert canonical["state"] == "succeeded"
    assert service.get_campaign("campaign-reserve-signal").state == "cancelled"
    assert service.get_operation(start_operation_id)["state"] == "cancelled"
    assert service.store.campaign_cancel_reservation("campaign-reserve-signal")[
        "phase"
    ] == "terminal"
    service.close()


def test_lost_start_fence_never_reaches_worker_pool_or_failure_event(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    _planned_campaign(service, root, "campaign-lost-fence")
    start_operation_id = "engine.trainer." + "6" * 32
    actions: list[Any] = []
    service._spawn = lambda _name, action: actions.append(action)  # type: ignore[method-assign]
    service.start_campaign("campaign-lost-fence", operation_id=start_operation_id)
    assert service.store.mark_campaign_start_reconciliation_required(
        "campaign-lost-fence",
        start_operation_id,
        controller_id=service._controller_lease.instance_id,
        reason="deterministic lost-fence test",
    )
    service.pool.select = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("worker pool must not be reached after a lost start fence")
    )
    emitted: list[str] = []
    service._event = lambda event_type, **_kwargs: emitted.append(event_type)  # type: ignore[method-assign]

    actions[0]()

    assert service.get_operation(start_operation_id)["state"] == "reconciliation-required"
    assert "CAMPAIGN_FAILED" not in emitted
    service.close()


def test_event_failure_after_success_does_not_publish_false_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    _planned_campaign(service, root, "campaign-event-race")
    start_operation_id = "engine.trainer." + "7" * 32
    actions: list[Any] = []
    service._spawn = lambda _name, action: actions.append(action)  # type: ignore[method-assign]
    service.pool.select = lambda *_args, **_kwargs: []  # type: ignore[method-assign]
    emitted: list[str] = []

    def fail_success_event(event_type: str, **_kwargs: Any) -> None:
        emitted.append(event_type)
        if event_type == "CAMPAIGN_SUCCEEDED":
            raise OSError("event sink unavailable")

    service._event = fail_success_event  # type: ignore[method-assign]
    service.start_campaign("campaign-event-race", operation_id=start_operation_id)

    actions[0]()

    assert service.get_campaign("campaign-event-race").state == "succeeded"
    assert service.get_operation(start_operation_id)["state"] == "succeeded"
    assert emitted == ["CAMPAIGN_SUCCEEDED"]
    service.close()


def test_failed_terminal_cancel_cannot_be_reclassified_as_cancelled(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    _planned_campaign(service, root, "campaign-cancel-failed")
    start_operation_id = "engine.trainer." + "8" * 32
    cancel_operation_id = "engine.trainer.cancel." + "9" * 32
    service._spawn = lambda _name, _action: None  # type: ignore[method-assign]
    service.start_campaign("campaign-cancel-failed", operation_id=start_operation_id)
    service.cancel_campaign(
        "campaign-cancel-failed",
        start_operation_id=start_operation_id,
        cancel_operation_id=cancel_operation_id,
    )
    assert service.store.settle_campaign_failed(
        "campaign-cancel-failed",
        start_operation_id,
        {"code": "worker_failed", "detail": "natural failure won"},
        controller_id=service._controller_lease.instance_id,
    )

    assert not service.store.settle_campaign_cancelled(
        "campaign-cancel-failed",
        start_operation_id,
        controller_id=service._controller_lease.instance_id,
    )
    with pytest.raises(ConflictError) as terminal:
        service.store.mark_campaign_cancel_signal_committed(
            "campaign-cancel-failed",
            cancel_operation_id,
            controller_id=service._controller_lease.instance_id,
        )
    assert terminal.value.code == "campaign_cancel_terminal_conflict"
    assert service.get_campaign("campaign-cancel-failed").state == "failed"
    assert service.get_operation(start_operation_id)["state"] == "failed"
    assert service.get_operation(cancel_operation_id)["state"] == "failed"
    service.close()


def test_lost_worker_reply_and_cleanup_require_reconciliation_without_retry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    _planned_campaign(service, root, "campaign-worker-unknown")
    job = TrainingJob(
        jobId="campaign-worker-unknown-job",
        campaignId="campaign-worker-unknown",
        facetId="tiny-coder",
        candidateId="default",
    )
    service.store.put_entity("job", job.job_id, job.api_dict())
    calls: list[tuple[str, str]] = []

    class AmbiguousWorker:
        descriptor = WorkerDescriptor(
            worker_id="worker-1",
            node_id="worker-1",
            instance_id="worker-1-instance",
            base_url="http://127.0.0.1:1",
            transport="local-loopback",
            channel_id="trainer-to-worker-1",
            protocol="http-json+binary",
            resumable=True,
            requested_backend="cpu",
            effective_backend="cpu",
        )

        def json(self, method: str, path: str, _payload: Any = None) -> dict[str, Any]:
            calls.append((method, path))
            detail = "create reply lost" if method == "POST" else "delete reply lost"
            raise TrainerError(detail, code="worker_unavailable", status=503)

    worker = AmbiguousWorker()
    service.pool.select = lambda *_args, **_kwargs: [worker]  # type: ignore[method-assign]
    actions: list[Any] = []
    service._spawn = lambda _name, action: actions.append(action)  # type: ignore[method-assign]
    start_operation_id = "engine.trainer." + "a" * 32
    service.start_campaign(
        "campaign-worker-unknown", operation_id=start_operation_id
    )
    actions[0]()

    assert calls == [
        ("POST", "/v1/sessions"),
        ("DELETE", f"/v1/sessions/{job.job_id}-a1-worker-1"),
    ]
    assert service.get_operation(start_operation_id)["state"] == "reconciliation-required"
    assert service.get_campaign("campaign-worker-unknown").state == "recovering"
    assert service.get_job(job.job_id)["state"] == "running"
    attempt = service.store.get_entity("attempt", f"{job.job_id}-a1")
    assert attempt["state"] == "reconciliation-required"
    assert attempt["error"]["cause"]["code"] == "worker_unavailable"
    observation = service.store.get_entity(
        "worker-session", f"{job.job_id}-a1-worker-1"
    )
    assert observation["createState"] == "dispatched"
    assert observation["cleanupState"] == "outcome-unknown"
    assert not any(
        event["type"] in {"CAMPAIGN_FAILED", "CAMPAIGN_CANCELLED"}
        for event in service.store.list_events()
    )
    service.close()


def test_parameter_server_safe_frontier_can_settle_cancel_with_lost_delete_reply(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    _planned_campaign(service, root, "campaign-safe-frontier")
    job = TrainingJob(
        jobId="campaign-safe-frontier-job",
        campaignId="campaign-safe-frontier",
        facetId="tiny-coder",
        candidateId="default",
    )
    service.store.put_entity("job", job.job_id, job.api_dict())
    calls: list[tuple[str, str]] = []

    class SafeBoundaryWorker:
        descriptor = WorkerDescriptor(
            worker_id="worker-1",
            node_id="worker-1",
            instance_id="worker-1-instance",
            base_url="http://127.0.0.1:1",
            transport="local-loopback",
            channel_id="trainer-to-worker-1",
            protocol="http-json+binary",
            resumable=True,
            requested_backend="cpu",
            effective_backend="cpu",
        )

        def json(self, method: str, path: str, payload: Any = None) -> dict[str, Any]:
            calls.append((method, path))
            if method == "POST":
                return {
                    "id": payload["sessionId"],
                    "weightVersion": "worker-init",
                    "weightSha256": "0" * 64,
                    "engine": {"tensorSpecs": {}, "base": {}},
                }
            raise TrainerError(
                "delete reply lost", code="worker_unavailable", status=503
            )

        def download(
            self,
            _path: str,
            destination: Path,
            **_kwargs: Any,
        ) -> Path:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"not-read-in-this-test")
            return destination

    worker = SafeBoundaryWorker()
    service.pool.select = lambda *_args, **_kwargs: [worker]  # type: ignore[method-assign]
    actions: list[Any] = []
    service._spawn = lambda _name, action: actions.append(action)  # type: ignore[method-assign]
    start_operation_id = "engine.trainer." + "b" * 32
    cancel_operation_id = "engine.trainer.cancel." + "c" * 32

    def save_weights(server: ParameterServer, version: str) -> Path:
        path = server.output / "coordinator" / f"{version}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
        return path

    def cancel_at_safe_boundary(
        _server: ParameterServer, _path: Path, _version: str
    ) -> None:
        service.cancel_campaign(
            "campaign-safe-frontier",
            start_operation_id=start_operation_id,
            cancel_operation_id=cancel_operation_id,
        )

    with (
        patch.object(ParameterServer, "_initialize_parameters", lambda *_args: None),
        patch.object(ParameterServer, "_build_optimizer", lambda *_args: None),
        patch.object(ParameterServer, "_resume_if_requested", lambda *_args: None),
        patch.object(ParameterServer, "_save_coordinator_weights", save_weights),
        patch.object(ParameterServer, "_publish_weights", cancel_at_safe_boundary),
    ):
        service.start_campaign(
            "campaign-safe-frontier", operation_id=start_operation_id
        )
        actions[0]()

    assert service.get_campaign("campaign-safe-frontier").state == "cancelled"
    assert service.get_operation(start_operation_id)["state"] == "cancelled"
    assert service.get_operation(cancel_operation_id)["state"] == "succeeded"
    attempt = service.store.get_entity("attempt", f"{job.job_id}-a1")
    assert attempt["state"] == "cancelled"
    observation = service.store.get_entity(
        "worker-session", f"{job.job_id}-a1-worker-1"
    )
    assert observation["createState"] == "acknowledged"
    assert observation["cleanupState"] == "outcome-unknown"
    assert calls == [
        ("POST", "/v1/sessions"),
        ("DELETE", f"/v1/sessions/{job.job_id}-a1-worker-1"),
    ]
    service.close()


@pytest.mark.parametrize(
    ("other_code", "other_first"),
    [
        ("remote_session_outcome_unknown", False),
        ("remote_session_outcome_unknown", True),
        ("natural_worker_failure", False),
        ("natural_worker_failure", True),
    ],
)
def test_parallel_job_error_precedence_is_independent_of_completion_order(
    tmp_path: Path, other_code: str, other_first: bool
) -> None:
    root = tmp_path / f"trainer-{other_code}-{other_first}"
    service = TrainerService(_settings(root))
    campaign_id = f"campaign-precedence-{other_code}-{other_first}".replace("_", "-")
    _planned_campaign(service, root, campaign_id)
    current_campaign = service.get_campaign(campaign_id)
    independent = current_campaign.model_copy(
        update={
            "spec": current_campaign.spec.model_copy(update={"mode": "independent"})
        }
    )
    service.store.put_entity(
        "campaign",
        campaign_id,
        independent.api_dict(),
        expected_generation=current_campaign.generation,
    )
    cancel_job = TrainingJob(
        jobId=f"{campaign_id}-cancel",
        campaignId=campaign_id,
        facetId="tiny-coder",
        candidateId="default",
    )
    other_job = TrainingJob(
        jobId=f"{campaign_id}-other",
        campaignId=campaign_id,
        facetId="tiny-coder",
        candidateId="default",
    )
    for job in (cancel_job, other_job):
        service.store.put_entity("job", job.job_id, job.api_dict())
    workers = [
        type(
            "ParallelWorker",
            (),
            {
                "descriptor": WorkerDescriptor(
                    worker_id=f"worker-{index}",
                    node_id=f"worker-{index}",
                    instance_id=f"worker-{index}-instance",
                    base_url="http://127.0.0.1:1",
                    transport="local-loopback",
                    channel_id=f"trainer-to-worker-{index}",
                    protocol="http-json+binary",
                    resumable=True,
                    requested_backend="cpu",
                    effective_backend="cpu",
                )
            },
        )()
        for index in (1, 2)
    ]
    service.pool.select = lambda *_args, **_kwargs: workers  # type: ignore[method-assign]
    actions: list[Any] = []
    service._spawn = lambda _name, action: actions.append(action)  # type: ignore[method-assign]
    start_operation_id = "engine.trainer." + ("d" if other_first else "e") * 32
    cancel_operation_id = "engine.trainer.cancel." + ("f" if other_first else "0") * 32
    barrier = threading.Barrier(2)

    def execute_job(_campaign: Any, job: TrainingJob, _workers: Any) -> None:
        barrier.wait(timeout=5)
        is_other = job.job_id == other_job.job_id
        finishes_first = is_other == other_first
        if not finishes_first:
            time.sleep(0.1)
        if is_other:
            raise TrainerError(
                f"{other_code} from {job.job_id}",
                code=other_code,
                status=503 if other_code == "remote_session_outcome_unknown" else 500,
            )
        service.cancel_campaign(
            campaign_id,
            start_operation_id=start_operation_id,
            cancel_operation_id=cancel_operation_id,
        )
        raise TrainerError(
            "clean safe-boundary cancellation",
            code="campaign_cancelled",
            status=409,
            fields={"safeBoundary": True},
        )

    service._run_job_with_retries = execute_job  # type: ignore[method-assign]
    service.start_campaign(campaign_id, operation_id=start_operation_id)
    actions[0]()

    if other_code == "remote_session_outcome_unknown":
        execution = service.campaign_execution(
            campaign_id, operation_id=start_operation_id
        )
        assert execution["campaign"]["state"] == "cancelling"
        assert execution["operation"]["state"] == "reconciliation-required"
        assert execution["cancellation"]["phase"] == "reconciliation-required"
        assert execution["cancellation"]["operation"]["state"] == "reconciliation-required"
        error = execution["cancellation"]["operation"]["error"]
        assert service.store.active_campaign_start(campaign_id) is not None
    else:
        assert service.get_campaign(campaign_id).state == "failed"
        assert service.get_operation(start_operation_id)["state"] == "failed"
        assert service.get_operation(cancel_operation_id)["state"] == "failed"
        error = service.get_operation(start_operation_id)["error"]
    assert [item["jobId"] for item in error["jobErrors"]] == sorted(
        [cancel_job.job_id, other_job.job_id]
    )
    assert {item["code"] for item in error["jobErrors"]} == {
        "campaign_cancelled",
        other_code,
    }
    service.close()


def test_cancel_reservation_atomically_fences_running_transition(tmp_path: Path) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    _planned_campaign(service, root, "campaign-start-race")
    service._spawn = lambda _name, _action: None  # type: ignore[method-assign]
    start_operation_id = "engine.trainer." + "2" * 32
    service.start_campaign("campaign-start-race", operation_id=start_operation_id)
    service.cancel_campaign(
        "campaign-start-race",
        start_operation_id=start_operation_id,
        cancel_operation_id="engine.trainer.cancel." + "3" * 32,
    )
    assert not service.store.mark_campaign_start_running(
        "campaign-start-race",
        start_operation_id,
        controller_id=service._controller_lease.instance_id,
    )
    assert service.get_campaign("campaign-start-race").state == CampaignState.CANCELLING
    assert service.get_operation(start_operation_id)["state"] == "queued"
    service.close()


def test_legacy_admin_cancel_without_operation_ids_remains_safe(tmp_path: Path) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    _planned_campaign(service, root, "campaign-admin")
    headers = {"Authorization": f"Bearer {service.token}", "Idempotency-Key": "admin-cancel"}
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/campaigns/campaign-admin/actions/cancel", headers=headers, json={}
        )
        assert response.status_code == 202
        assert response.json()["state"] == "cancelled"
        assert service.store.campaign_cancel_reservation("campaign-admin") is None


def _managed_guard() -> ManagedChannelGuard:
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

    return ManagedChannelGuard(
        ManagedChannelContext(rm_url="http://unused"),
        introspect=introspect,
        expectation_provider=local,
    )


class _PlanningWorker:
    def __init__(self) -> None:
        self.descriptor = WorkerDescriptor(
            worker_id="worker-1",
            node_id="worker-1",
            instance_id="worker-1-instance",
            base_url="http://unused",
            transport="local-loopback",
            channel_id="trainer-to-worker-1",
            protocol="http-json+binary",
            resumable=True,
        )

    def json(self, _method: str, _path: str) -> dict[str, object]:
        return {
            "backends": [
                {
                    "backend": "cpu",
                    "workerCompatible": True,
                    "autogradCapable": True,
                }
            ]
        }


def test_two_services_replay_same_operation_without_duplicate_plan_or_spawn(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trainer"
    first = TrainerService(_settings(root))
    second = TrainerService(_settings(root))
    _planned_campaign(first, root, "campaign-one")
    operation_id = "engine.trainer." + "a" * 32
    spawns: list[str] = []
    first._spawn = lambda _name, _action: spawns.append("first")  # type: ignore[method-assign]
    second._spawn = lambda _name, _action: spawns.append("second")  # type: ignore[method-assign]

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(
            executor.map(
                lambda service: service.start_campaign(
                    "campaign-one", operation_id=operation_id
                ),
                (first, second),
            )
        )

    assert responses[0] == responses[1]
    assert responses[0]["operationId"] == operation_id
    assert responses[0]["state"] == "queued"
    assert len(spawns) == 1
    first.close()
    second.close()


def test_draft_campaign_same_operation_plans_once_across_services(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trainer"
    first = TrainerService(_settings(root))
    second = TrainerService(_settings(root))
    _draft_campaign(first, root, "campaign-draft")
    operation_id = "engine.trainer." + "1" * 32
    plan_calls: list[str] = []
    plan_lock = threading.Lock()
    worker = _PlanningWorker()
    first.pool.select = lambda *_args, **_kwargs: [worker]  # type: ignore[method-assign, return-value]
    second.pool.select = lambda *_args, **_kwargs: [worker]  # type: ignore[method-assign, return-value]

    def count_plan(service: TrainerService, label: str) -> Any:
        original = service.plan_campaign

        def planned(campaign_id: str, *, _start_operation_id: str | None = None) -> Any:
            with plan_lock:
                plan_calls.append(label)
            return original(
                campaign_id, _start_operation_id=_start_operation_id
            )

        return planned

    first.plan_campaign = count_plan(first, "first")  # type: ignore[method-assign]
    second.plan_campaign = count_plan(second, "second")  # type: ignore[method-assign]
    first._spawn = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    second._spawn = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(
            executor.map(
                lambda service: service.start_campaign(
                    "campaign-draft", operation_id=operation_id
                ),
                (first, second),
            )
        )

    assert responses[0] == responses[1]
    assert len(plan_calls) == 1
    assert first.get_campaign("campaign-draft").state == CampaignState.PLANNED
    first.close()
    second.close()


def test_public_plan_is_fenced_after_start_reservation(tmp_path: Path) -> None:
    root = tmp_path / "trainer"
    owner = TrainerService(_settings(root))
    peer = TrainerService(_settings(root))
    _draft_campaign(owner, root, "campaign-fenced")
    operation_id = "engine.trainer." + "2" * 32
    operation = {
        "operationId": operation_id,
        "action": "campaign.start",
        "targetId": "campaign-fenced",
        "state": "queued",
    }
    owner.store.reserve_campaign_start(
        "campaign-fenced",
        operation,
        controller_id=owner._controller_lease.instance_id,
    )

    with pytest.raises(ConflictError) as conflict:
        peer.plan_campaign("campaign-fenced")

    assert conflict.value.code == "campaign_start_reserved"
    assert peer.campaign_jobs("campaign-fenced") == []
    owner.close()
    peer.close()


def test_public_plan_final_commit_loses_to_concurrent_start_reservation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trainer"
    planner = TrainerService(_settings(root))
    starter = TrainerService(_settings(root))
    _draft_campaign(planner, root, "campaign-plan-race")
    validation_started = threading.Event()
    allow_commit = threading.Event()
    worker = _PlanningWorker()

    def delayed_select(*_args: Any, **_kwargs: Any) -> list[Any]:
        validation_started.set()
        assert allow_commit.wait(10)
        return [worker]

    planner.pool.select = delayed_select  # type: ignore[method-assign]
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(planner.plan_campaign, "campaign-plan-race")
        assert validation_started.wait(10)
        starter.store.reserve_campaign_start(
            "campaign-plan-race",
            {
                "operationId": "engine.trainer." + "4" * 32,
                "action": "campaign.start",
                "targetId": "campaign-plan-race",
                "state": "queued",
            },
            controller_id=starter._controller_lease.instance_id,
        )
        allow_commit.set()
        with pytest.raises(ConflictError) as conflict:
            future.result(timeout=10)

    assert conflict.value.code == "campaign_start_reserved"
    assert planner.campaign_jobs("campaign-plan-race") == []
    assert planner.get_campaign("campaign-plan-race").state == CampaignState.DRAFT
    planner.close()
    starter.close()


def test_close_keeps_controller_lease_while_campaign_thread_is_active(
    tmp_path: Path,
) -> None:
    service = TrainerService(_settings(tmp_path / "trainer"))
    release = threading.Event()
    service._spawn("campaign-blocked", lambda: release.wait(30))

    with pytest.raises(TrainerError) as conflict:
        service.close()

    assert conflict.value.code == "campaign_workers_active"
    assert service._controller_lease.owner_is_active(
        service._controller_lease.instance_id
    )
    release.set()
    for thread in list(service._threads):
        thread.join(timeout=5)
    service.close()


def test_escaped_campaign_runner_is_fenced_as_reconciliation_required(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    _planned_campaign(service, root, "campaign-runner-escaped")
    actions: list[Any] = []
    service._spawn = lambda _name, action: actions.append(action)  # type: ignore[method-assign]
    operation_id = "engine.trainer." + "4" * 32
    service.start_campaign("campaign-runner-escaped", operation_id=operation_id)

    def escaped(current_operation_id: str, current_campaign_id: str) -> None:
        assert service.store.mark_campaign_start_running(
            current_campaign_id,
            current_operation_id,
            controller_id=service._controller_lease.instance_id,
        )
        raise RuntimeError("runner settlement escaped")

    service._run_campaign = escaped  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="runner settlement escaped"):
        actions[0]()

    operation = service.get_operation(operation_id)
    assert operation["state"] == "reconciliation-required"
    assert operation["error"]["code"] == "campaign_start_outcome_unknown"
    assert operation["error"]["causeCode"] == "campaign_runner_outcome_unknown"
    assert operation["error"]["exceptionType"] == "RuntimeError"
    assert service.get_campaign("campaign-runner-escaped").state == "recovering"
    service.close()


def test_escaped_campaign_runner_during_cancel_fences_both_operations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    campaign_id = "campaign-cancel-runner-escaped"
    _planned_campaign(service, root, campaign_id)
    actions: list[Any] = []
    service._spawn = lambda _name, action: actions.append(action)  # type: ignore[method-assign]
    start_operation_id = "engine.trainer." + "5" * 32
    cancel_operation_id = "engine.trainer.cancel." + "6" * 32
    service.start_campaign(campaign_id, operation_id=start_operation_id)
    assert service.store.mark_campaign_start_running(
        campaign_id,
        start_operation_id,
        controller_id=service._controller_lease.instance_id,
    )
    service.cancel_campaign(
        campaign_id,
        start_operation_id=start_operation_id,
        cancel_operation_id=cancel_operation_id,
    )

    def escaped(_operation_id: str, _campaign_id: str) -> None:
        raise RuntimeError("runner escaped during cancel")

    service._run_campaign = escaped  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="runner escaped during cancel"):
        actions[0]()

    execution = service.campaign_execution(
        campaign_id, operation_id=start_operation_id
    )
    assert execution["operation"]["state"] == "reconciliation-required"
    assert execution["operation"]["error"]["causeCode"] == (
        "campaign_runner_outcome_unknown"
    )
    assert execution["cancellation"]["phase"] == "reconciliation-required"
    assert execution["cancellation"]["operation"]["state"] == (
        "reconciliation-required"
    )
    assert execution["cancellation"]["operation"]["error"]["causeCode"] == (
        "campaign_runner_outcome_unknown"
    )
    assert service.store.active_campaign_start(campaign_id) is not None
    service.close()


def test_reconciliation_fences_late_controller_writes(tmp_path: Path) -> None:
    service = TrainerService(_settings(tmp_path / "trainer"))
    root = tmp_path / "trainer"
    _planned_campaign(service, root, "campaign-stale")
    operation_id = "engine.trainer." + "3" * 32
    operation = {
        "operationId": operation_id,
        "action": "campaign.start",
        "targetId": "campaign-stale",
        "state": "queued",
    }
    controller_id = service._controller_lease.instance_id
    service.store.reserve_campaign_start(
        "campaign-stale", operation, controller_id=controller_id
    )
    service.store.mark_campaign_start_launch_committed(
        "campaign-stale", operation_id, controller_id=controller_id
    )
    assert service.store.mark_campaign_start_reconciliation_required(
        "campaign-stale",
        operation_id,
        controller_id=controller_id,
        reason="test lost acknowledgement",
    )

    with pytest.raises(ConflictError) as operation_conflict:
        service.store.put_operation(
            {**operation, "state": "succeeded"}, controller_id=controller_id
        )
    assert operation_conflict.value.code == "campaign_start_reconciliation_required"

    campaign_state = service.get_campaign("campaign-stale")
    with pytest.raises(ConflictError) as campaign_conflict:
        service.store.put_entity(
            "campaign",
            "campaign-stale",
            campaign_state.model_copy(
                update={"state": CampaignState.SUCCEEDED}
            ).api_dict(),
            expected_generation=campaign_state.generation,
        )
    assert campaign_conflict.value.code == "campaign_start_reconciliation_required"
    service.close()


def test_campaign_write_guard_is_atomic_across_connections(tmp_path: Path) -> None:
    root = tmp_path / "trainer"
    writer = TrainerService(_settings(root))
    reconciler = TrainerService(_settings(root))
    _planned_campaign(writer, root, "campaign-atomic-fence")
    operation_id = "engine.trainer." + "6" * 32
    writer.store.reserve_campaign_start(
        "campaign-atomic-fence",
        {
            "operationId": operation_id,
            "action": "campaign.start",
            "targetId": "campaign-atomic-fence",
            "state": "queued",
        },
        controller_id=writer._controller_lease.instance_id,
    )
    campaign_state = writer.get_campaign("campaign-atomic-fence")
    stale = campaign_state.model_copy(
        update={"state": CampaignState.SUCCEEDED}
    ).api_dict()
    entered = threading.Event()

    with reconciler.store._lock:
        reconciler.store.connection.execute("BEGIN IMMEDIATE")
        try:
            reconciler.store.connection.execute(
                """
                UPDATE campaign_start_reservations
                SET phase='reconciliation-required'
                WHERE campaign_id=?
                """,
                ("campaign-atomic-fence",),
            )

            def stale_write() -> dict[str, Any]:
                entered.set()
                return writer.store.put_entity(
                    "campaign",
                    "campaign-atomic-fence",
                    stale,
                    expected_generation=campaign_state.generation,
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(stale_write)
                assert entered.wait(5)
                assert not future.done()
                reconciler.store.connection.commit()
                with pytest.raises(ConflictError) as conflict:
                    future.result(timeout=10)
        except BaseException:
            reconciler.store.connection.rollback()
            raise

    assert conflict.value.code == "campaign_start_reconciliation_required"
    writer.close()
    reconciler.close()


def test_closed_service_rejects_new_start(tmp_path: Path) -> None:
    root = tmp_path / "trainer"
    service = TrainerService(_settings(root))
    _planned_campaign(service, root, "campaign-closed")
    service.close()

    with pytest.raises(TrainerError) as conflict:
        service.start_campaign(
            "campaign-closed", operation_id="engine.trainer." + "5" * 32
        )

    assert conflict.value.code == "trainer_service_closing"


@pytest.mark.parametrize("same_operation", [True, False])
def test_sqlite_reservation_coordinates_independent_processes(
    tmp_path: Path, same_operation: bool
) -> None:
    root = tmp_path / "trainer"
    setup = TrainerService(_settings(root))
    _planned_campaign(setup, root, "campaign-process")
    setup.close()

    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    output = context.Queue()
    go = context.Event()
    operation_ids = [
        "engine.trainer." + "b" * 32,
        "engine.trainer." + ("b" if same_operation else "c") * 32,
    ]
    processes = [
        context.Process(
            target=_process_start,
            args=(str(root), "campaign-process", operation_id, ready, go, output),
        )
        for operation_id in operation_ids
    ]
    for process in processes:
        process.start()
    # Windows spawn imports the Trainer's native ML dependencies in each child.
    assert {ready.get(timeout=120) for _ in processes} == set(operation_ids)
    go.set()
    messages = [output.get(timeout=120) for _ in range(3)]
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0

    assert sum(kind == "spawn" for kind, _ in messages) == 1
    results = [payload for kind, payload in messages if kind == "result"]
    errors = [payload for kind, payload in messages if kind == "error"]
    if same_operation:
        assert len(results) == 2
        assert results[0] == results[1]
        assert errors == []
    else:
        assert len(results) == 1
        assert errors == [{"type": "ConflictError", "code": "campaign_start_reserved"}]


def test_managed_same_key_replay_is_stable_across_service_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "trainer"
    digest = _managed_artifact(root, monkeypatch)
    first = TrainerService(_settings(root))
    second = TrainerService(_settings(root))
    _planned_managed_campaign(first, root, "campaign-managed", digest)
    spawns: list[str] = []
    first._spawn = lambda _name, _action: spawns.append("first")  # type: ignore[method-assign]
    second._spawn = lambda _name, _action: spawns.append("second")  # type: ignore[method-assign]
    headers = {
        "X-Jester-Component-Session": "owner-a",
        "Idempotency-Key": "stable-start",
    }
    payload = {"operationId": "engine.trainer." + "d" * 32}

    with (
        TestClient(
            create_app(first, managed_channel_guard=_managed_guard(), require_managed_session=True)
        ) as first_client,
        TestClient(
            create_app(second, managed_channel_guard=_managed_guard(), require_managed_session=True)
        ) as second_client,
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        responses = list(
            executor.map(
                lambda client: client.post(
                    "/v1/campaigns/campaign-managed/actions/start",
                    headers=headers,
                    json=payload,
                ),
                (first_client, second_client),
            )
        )

    assert [response.status_code for response in responses] == [202, 202]
    assert responses[0].json() == responses[1].json()
    assert responses[0].json()["operationId"] == payload["operationId"]
    assert len(spawns) == 1


@pytest.mark.parametrize(
    ("crash_point", "exit_code"),
    [
        ("before-launch-commit", 70),
        ("after-launch-commit", 71),
        ("after-ack", 72),
    ],
)
def test_crashed_start_is_fenced_for_explicit_reconciliation(
    tmp_path: Path, crash_point: str, exit_code: int
) -> None:
    root = tmp_path / "trainer"
    setup = TrainerService(_settings(root))
    _planned_campaign(setup, root, "campaign-crash")
    if crash_point == "before-launch-commit":
        current = setup.get_campaign("campaign-crash")
        draft = current.model_copy(update={"state": CampaignState.DRAFT})
        setup.store.put_entity(
            "campaign",
            "campaign-crash",
            draft.api_dict(),
            expected_generation=current.generation,
        )
    setup.close()
    operation_id = "engine.trainer." + "e" * 32
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_crash_during_start,
        args=(str(root), "campaign-crash", operation_id, crash_point),
    )
    process.start()
    process.join(timeout=120)
    assert process.exitcode == exit_code

    recovered = TrainerService(_settings(root))
    execution = recovered.campaign_execution(
        "campaign-crash", operation_id=operation_id
    )
    assert execution["operation"]["state"] == "reconciliation-required"
    assert execution["operation"]["error"]["code"] == "campaign_start_outcome_unknown"
    recovered._spawn = lambda *_args, **_kwargs: pytest.fail("replay spawned work")  # type: ignore[method-assign]
    replay = recovered.start_campaign("campaign-crash", operation_id=operation_id)
    assert replay == execution["operation"]
    with pytest.raises(ConflictError, match="reserved start operation") as conflict:
        recovered.start_campaign(
            "campaign-crash", operation_id="engine.trainer." + "f" * 32
        )
    assert conflict.value.code == "campaign_start_reserved"
    recovered.close()
