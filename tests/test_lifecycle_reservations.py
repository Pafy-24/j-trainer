from __future__ import annotations

import multiprocessing
import os
import threading
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
from jester_trainer.models import CampaignState
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
    tmp_path: Path,
) -> None:
    root = tmp_path / "trainer"
    first = TrainerService(_settings(root))
    second = TrainerService(_settings(root))
    first.store.bind_campaign_owner(
        "campaign-managed", project_id="project-a", deployment_id="deployment-a"
    )
    _planned_campaign(first, root, "campaign-managed")
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
