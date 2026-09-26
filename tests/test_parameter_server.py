from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from jester_trainer.config import Settings
from jester_trainer.errors import TrainerError
from jester_trainer.models import TrainingAttempt, TrainingJob
from jester_trainer.parameter_server import ParameterServer
from jester_trainer.service import TrainerService

from .helpers import (
    artifact_weights,
    campaign,
    dataset_manifest,
    tiny_model,
    training_dataset,
    worker_topology,
)


def _run_campaign_observed(
    service: TrainerService, campaign_model: object, timeout: float = 90
) -> dict:
    background_errors: list[dict[str, str]] = []
    previous_hook = threading.excepthook

    def capture(args: threading.ExceptHookArgs) -> None:
        background_errors.append(
            {
                "thread": args.thread.name if args.thread is not None else "unknown",
                "exceptionType": args.exc_type.__name__,
                "detail": str(args.exc_value),
            }
        )
        previous_hook(args)

    threading.excepthook = capture
    try:
        service.put_campaign(campaign_model)  # type: ignore[arg-type]
        campaign_id = campaign_model.metadata.id  # type: ignore[attr-defined]
        service.plan_campaign(campaign_id)
        operation = service.start_campaign(campaign_id)
        deadline = time.monotonic() + timeout
        current = service.get_operation(operation["operationId"])
        while time.monotonic() < deadline and current["state"] not in {
            "succeeded",
            "failed",
            "cancelled",
            "reconciliation-required",
        }:
            time.sleep(0.05)
            current = service.get_operation(operation["operationId"])
        if current["state"] == "succeeded":
            return current
        time.sleep(0.05)
        diagnostics = {
            "operation": current,
            "campaign": service.get_campaign(campaign_id).api_dict(),
            "cancellation": service.campaign_execution(
                campaign_id, operation_id=operation["operationId"]
            )["cancellation"],
            "jobs": service.campaign_jobs(campaign_id),
            "attempts": service.store.list_entities("attempt"),
            "workerSessions": service.store.list_entities("worker-session"),
            "activeStart": service.store.active_campaign_start(campaign_id),
            "backgroundExceptions": background_errors,
            "liveThreads": sorted(thread.name for thread in threading.enumerate()),
        }
        raise AssertionError(json.dumps(diagnostics, sort_keys=True, default=str))
    finally:
        threading.excepthook = previous_hook


def test_two_worker_update_matches_combined_single_worker_batch() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model = root / "base"
        data = root / "train.jsonl"
        tiny_model(model)
        training_dataset(data)
        with worker_topology(root / "workers", count=2) as topology:
            service = TrainerService(Settings.for_test(root / "trainer", topology))
            try:
                service.put_dataset(dataset_manifest(data))
                distributed = _run_campaign_observed(
                    service,
                    campaign(
                        root,
                        model,
                        campaign_id="distributed",
                        version="distributed",
                        workers=["worker-1", "worker-2"],
                    ),
                )
                combined = _run_campaign_observed(
                    service,
                    campaign(
                        root,
                        model,
                        campaign_id="combined",
                        version="combined",
                        workers=["worker-1"],
                    ),
                )
                distributed_state = artifact_weights(distributed)
                combined_state = artifact_weights(combined)
                assert set(distributed_state) == set(combined_state)
                for name in distributed_state:
                    assert torch.allclose(
                        distributed_state[name],
                        combined_state[name],
                        rtol=2e-5,
                        atol=2e-6,
                    ), name
                artifact_path = Path(
                    distributed["result"]["results"][0]["artifactPath"]
                )
                manifest = json.loads(
                    (artifact_path / "facet.mi.json").read_text(encoding="utf-8")
                )
                expected_parameters = sum(
                    int(tensor.numel()) for tensor in distributed_state.values()
                )
                assert manifest["artifact"]["parameterCount"] == expected_parameters
                assert manifest["artifact"]["tensorCount"] == len(distributed_state)
                assert manifest["artifact"]["sizeBytes"] == (
                    artifact_path / "facet.safetensors"
                ).stat().st_size
                assert sum(
                    manifest["artifact"]["dtypeParameterCounts"].values()
                ) == expected_parameters
                rounds = service.store.list_entities("round")
                assert len(rounds) == 2
                assert all(item["state"] == "committed" for item in rounds)
                assert all(len(item["contributions"]) >= 1 for item in rounds)
                assert service.get_campaign("distributed").state == "succeeded"
            finally:
                service.close()


def test_remote_cas_binding_uses_digest_as_portable_base_identity() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        campaign_model = campaign(
            root,
            root / "runtime-model-path",
            campaign_id="portable-identity",
            version="portable",
            workers=["worker-1"],
        )
        digest = "sha256:" + "c" * 64
        server = object.__new__(ParameterServer)
        server.campaign = campaign_model
        server.mi_config = campaign_model.spec.facets[0].mi_config
        server.mi_config["baseModel"]["artifactDigest"] = digest
        server.shard_plan = {"sources": []}
        server.output = root / "output"
        server.artifact_bindings = []

        config = server._worker_training_config()

        assert config["baseModel"]["id"] == "rm-artifact-pending"
        assert config["baseModel"]["canonicalId"] == digest
        assert server.artifact_bindings[0] == {
            "target": "/baseModel/id",
            "digest": digest,
            "logicalPath": None,
        }


def test_session_identity_is_persisted_before_create_request_and_cleanup_is_exact(
    tmp_path: Path,
) -> None:
    stored: dict[str, dict[str, object]] = {}

    def persist(
        kind: str,
        entity_id: str,
        payload: dict[str, object],
        expected_generation: int | None,
    ) -> dict[str, object]:
        assert kind == "worker-session"
        previous = stored.get(entity_id)
        assert expected_generation == (previous or {}).get("generation")
        result = dict(payload)
        result["generation"] = int((previous or {}).get("generation", 0)) + 1
        stored[entity_id] = result
        return result

    class LostCreateReplyWorker:
        descriptor = SimpleNamespace(
            worker_id="worker-1",
            requested_backend="cpu",
            effective_backend="cpu",
            fallback_reason=None,
        )

        def json(
            self, method: str, path: str, payload: dict[str, object]
        ) -> dict[str, object]:
            session_id = "job-a1-worker-1"
            observation = stored[session_id]
            assert method == "POST"
            assert path == "/v1/sessions"
            assert payload["sessionId"] == session_id
            assert observation["sessionId"] == session_id
            assert observation["createState"] == "dispatched"
            raise TrainerError(
                "reply lost", code="worker_unavailable", status=503
            )

    server = object.__new__(ParameterServer)
    server.campaign = campaign(
        tmp_path,
        tmp_path / "model",
        campaign_id="session-observation",
        version="v1",
        workers=["worker-1"],
    )
    server.job = TrainingJob(
        jobId="job", campaignId="session-observation", facetId="one", candidateId="default"
    )
    server.attempt = TrainingAttempt(
        attemptId="job-a1", jobId="job", number=1, cohort=["worker-1"]
    )
    server.workers = [LostCreateReplyWorker()]
    server.persist = persist
    server.artifact_bindings = []
    server.session_ids = {}
    server.session_summaries = {}
    server.session_observations = {}
    server._worker_training_config = lambda: {}  # type: ignore[method-assign]

    with pytest.raises(TrainerError) as lost:
        server._create_sessions()
    assert lost.value.code == "worker_unavailable"
    assert server.session_ids == {"worker-1": "job-a1-worker-1"}

    invalid = server.record_session_cleanup(
        "worker-1",
        response={
            "id": "different-session",
            "deleted": True,
            "artifactsRetained": True,
        },
    )
    assert invalid["cleanupState"] == "outcome-unknown"
    assert invalid["cleanupError"]["code"] == "worker_session_delete_invalid_ack"
    exact = server.record_session_cleanup(
        "worker-1",
        response={
            "id": "job-a1-worker-1",
            "deleted": True,
            "artifactsRetained": True,
        },
    )
    assert exact["cleanupState"] == "acknowledged"
    assert "cleanupError" not in exact


def test_checkpoint_metrics_survive_a_resumed_completed_attempt() -> None:
    source = object.__new__(ParameterServer)
    source.losses = [4.5, 4.0]
    source.round_durations = [2.0, 3.0]
    source.worker_wait_seconds = {"worker-1": 0.25}
    source.total_supervised_tokens = 500

    resumed = object.__new__(ParameterServer)
    resumed.start_round = 2
    resumed.losses = []
    resumed.round_durations = []
    resumed.worker_wait_seconds = {"worker-1": 0.0}
    resumed.total_supervised_tokens = 0
    resumed._restore_checkpoint_metrics({"metrics": source._checkpoint_metrics()})

    assert resumed.losses == [4.5, 4.0]
    assert resumed.round_durations == [2.0, 3.0]
    assert resumed.worker_wait_seconds == {"worker-1": 0.25}
    assert resumed.total_supervised_tokens == 500


def test_global_round_selection_is_identical_across_cohort_sizes() -> None:
    groups = [
        {"semanticCaseId": "a", "indices": [0, 1]},
        {"semanticCaseId": "b", "indices": [2]},
        {"semanticCaseId": "c", "indices": [3]},
        {"semanticCaseId": "d", "indices": [4]},
        {"semanticCaseId": "e", "indices": [5]},
    ]

    def server(worker_ids: list[str], examples_per_worker: int) -> ParameterServer:
        value = object.__new__(ParameterServer)
        value.campaign = SimpleNamespace(
            spec=SimpleNamespace(
                training=SimpleNamespace(
                    examples_per_worker=examples_per_worker,
                    seed=42,
                )
            )
        )
        value.workers = [
            SimpleNamespace(descriptor=SimpleNamespace(worker_id=worker_id))
            for worker_id in worker_ids
        ]
        value.shard_plan = {
            "globalSemanticGroups": groups,
            "roundSelectionPolicy": "cohort-invariant-semantic-groups-v1",
        }
        return value

    one_assignments, one_hash = server(["one"], 4)._round_indices(0)
    two_assignments, two_hash = server(["left", "right"], 2)._round_indices(0)

    assert one_hash == two_hash
    assert sorted(index for values in one_assignments.values() for index in values) == [
        0,
        1,
        2,
        3,
    ]
    assert sorted(index for values in two_assignments.values() for index in values) == [
        0,
        1,
        2,
        3,
    ]
    assert any({0, 1}.issubset(set(values)) for values in two_assignments.values())
