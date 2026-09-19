from __future__ import annotations

import copy
import json
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import psutil

from .config import Settings
from .datasets import DatasetManager
from .db import StateStore
from .errors import ConflictError, NotFoundError, TrainerError
from .models import (
    Campaign,
    CampaignState,
    DatasetManifest,
    Event,
    Operation,
    OperationState,
    TrainingAttempt,
    TrainingJob,
    TrainingResult,
)
from .parameter_server import ParameterServer
from .rm_client import RMScopedClient
from .util import sha256_json, utc_now
from .workers import WorkerClient, WorkerPool


class TrainerService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_rm()
        self.token = self.settings.initialize()
        self.store = StateStore(self.settings.database)
        artifact_path = os.environ.get("JESTER_RM_ARTIFACT_MANIFEST")
        self.datasets = DatasetManager(artifact_path)
        self.pool = WorkerPool(self.settings.topology)
        self.rm = RMScopedClient(self.settings.topology)
        self._lock = threading.RLock()
        self._idempotency_lock = threading.RLock()
        self._threads: set[threading.Thread] = set()
        self._campaign_controls: dict[str, dict[str, threading.Event]] = {}
        self._recover_campaigns()

    def close(self) -> None:
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout=5)
        self.store.close()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "component": "Jester Trainer",
            "componentNumber": "5/5",
            "version": "0.1.0",
            "apiVersion": "jester.trainer/v1",
            "topologyId": self.settings.topology.get("topologyId"),
            "leaseId": self.settings.lease.get("leaseId"),
            "workers": len(self.pool.clients),
        }

    def pool_status(self) -> dict[str, Any]:
        return {
            "topologyId": self.settings.topology.get("topologyId"),
            "persistent": True,
            "releaseAuthority": "jester-rm",
            "workers": self.pool.status(),
        }

    def worker(self, worker_id: str) -> dict[str, Any]:
        try:
            client = self.pool.clients[worker_id]
        except KeyError as exc:
            raise NotFoundError("worker", worker_id) from exc
        return {
            **self.pool.descriptors[worker_id].api_dict(),
            "health": client.json("GET", "/v1/health"),
            "capabilities": client.json("GET", "/v1/capabilities"),
        }

    def pool_metrics(self) -> dict[str, Any]:
        process = psutil.Process()
        own = {
            "pid": process.pid,
            "cpuPercent": process.cpu_percent(interval=None),
            "ramBytes": process.memory_info().rss,
        }
        rm_metrics: dict[str, Any] | None = None
        rm_error: dict[str, Any] | None = None
        if self.rm.available:
            try:
                rm_metrics = self.rm.metrics()
            except TrainerError as exc:
                rm_error = {"code": exc.code, "detail": str(exc)}
        samples = list((rm_metrics or {}).get("samples", []))
        latest: dict[tuple[str, ...], dict[str, Any]] = {}
        for sample in samples:
            key = (
                str(sample.get("hostId")),
                str(sample.get("instanceId")),
                str(sample.get("leaseId")),
                str(sample.get("pid")),
                str(sample.get("processCreateTime")),
                str(sample.get("role")),
                str(sample.get("resourceId")),
                str(sample.get("metric")),
            )
            previous = latest.get(key)
            if previous is None or str(sample.get("sampledAt", "")) >= str(
                previous.get("sampledAt", "")
            ):
                latest[key] = sample
        accelerator_by_process: dict[tuple[str, ...], int] = {}
        for item in latest.values():
            if item.get("metric") not in {
                "process.accelerator.memory.used",
                "accelerator.memory.allocated",
                "accelerator.memory.reserved",
            }:
                continue
            key = (
                str(item.get("hostId")),
                str(item.get("instanceId")),
                str(item.get("pid")),
                str(item.get("processCreateTime")),
                str(item.get("resourceId")),
            )
            accelerator_by_process[key] = max(
                accelerator_by_process.get(key, 0), int(item.get("value", 0))
            )
        measured = {
            "cpuPercent": sum(
                float(item.get("value", 0))
                for item in latest.values()
                if item.get("metric") == "process.cpu.percent"
            ),
            "ramBytes": sum(
                int(item.get("value", 0))
                for item in latest.values()
                if item.get("metric") == "process.memory.rss"
            ),
            "acceleratorMemoryBytes": sum(accelerator_by_process.values()),
        }
        trainer_instance_id = str(self.settings.lease.get("instanceId", ""))
        trainer_seen = any(
            str(item.get("instanceId", "")) == trainer_instance_id
            and item.get("metric") in {"process.cpu.percent", "process.memory.rss"}
            for item in latest.values()
        )
        if not trainer_seen:
            measured["cpuPercent"] += own["cpuPercent"]
            measured["ramBytes"] += own["ramBytes"]
        return {
            "topologyId": self.settings.topology.get("topologyId"),
            "trainer": own,
            "topologyMeasured": measured,
            "physicalCapacity": (rm_metrics or {}).get("physicalCapacity", []),
            "channels": (rm_metrics or {}).get("channels", []),
            "source": "rm-topology-metrics" if rm_metrics else "trainer-process-only",
            "trainerIncludedByRM": trainer_seen,
            "rmError": rm_error,
        }

    def list_datasets(self) -> list[dict[str, Any]]:
        return self.store.list_entities("dataset")

    def get_dataset(self, dataset_id: str) -> DatasetManifest:
        return DatasetManifest.model_validate(
            self.store.get_entity("dataset", dataset_id)
        )

    def put_dataset(
        self, manifest: DatasetManifest, *, expected_generation: int | None = None
    ) -> dict[str, Any]:
        self.datasets.validate(manifest)
        return self._put_new_or_expected(
            "dataset",
            manifest.metadata.id,
            manifest.api_dict(),
            expected_generation,
        )

    def validate_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self.datasets.validate(self.get_dataset(dataset_id))

    def dataset_shard_plan(
        self,
        dataset_id: str,
        *,
        split: str,
        workers: list[str] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        from .models import DatasetUse

        manifest = self.get_dataset(dataset_id)
        selected = workers or sorted(self.pool.clients)
        use = DatasetUse(datasetId=dataset_id, split=split)
        return self.datasets.shard_plan(
            [(manifest, use)], selected, seed=seed
        )

    def list_campaigns(self) -> list[dict[str, Any]]:
        return self.store.list_entities("campaign")

    def get_campaign(self, campaign_id: str) -> Campaign:
        return Campaign.model_validate(
            self.store.get_entity("campaign", campaign_id)
        )

    def put_campaign(
        self, campaign: Campaign, *, expected_generation: int | None = None
    ) -> dict[str, Any]:
        if campaign.spec.topology_id != self.settings.topology.get("topologyId"):
            raise ConflictError(
                "Campaign topologyId differs from the Trainer RM lease",
                code="campaign_topology_mismatch",
            )
        declared_uses = list(campaign.spec.datasets)
        declared_uses.extend(
            use for facet in campaign.spec.facets for use in facet.datasets
        )
        for use in {
            (item.dataset_id, item.split): item for item in declared_uses
        }.values():
            self.get_dataset(use.dataset_id)
        if self.rm.available:
            campaign = campaign.model_copy(deep=True)
            campaign.spec.checkpoint.output_directory = str(
                self._managed_path(campaign.spec.checkpoint.output_directory)
            )
            self._relative_to_instance_root(
                Path(campaign.spec.checkpoint.output_directory)
            )
        return self._put_new_or_expected(
            "campaign",
            campaign.metadata.id,
            campaign.api_dict(),
            expected_generation,
        )

    def patch_campaign(
        self, campaign_id: str, patch: dict[str, Any], expected_generation: int
    ) -> dict[str, Any]:
        current = self.get_campaign(campaign_id)
        if current.state not in {
            CampaignState.DRAFT,
            CampaignState.PLANNED,
            CampaignState.FAILED,
            CampaignState.CANCELLED,
        }:
            raise ConflictError(
                f"Campaign is immutable while state is {current.state}",
                code="campaign_active",
            )
        forbidden = sorted(set(patch).difference({"metadata", "spec"}))
        if forbidden:
            raise TrainerError(
                "Campaign patch contains controller-owned fields",
                code="immutable_field",
                status=422,
                fields={item: "controller-owned" for item in forbidden},
            )
        merged = _deep_merge(current.api_dict(), patch)
        merged["metadata"]["id"] = campaign_id
        merged["state"] = "draft"
        return self.put_campaign(
            Campaign.model_validate(merged),
            expected_generation=expected_generation,
        )

    def delete_campaign(
        self, campaign_id: str, *, expected_generation: int
    ) -> dict[str, Any]:
        campaign = self.get_campaign(campaign_id)
        if campaign.state in {CampaignState.RUNNING, CampaignState.RECOVERING}:
            raise ConflictError("Cancel the campaign before deletion")
        return self.store.archive_entity(
            "campaign", campaign_id, expected_generation=expected_generation
        )

    def plan_campaign(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.get_campaign(campaign_id)
        dataset_uses = list(campaign.spec.datasets)
        dataset_uses.extend(
            use for facet in campaign.spec.facets for use in facet.datasets
        )
        unique_uses = {
            (use.dataset_id, use.split): use for use in dataset_uses
        }
        manifests = [
            (self.get_dataset(use.dataset_id), use)
            for use in unique_uses.values()
        ]
        for manifest, _ in manifests:
            self.datasets.validate(manifest)
        workers = self.pool.select(
            campaign.spec.cohort.worker_ids,
            campaign.spec.cohort.size,
            retain_pool=campaign.spec.mode == "independent",
        )
        self._validate_cohort(campaign, workers)
        existing = self.campaign_jobs(campaign_id)
        if existing and campaign.state == CampaignState.PLANNED:
            return self._campaign_plan_payload(campaign, existing, workers)
        jobs = _carry_forward_job_attempts(
            self._expand_jobs(campaign),
            existing,
        )
        for job in jobs:
            self.store.put_entity("job", job.job_id, job.api_dict())
        planned = campaign.model_copy(
            update={"state": CampaignState.PLANNED, "updated_at": utc_now()}
        )
        campaign = Campaign.model_validate(
            self.store.put_entity(
                "campaign",
                campaign_id,
                planned.api_dict(),
                expected_generation=campaign.generation,
            )
        )
        self._event(
            "CAMPAIGN_PLANNED",
            campaign_id=campaign_id,
            data={"jobs": len(jobs), "workers": [item.descriptor.worker_id for item in workers]},
        )
        return self._campaign_plan_payload(campaign, jobs, workers)

    def start_campaign(
        self, campaign_id: str, *, operation_id: str | None = None
    ) -> dict[str, Any]:
        campaign = self.get_campaign(campaign_id)
        if operation_id is not None:
            try:
                existing = self.store.get_operation(operation_id)
            except NotFoundError:
                existing = None
            if existing is not None:
                if (
                    existing.get("action") != "campaign.start"
                    or existing.get("targetId") != campaign_id
                ):
                    raise ConflictError(
                        "Operation identifier belongs to another campaign",
                        code="operation_identity_conflict",
                    )
                return existing
        if campaign.state == CampaignState.RUNNING:
            raise ConflictError("Campaign is already running", code="campaign_running")
        if campaign.state != CampaignState.PLANNED:
            self.plan_campaign(campaign_id)
            campaign = self.get_campaign(campaign_id)
        operation = Operation(
            operationId=operation_id or uuid.uuid4().hex,
            action="campaign.start",
            targetId=campaign_id,
            state=OperationState.QUEUED,
        )
        self.store.put_operation(operation.api_dict())
        controls = {
            "pause": threading.Event(),
            "cancel": threading.Event(),
        }
        with self._lock:
            self._campaign_controls[campaign_id] = controls
        self._spawn(
            f"campaign-{campaign_id}",
            lambda: self._run_campaign(operation.operation_id, campaign_id),
        )
        return operation.api_dict()

    def pause_campaign(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.get_campaign(campaign_id)
        if campaign.state != CampaignState.RUNNING:
            raise ConflictError("Only a running campaign can be paused")
        controls = self._controls(campaign_id)
        controls["pause"].set()
        self._set_campaign_state(campaign_id, CampaignState.PAUSED)
        self._event("CAMPAIGN_PAUSED", campaign_id=campaign_id)
        return self.get_campaign(campaign_id).api_dict()

    def resume_campaign(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.get_campaign(campaign_id)
        if campaign.state != CampaignState.PAUSED:
            raise ConflictError("Only a paused campaign can be resumed")
        controls = self._controls(campaign_id)
        controls["pause"].clear()
        self._set_campaign_state(campaign_id, CampaignState.RUNNING)
        self._event("CAMPAIGN_RESUMED", campaign_id=campaign_id)
        return self.get_campaign(campaign_id).api_dict()

    def cancel_campaign(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.get_campaign(campaign_id)
        if campaign.state in {
            CampaignState.SUCCEEDED,
            CampaignState.FAILED,
            CampaignState.CANCELLED,
        }:
            return campaign.api_dict()
        controls = self._controls(campaign_id)
        controls["cancel"].set()
        controls["pause"].clear()
        self._set_campaign_state(campaign_id, CampaignState.CANCELLED)
        self._event("CAMPAIGN_CANCEL_REQUESTED", campaign_id=campaign_id)
        return self.get_campaign(campaign_id).api_dict()

    def campaign_jobs(self, campaign_id: str) -> list[dict[str, Any]]:
        return [
            item
            for item in self.store.list_entities("job")
            if item.get("campaignId") == campaign_id
        ]

    def campaign_events(
        self, campaign_id: str, *, after_sequence: int = 0
    ) -> list[dict[str, Any]]:
        self.get_campaign(campaign_id)
        return self.store.list_events(
            campaign_id=campaign_id, after_sequence=after_sequence
        )

    def campaign_results(self, campaign_id: str) -> list[dict[str, Any]]:
        self.get_campaign(campaign_id)
        return [
            item
            for item in self.store.list_entities("result")
            if item.get("campaignId") == campaign_id
        ]

    def campaign_metrics(self, campaign_id: str) -> dict[str, Any]:
        jobs = self.campaign_jobs(campaign_id)
        results = self.campaign_results(campaign_id)
        return {
            "campaignId": campaign_id,
            "jobs": {
                state: sum(item.get("state") == state for item in jobs)
                for state in ("queued", "preparing", "running", "succeeded", "failed", "cancelled")
            },
            "training": [item.get("metrics", {}) for item in results],
            "resources": self.pool_metrics(),
        }

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self.store.get_entity("job", job_id)

    def retry_job(self, job_id: str) -> dict[str, Any]:
        job = TrainingJob.model_validate(self.get_job(job_id))
        if job.state != "failed":
            raise ConflictError("Only a failed job can be retried")
        campaign = self.get_campaign(job.campaign_id)
        queued = job.model_copy(update={"state": "queued", "error": None})
        self.store.put_entity(
            "job", job_id, queued.api_dict(), expected_generation=job.generation
        )
        if campaign.state in {CampaignState.FAILED, CampaignState.PAUSED}:
            self._set_campaign_state(campaign.metadata.id, CampaignState.PLANNED)
        return self.start_campaign(campaign.metadata.id)

    def get_operation(self, operation_id: str) -> dict[str, Any]:
        return self.store.get_operation(operation_id)

    def campaign_execution(
        self, campaign_id: str, *, operation_id: str | None = None
    ) -> dict[str, Any]:
        campaign = self.get_campaign(campaign_id)
        operations = [
            item for item in self.store.list_operations()
            if item.get("action") == "campaign.start"
            and item.get("targetId") == campaign_id
            and (operation_id is None or item.get("operationId") == operation_id)
        ]
        latest = operations[-1] if operations else None
        return {
            "schemaVersion": "jester-trainer-campaign-execution-v1",
            "campaign": campaign.api_dict(),
            "operation": latest,
            "jobs": self.campaign_jobs(campaign_id),
        }

    def list_results(self) -> list[dict[str, Any]]:
        return self.store.list_entities("result")

    def get_result(self, result_id: str) -> dict[str, Any]:
        return self.store.get_entity("result", result_id)

    def promote_result(self, result_id: str) -> dict[str, Any]:
        result = TrainingResult.model_validate(self.get_result(result_id))
        promoted = result.model_copy(update={"promoted": True})
        return self.store.put_entity(
            "result",
            result_id,
            promoted.api_dict(),
            expected_generation=result.generation,
        )

    def export_result(self, result_id: str, destination: str | Path) -> dict[str, Any]:
        result = TrainingResult.model_validate(self.get_result(result_id))
        source = Path(result.artifact_path).resolve()
        destination = Path(destination).expanduser().resolve()
        if destination.exists():
            raise ConflictError(f"Export destination already exists: {destination}")
        shutil.copytree(source, destination)
        return {"resultId": result_id, "exportedTo": str(destination)}

    def idempotent(
        self,
        scope: str,
        key: str | None,
        payload: Any,
        action: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        if not key:
            return action()
        request_hash = sha256_json(payload)
        with self._idempotency_lock:
            previous = self.store.get_idempotent(scope, key, request_hash)
            if previous is not None:
                return previous
            result = action()
            self.store.put_idempotent(scope, key, request_hash, result)
            return result

    def _run_campaign(self, operation_id: str, campaign_id: str) -> None:
        operation = Operation.model_validate(self.store.get_operation(operation_id))
        operation = operation.model_copy(update={"state": OperationState.RUNNING})
        self.store.put_operation(operation.api_dict())
        try:
            self._set_campaign_state(campaign_id, CampaignState.RUNNING)
            campaign = self.get_campaign(campaign_id)
            jobs = {
                item["jobId"]: TrainingJob.model_validate(item)
                for item in self.campaign_jobs(campaign_id)
                if item.get("state") not in {"succeeded", "cancelled"}
            }
            selected_pool = self.pool.select(
                campaign.spec.cohort.worker_ids,
                campaign.spec.cohort.size,
                retain_pool=campaign.spec.mode == "independent",
            )
            available = {
                worker.descriptor.worker_id: worker for worker in selected_pool
            }
            completed = {
                item["jobId"]
                for item in self.campaign_jobs(campaign_id)
                if item.get("state") == "succeeded"
            }
            while jobs:
                self._control_point(campaign_id)
                ready = [
                    job
                    for job in jobs.values()
                    if set(job.depends_on).issubset(completed)
                ]
                if not ready:
                    raise ConflictError(
                        "No runnable campaign job remains",
                        code="campaign_dag_blocked",
                    )
                assignments: list[tuple[TrainingJob, list[WorkerClient]]] = []
                remaining_workers = dict(available)
                for job in sorted(ready, key=lambda item: item.job_id):
                    cohort_size = (
                        1
                        if campaign.spec.mode == "independent"
                        else campaign.spec.cohort.size
                        or len(selected_pool)
                    )
                    if len(remaining_workers) < cohort_size:
                        continue
                    worker_ids = sorted(remaining_workers)[:cohort_size]
                    cohort = [remaining_workers.pop(item) for item in worker_ids]
                    assignments.append((job, cohort))
                if not assignments:
                    # An explicit full cohort intentionally serializes jobs.
                    first = sorted(ready, key=lambda item: item.job_id)[0]
                    assignments = [(first, list(selected_pool))]
                errors: list[BaseException] = []
                with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
                    futures = {
                        executor.submit(
                            self._run_job_with_retries,
                            campaign,
                            job,
                            cohort,
                        ): job
                        for job, cohort in assignments
                    }
                    for future in as_completed(futures):
                        job = futures[future]
                        try:
                            future.result()
                            completed.add(job.job_id)
                            jobs.pop(job.job_id, None)
                        except BaseException as exc:
                            errors.append(exc)
                if errors:
                    raise errors[0]
            self._set_campaign_state(campaign_id, CampaignState.SUCCEEDED)
            result = {
                "campaignId": campaign_id,
                "results": self.campaign_results(campaign_id),
            }
            completed_operation = operation.model_copy(
                update={
                    "state": OperationState.SUCCEEDED,
                    "updated_at": utc_now(),
                    "result": result,
                }
            )
            self.store.put_operation(completed_operation.api_dict())
            self._event("CAMPAIGN_SUCCEEDED", campaign_id=campaign_id)
        except BaseException as exc:
            current = self.get_campaign(campaign_id)
            if current.state != CampaignState.CANCELLED:
                self._set_campaign_state(campaign_id, CampaignState.FAILED)
            failed = operation.model_copy(
                update={
                    "state": OperationState.FAILED,
                    "updated_at": utc_now(),
                    "error": {
                        "code": getattr(exc, "code", "campaign_failed"),
                        "detail": f"{type(exc).__name__}: {exc}",
                    },
                }
            )
            self.store.put_operation(failed.api_dict())
            self._event(
                "CAMPAIGN_FAILED",
                campaign_id=campaign_id,
                severity="error",
                data=failed.error or {},
            )

    def _run_job_with_retries(
        self,
        campaign: Campaign,
        job: TrainingJob,
        workers: list[WorkerClient],
    ) -> TrainingResult:
        # A manual retry starts a new retry window while preserving the
        # monotonic attempt number for audit/recovery.  Using only
        # maxRetries+1 made an exhausted job execute an empty range and hit
        # the assertion below immediately.
        maximum = job.attempt + campaign.spec.checkpoint.max_retries + 1
        last_error: BaseException | None = None
        for attempt_number in range(job.attempt + 1, maximum + 1):
            self._control_point(campaign.metadata.id)
            try:
                return self._run_job(
                    campaign, job, workers, attempt_number=attempt_number
                )
            except BaseException as exc:
                last_error = exc
                self._event(
                    "JOB_ATTEMPT_FAILED",
                    campaign_id=campaign.metadata.id,
                    job_id=job.job_id,
                    severity="error",
                    data={
                        "attempt": attempt_number,
                        "code": getattr(exc, "code", "job_failed"),
                        "detail": str(exc),
                    },
                )
                if attempt_number >= maximum:
                    current = TrainingJob.model_validate(self.get_job(job.job_id))
                    failed = current.model_copy(
                        update={
                            "state": "failed",
                            "error": {
                                "code": getattr(exc, "code", "job_failed"),
                                "detail": str(exc),
                            },
                        }
                    )
                    self.store.put_entity(
                        "job",
                        job.job_id,
                        failed.api_dict(),
                        expected_generation=current.generation,
                    )
                    raise
                self._set_campaign_state(
                    campaign.metadata.id, CampaignState.RECOVERING
                )
                time.sleep(min(2**attempt_number, 10))
                self._set_campaign_state(
                    campaign.metadata.id, CampaignState.RUNNING
                )
        assert last_error is not None
        raise last_error

    def _run_job(
        self,
        campaign: Campaign,
        job: TrainingJob,
        workers: list[WorkerClient],
        *,
        attempt_number: int,
    ) -> TrainingResult:
        current = TrainingJob.model_validate(self.get_job(job.job_id))
        running = current.model_copy(
            update={
                "state": "running",
                "worker_ids": [item.descriptor.worker_id for item in workers],
                "attempt": attempt_number,
                "error": None,
            }
        )
        current = TrainingJob.model_validate(
            self.store.put_entity(
                "job",
                job.job_id,
                running.api_dict(),
                expected_generation=current.generation,
            )
        )
        attempt_id = f"{job.job_id}-a{attempt_number}"
        attempt = TrainingAttempt(
            attemptId=attempt_id,
            jobId=job.job_id,
            number=attempt_number,
            cohort=[item.descriptor.worker_id for item in workers],
        )
        attempt = TrainingAttempt.model_validate(
            self.store.put_entity("attempt", attempt_id, attempt.api_dict())
        )
        facet = next(
            item for item in campaign.spec.facets if item.id == job.facet_id
        )
        dataset_uses = _facet_dataset_uses(campaign, job.facet_id)
        manifests = [
            (self.get_dataset(use.dataset_id), use)
            for use in dataset_uses
        ]
        shard_plan = self.datasets.shard_plan(
            manifests,
            [item.descriptor.worker_id for item in workers],
            seed=campaign.spec.training.seed,
        )
        candidate = next(
            item for item in facet.candidates if item.id == job.candidate_id
        )
        mi_config = _deep_merge(copy.deepcopy(facet.mi_config), candidate.overrides)
        self._materialize_parent_results(campaign, job, mi_config)
        output = (
            Path(campaign.spec.checkpoint.output_directory).expanduser().resolve()
            / campaign.metadata.id
            / job.job_id
            / f"attempt-{attempt_number:04d}"
        )
        if attempt_number > 1 and not campaign.spec.checkpoint.resume_from:
            latest = self._latest_checkpoint(
                output.parent / f"attempt-{attempt_number - 1:04d}"
            )
            if latest:
                campaign = campaign.model_copy(deep=True)
                campaign.spec.checkpoint.resume_from = str(latest)
        server = ParameterServer(
            campaign=campaign,
            job=current,
            attempt=attempt,
            workers=workers,
            shard_plan=shard_plan,
            mi_config=mi_config,
            output_directory=output,
            persist=self._persist_callback,
            emit=self._event,
            cancel_check=lambda: self._control_point(campaign.metadata.id),
        )
        try:
            result = server.run()
            published_digest = None
            if self.rm.available:
                published = self.rm.publish_artifact(
                    self._relative_to_instance_root(result.artifact_path)
                )
                published_digest = published["artifact"]["digest"]
            succeeded_attempt = attempt.model_copy(
                update={"state": "succeeded", "completed_at": utc_now()}
            )
            self.store.put_entity(
                "attempt",
                attempt_id,
                succeeded_attempt.api_dict(),
                expected_generation=attempt.generation,
            )
            result_id = f"{job.job_id}-result"
            training_result = TrainingResult(
                resultId=result_id,
                campaignId=campaign.metadata.id,
                jobId=job.job_id,
                facetReference=result.facet_reference,
                artifactPath=str(result.artifact_path),
                artifactSha256=result.artifact_sha256,
                artifactDigest=published_digest,
                metrics=result.metrics,
            )
            stored_result = TrainingResult.model_validate(
                self.store.put_entity(
                    "result", result_id, training_result.api_dict()
                )
            )
            current = TrainingJob.model_validate(self.get_job(job.job_id))
            succeeded = current.model_copy(
                update={"state": "succeeded", "result_id": result_id}
            )
            self.store.put_entity(
                "job",
                job.job_id,
                succeeded.api_dict(),
                expected_generation=current.generation,
            )
            self._event(
                "JOB_SUCCEEDED",
                campaign_id=campaign.metadata.id,
                job_id=job.job_id,
                data={"resultId": result_id, "facet": result.facet_reference},
            )
            return stored_result
        except BaseException as exc:
            # Keep attempt audit state truthful even when session creation,
            # model loading, a round, or artifact publication fails.  Leaving
            # exhausted attempts as ``running`` makes recovery and benchmark
            # provenance ambiguous after the enclosing job is already failed.
            try:
                current_attempt = TrainingAttempt.model_validate(
                    self.store.get_entity("attempt", attempt_id)
                )
                if current_attempt.state == "running":
                    failed_attempt = current_attempt.model_copy(
                        update={
                            "state": "failed",
                            "completed_at": utc_now(),
                            "error": {
                                "code": getattr(exc, "code", "job_failed"),
                                "detail": f"{type(exc).__name__}: {exc}",
                            },
                        }
                    )
                    self.store.put_entity(
                        "attempt",
                        attempt_id,
                        failed_attempt.api_dict(),
                        expected_generation=current_attempt.generation,
                    )
            except BaseException:
                # Never replace the actual training failure with a secondary
                # audit-write failure; the job/campaign handler records it.
                pass
            raise
        finally:
            for worker in workers:
                session_id = server.session_ids.get(worker.descriptor.worker_id)
                if session_id:
                    try:
                        worker.json("DELETE", f"/v1/sessions/{session_id}")
                    except TrainerError:
                        pass

    def _expand_jobs(self, campaign: Campaign) -> list[TrainingJob]:
        by_facet: dict[str, list[str]] = {}
        jobs: list[TrainingJob] = []
        references: set[tuple[str, str]] = set()
        for facet in campaign.spec.facets:
            ids: list[str] = []
            for candidate in facet.candidates:
                config = _deep_merge(copy.deepcopy(facet.mi_config), candidate.overrides)
                draft = config.get("facet", {})
                reference = (str(draft.get("id", facet.id)), str(draft.get("version", "")))
                if not reference[1]:
                    raise TrainerError(
                        f"Facet {facet.id}/{candidate.id} miConfig has no facet.version",
                        code="invalid_mi_config",
                        status=422,
                    )
                if reference in references:
                    raise ConflictError(
                        f"Sweep candidates would publish duplicate facet {reference[0]}@{reference[1]}",
                        code="duplicate_sweep_artifact",
                    )
                references.add(reference)
                job_id = f"{campaign.metadata.id}-{facet.id}-{candidate.id}"
                ids.append(job_id)
                jobs.append(
                    TrainingJob(
                        jobId=job_id,
                        campaignId=campaign.metadata.id,
                        facetId=facet.id,
                        candidateId=candidate.id,
                        dependsOn=[],
                    )
                )
            by_facet[facet.id] = ids
        for job in jobs:
            facet = next(item for item in campaign.spec.facets if item.id == job.facet_id)
            dependencies: list[str] = []
            for parent in facet.depends_on:
                parent_jobs = by_facet[parent]
                if len(parent_jobs) > 1 and campaign.spec.selection is None:
                    raise TrainerError(
                        f"Facet {facet.id} depends on sweep {parent}; configure selection.metric",
                        code="sweep_selection_required",
                        status=422,
                    )
                dependencies.extend(parent_jobs)
            job.depends_on = dependencies
        return jobs

    def _campaign_plan_payload(
        self,
        campaign: Campaign,
        jobs: list[TrainingJob] | list[dict[str, Any]],
        workers: list[WorkerClient],
    ) -> dict[str, Any]:
        parameter_estimate = 0
        for facet in campaign.spec.facets:
            slots = facet.mi_config.get("facet", {}).get("module", {}).get("slots", [])
            parameter_estimate += sum(len(slot.get("blocks", [])) for slot in slots)
        bytes_per_parameter = {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
        }[campaign.spec.training.transport_dtype]
        ssh_workers = [
            worker.descriptor.worker_id
            for worker in workers
            if worker.descriptor.transport == "ssh-relay"
        ]
        return {
            "campaign": campaign.api_dict(),
            "jobs": [item.api_dict() if isinstance(item, TrainingJob) else item for item in jobs],
            "pool": [item.descriptor.api_dict() for item in workers],
            "estimator": {
                "insertedBlocks": parameter_estimate,
                "transportBytesPerParameterPerRound": bytes_per_parameter * 2,
                "coordinatorOptimizerMultiplier": 3 if campaign.spec.optimizer.type == "adamw" else 1,
                "sshWorkers": ssh_workers,
                "requiresSshDataPlaneConfirmation": bool(ssh_workers and not campaign.spec.accept_ssh_data_plane),
                "exactAfterWorkerSession": True,
            },
        }

    def _validate_cohort(
        self, campaign: Campaign, workers: list[WorkerClient]
    ) -> None:
        if campaign.spec.mode == "parameter-server":
            backends = {worker.descriptor.effective_backend for worker in workers}
            protocols = {worker.descriptor.protocol for worker in workers}
            if len(backends) != 1:
                raise ConflictError(
                    f"Synchronous cohort backends are heterogeneous: {sorted(backends)}",
                    code="heterogeneous_cohort_backend",
                )
            if protocols != {"http-json+binary"}:
                raise ConflictError(
                    "Synchronous cohorts require http-json+binary channels",
                    code="incompatible_worker_protocol",
                )
        capabilities: list[dict[str, Any]] = []
        for worker in workers:
            capabilities.append(worker.json("GET", "/v1/capabilities"))
        for worker, capability in zip(workers, capabilities):
            effective = worker.descriptor.effective_backend
            match = next(
                (
                    item
                    for item in capability.get("backends", [])
                    if item.get("backend") == effective
                ),
                None,
            )
            if not match or not (
                match.get("workerCompatible") and match.get("autogradCapable")
            ):
                raise ConflictError(
                    f"Worker {worker.descriptor.worker_id} cannot train on effective backend {effective}",
                    code="worker_not_autograd_capable",
                )

    def _materialize_parent_results(
        self, campaign: Campaign, job: TrainingJob, mi_config: dict[str, Any]
    ) -> None:
        if not job.depends_on:
            return
        candidates = [
            TrainingResult.model_validate(self.store.get_entity("result", dependency + "-result"))
            for dependency in job.depends_on
        ]
        grouped: dict[str, list[TrainingResult]] = {}
        for result in candidates:
            parent_job = TrainingJob.model_validate(self.get_job(result.job_id))
            grouped.setdefault(parent_job.facet_id, []).append(result)
        selected: list[TrainingResult] = []
        for results in grouped.values():
            if len(results) == 1:
                selected.append(results[0])
                continue
            selection = campaign.spec.selection
            assert selection is not None
            missing = [
                item.result_id
                for item in results
                if selection.metric not in item.metrics
            ]
            if missing:
                raise TrainerError(
                    f"Selection metric {selection.metric!r} is missing for {missing}",
                    code="selection_metric_missing",
                    status=409,
                )
            selected.append(
                (min if selection.mode == "min" else max)(
                    results, key=lambda item: float(item.metrics[selection.metric])
                )
            )
        registry = self._managed_path(
            str(mi_config.get("registry", "facets"))
        )
        mi_config["registry"] = str(registry)
        for result in selected:
            facet_id, _, version = result.facet_reference.partition("@")
            destination = registry / facet_id / version
            if destination.exists():
                if (destination / "facet.safetensors").is_file():
                    continue
                raise ConflictError(
                    f"Parent registry destination is incomplete: {destination}",
                    code="parent_registry_conflict",
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(result.artifact_path, destination)
        if self.rm.available:
            published = self.rm.publish_artifact(
                self._relative_to_instance_root(registry)
            )
            mi_config["registryArtifactDigest"] = published["artifact"]["digest"]

    @staticmethod
    def _latest_checkpoint(attempt_directory: Path) -> Path | None:
        checkpoints = attempt_directory / "checkpoints"
        if not checkpoints.is_dir():
            return None
        complete = [
            item
            for item in checkpoints.glob("round-*")
            if (item / "checkpoint.json").is_file()
            and (item / "weights.safetensors").is_file()
            and (item / "trainer-state.pt").is_file()
        ]
        return sorted(complete)[-1] if complete else None

    def _control_point(self, campaign_id: str) -> None:
        controls = self._controls(campaign_id)
        if controls["cancel"].is_set():
            raise TrainerError(
                "Campaign was cancelled", code="campaign_cancelled", status=409
            )
        while controls["pause"].is_set():
            if controls["cancel"].wait(0.25):
                raise TrainerError(
                    "Campaign was cancelled", code="campaign_cancelled", status=409
                )

    def _controls(self, campaign_id: str) -> dict[str, threading.Event]:
        with self._lock:
            controls = self._campaign_controls.get(campaign_id)
            if controls is None:
                controls = {
                    "pause": threading.Event(),
                    "cancel": threading.Event(),
                }
                self._campaign_controls[campaign_id] = controls
            return controls

    def _set_campaign_state(
        self, campaign_id: str, state: CampaignState
    ) -> Campaign:
        with self._lock:
            campaign = self.get_campaign(campaign_id)
            updated = campaign.model_copy(
                update={"state": state, "updated_at": utc_now()}
            )
            return Campaign.model_validate(
                self.store.put_entity(
                    "campaign",
                    campaign_id,
                    updated.api_dict(),
                    expected_generation=campaign.generation,
                )
            )

    def _event(
        self,
        event_type: str,
        *,
        campaign_id: str | None = None,
        job_id: str | None = None,
        round_id: str | None = None,
        severity: str = "info",
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.store.add_event(
            Event(
                eventId=uuid.uuid4().hex,
                type=event_type,
                campaignId=campaign_id,
                jobId=job_id,
                roundId=round_id,
                severity=severity,
                data=data or {},
            ).api_dict()
        )

    def _persist_callback(
        self,
        kind: str,
        entity_id: str,
        payload: dict[str, Any],
        expected_generation: int | None,
    ) -> dict[str, Any]:
        return self.store.put_entity(
            kind,
            entity_id,
            payload,
            expected_generation=expected_generation,
        )

    def _relative_to_instance_root(self, path: Path) -> str:
        raw_root = self.settings.lease.get("paths", {}).get("root")
        if not raw_root:
            raise TrainerError(
                "RM lease does not expose an instance root",
                code="rm_instance_root_unavailable",
                status=409,
            )
        root = Path(str(raw_root)).resolve()
        resolved = path.resolve()
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise TrainerError(
                f"Trainer path must remain inside the RM instance volume: {resolved}",
                code="trainer_path_outside_lease",
                status=422,
            ) from exc

    def _managed_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute() or not self.rm.available:
            return path.resolve()
        paths = self.settings.lease.get("paths", {})
        base = Path(str(paths.get("artifacts") or paths.get("root"))).resolve()
        return (base / path).resolve()

    def _put_new_or_expected(
        self,
        kind: str,
        entity_id: str,
        payload: dict[str, Any],
        expected_generation: int | None,
    ) -> dict[str, Any]:
        try:
            return self.store.put_entity(
                kind,
                entity_id,
                payload,
                expected_generation=0
                if expected_generation is None
                else expected_generation,
            )
        except ConflictError as exc:
            if expected_generation is not None or exc.code != "generation_conflict":
                raise
            raise ConflictError(
                f"{kind} {entity_id} already exists; use PATCH with If-Match",
                code=f"{kind}_already_exists",
            ) from exc

    def _spawn(self, name: str, action: Callable[[], None]) -> None:
        def run() -> None:
            try:
                action()
            finally:
                with self._lock:
                    self._threads.discard(threading.current_thread())

        thread = threading.Thread(target=run, name=name, daemon=True)
        with self._lock:
            self._threads.add(thread)
        thread.start()

    def _recover_campaigns(self) -> None:
        for payload in self.store.list_entities("campaign"):
            if payload.get("state") in {"running", "recovering"}:
                campaign = Campaign.model_validate(payload)
                recovering = campaign.model_copy(
                    update={"state": CampaignState.RECOVERING}
                )
                self.store.put_entity(
                    "campaign",
                    campaign.metadata.id,
                    recovering.api_dict(),
                    expected_generation=campaign.generation,
                )


def _facet_dataset_uses(campaign: Campaign, facet_id: str) -> list[Any]:
    facet = next((item for item in campaign.spec.facets if item.id == facet_id), None)
    if facet is None:
        raise TrainerError(
            f"Campaign has no facet node {facet_id}",
            code="facet_not_found",
            status=422,
        )
    return list(facet.datasets or campaign.spec.datasets)


def _carry_forward_job_attempts(
    jobs: list[TrainingJob],
    existing: list[dict[str, Any]],
) -> list[TrainingJob]:
    """Keep attempt numbers monotonic when a failed campaign is replanned.

    Replanning intentionally rebuilds queued jobs from the current campaign DAG,
    but resetting ``attempt`` to zero would reuse attempt/checkpoint identifiers
    that are immutable audit records.  Only the monotonic counter is inherited;
    state, dependencies, errors and result references still come from the newly
    expanded plan.
    """

    attempts = {
        previous.job_id: previous.attempt
        for payload in existing
        for previous in [TrainingJob.model_validate(payload)]
    }
    return [
        job.model_copy(update={"attempt": attempts.get(job.job_id, 0)})
        for job in jobs
    ]


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
