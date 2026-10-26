from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
import secrets
import shutil
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from .config import Settings
from .datasets import DatasetManager
from .db import StateStore
from .errors import ConflictError, NotFoundError, TrainerError
from .lifecycle import ControllerLease
from .models import (
    ArtifactPublication,
    Campaign,
    CampaignState,
    DatasetManifest,
    Event,
    ExecutionLineage,
    Operation,
    OperationState,
    PublicationAuthority,
    PublicationRegistration,
    PublicationRegistrationRequest,
    PublicationSourceExpectation,
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
        self._controller_lease = ControllerLease(self.settings.home)
        self.store = StateStore(self.settings.database)
        artifact_path = os.environ.get("JESTER_RM_ARTIFACT_MANIFEST")
        self.datasets = DatasetManager(artifact_path)
        self.pool = WorkerPool(self.settings.topology)
        self.rm = RMScopedClient(self.settings.topology)
        self._lock = threading.RLock()
        self._idempotency_lock = threading.RLock()
        self._threads: set[threading.Thread] = set()
        self._campaign_controls: dict[str, dict[str, threading.Event]] = {}
        self._start_calls = 0
        self._closing = False
        self._closed = False
        self._recover_publications()
        self._recover_campaigns()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._closing or self._start_calls:
                raise TrainerError(
                    "Trainer has a campaign start in progress",
                    code="campaign_start_in_progress",
                    status=409,
                )
            self._closing = True
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout=5)
        active = [thread.name for thread in threads if thread.is_alive()]
        if active:
            with self._lock:
                self._closing = False
            raise TrainerError(
                "Trainer still has active campaign workers: " + ", ".join(active),
                code="campaign_workers_active",
                status=409,
            )
        self.store.close()
        self._controller_lease.close()
        with self._lock:
            self._closed = True

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
        return {
            "topologyId": self.settings.topology.get("topologyId"),
            "trainer": {"status": "external", "authority": "RMA"},
            "topologyMeasured": measured if samples else None,
            "physicalCapacity": (rm_metrics or {}).get("physicalCapacity", []),
            "channels": (rm_metrics or {}).get("channels", []),
            "source": "rm-topology-metrics" if rm_metrics else "unavailable",
            "trainerIncludedByRM": trainer_seen,
            "rmError": rm_error,
        }

    def list_datasets(self) -> list[dict[str, Any]]:
        return self.store.list_entities("dataset")

    def get_dataset(self, dataset_id: str) -> DatasetManifest:
        return DatasetManifest.model_validate(
            self.store.get_entity("dataset", dataset_id)
        )

    @staticmethod
    def managed_manifest_identity(
        manifest: DatasetManifest | dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        """Hash only approved fields, excluding server generation/timestamps."""
        value = manifest.api_dict() if isinstance(manifest, DatasetManifest) else dict(manifest)
        stable = {
            key: value[key]
            for key in (
                "apiVersion", "kind", "metadata", "files",
                "semanticCaseIdField", "seed",
            )
            if key in value
        }
        return stable, sha256_json(stable)

    def register_managed_dataset(
        self, manifest: DatasetManifest, *, project_id: str, deployment_id: str
    ) -> dict[str, Any]:
        for item in manifest.files:
            if item.path is not None or item.artifact_digest is None or not item.logical_path:
                raise TrainerError(
                    "Managed datasets require artifactDigest and logicalPath and forbid local path",
                    code="managed_dataset_artifact_binding_required",
                    status=422,
                )
        self.datasets.validate(manifest)
        _stable, digest = self.managed_manifest_identity(manifest)
        stored = self.store.register_managed_dataset(
            manifest.metadata.id,
            manifest.api_dict(),
            project_id=project_id,
            deployment_id=deployment_id,
            manifest_sha256=digest,
        )
        return {"manifest": stored, "manifestSha256": f"sha256:{digest}"}

    def get_managed_dataset(
        self, dataset_id: str, *, project_id: str, deployment_id: str
    ) -> dict[str, Any]:
        try:
            owned = self.store.managed_dataset(
                dataset_id, project_id=project_id, deployment_id=deployment_id
            )
        except ConflictError as exc:
            raise TrainerError(
                "managed channel session is not authorized",
                code="channel_session_denied",
                status=403,
            ) from exc
        if owned is None:
            raise NotFoundError("dataset", dataset_id)
        manifest, observed = owned
        return {"manifest": manifest, "manifestSha256": f"sha256:{observed}"}

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
        campaign = self._prepare_campaign(campaign)
        return self._put_new_or_expected(
            "campaign",
            campaign.metadata.id,
            campaign.api_dict(),
            expected_generation,
        )

    def put_managed_campaign(
        self, campaign: Campaign, *, project_id: str, deployment_id: str
    ) -> dict[str, Any]:
        campaign = self._prepare_campaign(campaign)
        dataset_ids = self._campaign_dataset_ids(campaign)
        for dataset_id in dataset_ids:
            self.get_managed_dataset(
                dataset_id, project_id=project_id, deployment_id=deployment_id
            )
            self.datasets.validate(self.get_dataset(dataset_id))
        return self.store.put_managed_campaign(
            campaign.metadata.id,
            campaign.api_dict(),
            project_id=project_id,
            deployment_id=deployment_id,
            dataset_ids=dataset_ids,
        )

    def _prepare_campaign(self, campaign: Campaign) -> Campaign:
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
            # Validate the authorized destination without rewriting the public
            # campaign identity with a device-specific session path.
            self._relative_to_instance_root(
                self._managed_path(campaign.spec.checkpoint.output_directory)
            )
        return campaign

    @staticmethod
    def _campaign_dataset_ids(campaign: Campaign) -> list[str]:
        uses = list(campaign.spec.datasets)
        uses.extend(use for facet in campaign.spec.facets for use in facet.datasets)
        return sorted({item.dataset_id for item in uses})

    def _validate_frozen_campaign_datasets(self, campaign: Campaign) -> None:
        owner = self.store.require_managed_campaign_datasets(
            campaign.metadata.id,
            dataset_ids=self._campaign_dataset_ids(campaign),
        )
        if owner is None:
            return
        for dataset_id in self._campaign_dataset_ids(campaign):
            self.get_managed_dataset(dataset_id, project_id=owner[0], deployment_id=owner[1])
            self.datasets.validate(self.get_dataset(dataset_id))

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

    def plan_campaign(
        self,
        campaign_id: str,
        *,
        _start_operation_id: str | None = None,
    ) -> dict[str, Any]:
        self.store.require_campaign_plan_access(
            campaign_id,
            operation_id=_start_operation_id,
            controller_id=self._controller_lease.instance_id,
        )
        campaign = self.get_campaign(campaign_id)
        self._validate_frozen_campaign_datasets(campaign)
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
        planned = campaign.model_copy(
            update={"state": CampaignState.PLANNED, "updated_at": utc_now()}
        )
        campaign = Campaign.model_validate(
            self.store.commit_campaign_plan(
                campaign_id,
                planned.api_dict(),
                [job.api_dict() for job in jobs],
                expected_generation=campaign.generation,
                operation_id=_start_operation_id,
                controller_id=self._controller_lease.instance_id,
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
        with self._lock:
            if self._closing or self._closed:
                raise TrainerError(
                    "Trainer service is closing",
                    code="trainer_service_closing",
                    status=409,
                )
            self._start_calls += 1
        try:
            return self._start_campaign(campaign_id, operation_id=operation_id)
        finally:
            with self._lock:
                self._start_calls -= 1

    def _start_campaign(
        self, campaign_id: str, *, operation_id: str | None = None
    ) -> dict[str, Any]:
        self._validate_frozen_campaign_datasets(self.get_campaign(campaign_id))
        operation = Operation(
            operationId=operation_id or uuid.uuid4().hex,
            action="campaign.start",
            targetId=campaign_id,
            state=OperationState.QUEUED,
        )
        receipt, acquired = self.store.reserve_campaign_start(
            campaign_id,
            operation.api_dict(),
            controller_id=self._controller_lease.instance_id,
        )
        if not acquired:
            return receipt
        try:
            campaign = self.get_campaign(campaign_id)
            if campaign.state != CampaignState.PLANNED:
                self.plan_campaign(
                    campaign_id, _start_operation_id=operation.operation_id
                )
            else:
                self.store.require_campaign_start_reservation(
                    campaign_id,
                    operation.operation_id,
                    controller_id=self._controller_lease.instance_id,
                )
        except BaseException as exc:
            failed = operation.model_copy(
                update={
                    "state": OperationState.FAILED,
                    "updated_at": utc_now(),
                    "error": {
                        "code": getattr(exc, "code", "campaign_start_failed"),
                        "detail": f"{type(exc).__name__}: {exc}",
                    },
                }
            )
            self.store.put_operation(
                failed.api_dict(), controller_id=self._controller_lease.instance_id
            )
            raise
        controls = {
            "pause": threading.Event(),
            "cancel": threading.Event(),
        }
        with self._lock:
            self._campaign_controls[campaign_id] = controls
        self.store.mark_campaign_start_launch_committed(
            campaign_id,
            operation.operation_id,
            controller_id=self._controller_lease.instance_id,
        )
        try:
            self._spawn(
                f"campaign-{campaign_id}",
                lambda: self._run_campaign_guarded(
                    operation.operation_id, campaign_id
                ),
            )
        except BaseException:
            self.store.mark_campaign_start_reconciliation_required(
                campaign_id,
                operation.operation_id,
                controller_id=self._controller_lease.instance_id,
                reason="Controller could not prove whether the committed launch began",
            )
            raise
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

    def cancel_campaign(
        self,
        campaign_id: str,
        *,
        start_operation_id: str | None = None,
        cancel_operation_id: str | None = None,
    ) -> dict[str, Any]:
        if start_operation_id is None and cancel_operation_id is None:
            active = self.store.active_campaign_start(campaign_id)
            if active is None:
                cancelled = self.store.cancel_unstarted_campaign(campaign_id)
                self._event("CAMPAIGN_CANCELLED", campaign_id=campaign_id)
                return cancelled
            if active["controllerId"] != self._controller_lease.instance_id:
                raise ConflictError(
                    "Active campaign belongs to another controller",
                    code="campaign_cancel_start_not_active",
                )
            start_operation_id = str(active["operationId"])
            cancel_operation_id = "admin.cancel." + uuid.uuid4().hex
        if start_operation_id is None or cancel_operation_id is None:
            raise TrainerError(
                "Active campaign cancellation requires both operation identifiers",
                code="cancel_operation_ids_required",
                status=422,
            )
        operation = Operation(
            operationId=cancel_operation_id,
            action="campaign.cancel",
            targetId=campaign_id,
            state=OperationState.QUEUED,
            result={"startOperationId": start_operation_id},
        )
        receipt, acquired = self.store.reserve_campaign_cancel(
            campaign_id,
            start_operation_id,
            operation.api_dict(),
            controller_id=self._controller_lease.instance_id,
        )
        if not acquired:
            return receipt
        running = self.store.mark_campaign_cancel_signal_committed(
            campaign_id,
            cancel_operation_id,
            controller_id=self._controller_lease.instance_id,
        )
        if running.get("state") == "running":
            controls = self._controls(campaign_id)
            controls["cancel"].set()
            controls["pause"].clear()
        self._event(
            "CAMPAIGN_CANCEL_REQUESTED",
            campaign_id=campaign_id,
            data={
                "startOperationId": start_operation_id,
                "cancelOperationId": cancel_operation_id,
            },
        )
        return self.store.get_operation(cancel_operation_id)

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
        self, campaign_id: str, *, operation_id: str | None = None,
        expected_project_id: str | None = None,
        expected_deployment_id: str | None = None,
    ) -> dict[str, Any]:
        snapshot = self.store.execution_snapshot(
            campaign_id, start_operation_id=operation_id
        )
        owner = snapshot["owner"]
        if owner is None and expected_project_id is None:
            cancellation = snapshot["cancellation"]
            if cancellation is not None:
                cancellation = {
                    **cancellation,
                    "operation": snapshot["selectedCancelOperation"],
                }
            return {
                "schemaVersion": "jester-trainer-campaign-execution-v1",
                "campaign": snapshot["campaign"],
                "operation": snapshot["selectedOperation"],
                "cancellation": cancellation,
                "jobs": snapshot["entities"]["job"],
            }
        if expected_project_id is not None and (
            owner is None
            or owner.get("projectId") != expected_project_id
            or owner.get("deploymentId") != expected_deployment_id
        ):
            raise TrainerError(
                "managed campaign execution belongs to another owner",
                code="channel_session_denied",
                status=403,
            )
        operation = snapshot["selectedOperation"]
        selected_operation_id = (
            str(operation.get("operationId")) if operation is not None else operation_id
        )
        cancellation = snapshot["cancellation"]
        if cancellation is not None:
            cancellation = {
                **cancellation,
                "operation": self._execution_operation_projection(
                    snapshot["selectedCancelOperation"]
                ),
            }
        campaign = snapshot["campaign"]
        jobs = [
            self._execution_job_projection(item)
            for item in snapshot["entities"]["job"]
            if item.get("campaignId") == campaign_id
        ]
        audit = self._execution_audit(
            snapshot,
            campaign_id=campaign_id,
            start_operation_id=selected_operation_id,
        )
        return {
            # Preserve the outer resource version consumed by existing
            # lifecycle clients while explicitly versioning the new bounded
            # managed projection and its nested audit contract.
            "schemaVersion": "jester-trainer-campaign-execution-v1",
            "projectionSchemaVersion": "jester-trainer-managed-execution-projection-v2",
            "campaign": {
                "metadata": {"id": campaign_id},
                "state": campaign.get("state"),
                "generation": campaign.get("generation"),
            },
            "owner": owner,
            "operation": self._execution_operation_projection(operation),
            "cancellation": cancellation,
            "jobs": jobs,
            "audit": audit,
        }

    @staticmethod
    def _execution_operation_projection(
        operation: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if operation is None:
            return None
        projected = {
            key: operation[key]
            for key in (
                "operationId", "action", "targetId", "state", "createdAt", "updatedAt"
            )
            if key in operation
        }
        error = operation.get("error")
        if isinstance(error, dict):
            bounded_error: dict[str, Any] = {}
            if isinstance(error.get("code"), str):
                bounded_error["code"] = error["code"]
            if isinstance(error.get("causeCode"), str):
                bounded_error["causeCode"] = error["causeCode"]
            job_errors = error.get("jobErrors")
            if isinstance(job_errors, list):
                bounded_error["jobErrors"] = [
                    {
                        key: item[key]
                        for key in ("jobId", "code")
                        if isinstance(item, dict) and isinstance(item.get(key), str)
                    }
                    for item in job_errors[:2000]
                    if isinstance(item, dict)
                ]
            if bounded_error:
                projected["error"] = bounded_error
        return projected

    @staticmethod
    def _execution_job_projection(job: dict[str, Any]) -> dict[str, Any]:
        return {
            key: job[key]
            for key in (
                "jobId", "campaignId", "facetId", "candidateId", "state", "workerIds",
                "attempt", "resultId", "generation",
            )
            if key in job
        }

    @staticmethod
    def _lineage_matches(
        lineage: Any, *, owner: dict[str, Any], campaign_id: str,
        start_operation_id: str, job_id: str | None = None,
        attempt_id: str | None = None,
    ) -> bool:
        if not isinstance(lineage, dict):
            return False
        expected = {
            "schemaVersion": "jester-trainer-execution-lineage-v1",
            "projectId": owner.get("projectId"),
            "deploymentId": owner.get("deploymentId"),
            "campaignId": campaign_id,
            "startOperationId": start_operation_id,
        }
        if any(lineage.get(key) != value for key, value in expected.items()):
            return False
        if job_id is not None and lineage.get("jobId") != job_id:
            return False
        if attempt_id is not None and lineage.get("attemptId") != attempt_id:
            return False
        return isinstance(lineage.get("jobId"), str) and isinstance(
            lineage.get("attemptId"), str
        )

    def _execution_audit(
        self, snapshot: dict[str, Any], *, campaign_id: str,
        start_operation_id: str | None,
    ) -> dict[str, Any]:
        owner = snapshot["owner"]
        reasons: list[str] = []
        unavailable: list[dict[str, str]] = []
        for kind in snapshot.get("truncated", []):
            reasons.append(f"evidence-truncated:{kind}")
        if owner is None:
            reasons.append("campaign-owner-unavailable")
        if start_operation_id is None:
            reasons.append("start-operation-unavailable")
        entities = snapshot["entities"]
        jobs = {
            str(item["jobId"]): item
            for item in entities["job"]
            if item.get("campaignId") == campaign_id and isinstance(item.get("jobId"), str)
        }
        exact: dict[str, list[dict[str, Any]]] = {
            kind: [] for kind in ("attempt", "round", "result", "publication", "worker-session")
        }
        if owner is not None and start_operation_id is not None:
            for kind in exact:
                for item in entities[kind]:
                    item_id = self._execution_entity_id(kind, item)
                    lineage = item.get("lineage")
                    job_id = (
                        lineage.get("jobId") if isinstance(lineage, dict)
                        else self._historical_entity_job_id(kind, item, entities)
                    )
                    if job_id not in jobs:
                        continue
                    if lineage is None:
                        unavailable.append({"kind": kind, "id": item_id})
                        continue
                    if self._lineage_matches(
                        lineage, owner=owner, campaign_id=campaign_id,
                        start_operation_id=start_operation_id,
                        job_id=str(job_id), attempt_id=str(lineage.get("attemptId", "")),
                    ):
                        exact[kind].append(item)
                    elif lineage.get("campaignId") == campaign_id and lineage.get(
                        "startOperationId"
                    ) == start_operation_id:
                        reasons.append(f"{kind}-lineage-mismatch")

        attempts = [self._attempt_audit_projection(item) for item in exact["attempt"]]
        attempt_by_id = {item["attemptId"]: item for item in attempts}
        rounds = [
            self._round_audit_projection(item) for item in exact["round"]
            if item.get("attemptId") in attempt_by_id
        ]
        sessions = [
            self._session_audit_projection(item) for item in exact["worker-session"]
            if item.get("attemptId") in attempt_by_id
        ]
        results = [self._result_audit_projection(item) for item in exact["result"]]
        publications = [
            self._publication_audit_projection(item) for item in exact["publication"]
        ]
        for item in rounds:
            parent = attempt_by_id.get(item.get("attemptId"))
            if parent is None or item.get("lineage", {}).get("jobId") != parent.get("jobId"):
                reasons.append("round-attempt-lineage-mismatch")
        for item in sessions:
            parent = attempt_by_id.get(item.get("attemptId"))
            if (
                parent is None
                or item.get("jobId") != parent.get("jobId")
                or item.get("lineage", {}).get("jobId") != parent.get("jobId")
            ):
                reasons.append("worker-session-attempt-lineage-mismatch")
        result_by_id = {item.get("resultId"): item for item in results}
        for item in results:
            lineage = item.get("lineage", {})
            parent = attempt_by_id.get(lineage.get("attemptId"))
            if (
                parent is None
                or item.get("jobId") != parent.get("jobId")
                or lineage.get("jobId") != parent.get("jobId")
            ):
                reasons.append("result-attempt-lineage-mismatch")
        for item in publications:
            lineage = item.get("lineage", {})
            parent = attempt_by_id.get(lineage.get("attemptId"))
            result = result_by_id.get(item.get("resultId"))
            if (
                parent is None
                or lineage.get("jobId") != parent.get("jobId")
                or (
                    result is not None
                    and (
                        result.get("lineage", {}).get("attemptId")
                        != lineage.get("attemptId")
                        or result.get("jobId") != parent.get("jobId")
                    )
                )
            ):
                reasons.append("publication-result-lineage-mismatch")
        if any(item.get("state") == "dispatched" for item in publications):
            reasons.append("artifact-publication-outcome-unknown")
        result_ids = {item.get("resultId") for item in results}
        if any(
            item.get("state") == "acknowledged"
            and item.get("resultId") not in result_ids
            for item in publications
        ):
            reasons.append("artifact-publication-result-pending")
        for attempt in attempts:
            attempt_id = attempt["attemptId"]
            cohort = attempt.get("cohort", [])
            if attempt.get("state") not in {"succeeded", "failed", "cancelled"}:
                reasons.append("attempt-not-terminal")
            if (
                not isinstance(cohort, list)
                or not cohort
                or any(not isinstance(item, str) or not item for item in cohort)
                or len(set(cohort)) != len(cohort)
            ):
                reasons.append("attempt-cohort-invalid")
                continue
            observed_sessions = [
                item for item in sessions if item.get("attemptId") == attempt_id
            ]
            if sorted(item.get("workerId") for item in observed_sessions) != sorted(cohort):
                reasons.append("worker-session-cohort-mismatch")
            for item in observed_sessions:
                create_ack = item.get("createAck")
                if not isinstance(create_ack, dict) or create_ack.get("id") != item.get("sessionId"):
                    reasons.append("worker-session-create-ack-missing")
                cleanup_ack = item.get("cleanupAck")
                if item.get("cleanupState") == "outcome-unknown":
                    reasons.append("worker-session-cleanup-outcome-unknown")
                elif not (
                    item.get("cleanupState") == "acknowledged"
                    and isinstance(cleanup_ack, dict)
                    and cleanup_ack.get("id") == item.get("sessionId")
                    and cleanup_ack.get("deleted") is True
                    and cleanup_ack.get("artifactsRetained") is True
                ):
                    reasons.append("worker-session-cleanup-ack-missing")
            if attempt.get("state") == "succeeded":
                committed = [
                    item for item in rounds
                    if item.get("attemptId") == attempt_id and item.get("state") == "committed"
                ]
                if not committed:
                    reasons.append("committed-round-unavailable")
                for item in committed:
                    if sorted(
                        contribution.get("workerId")
                        for contribution in item.get("contributions", [])
                    ) != sorted(cohort):
                        reasons.append("round-contribution-cohort-mismatch")
                    if not isinstance(item.get("aggregateSha256"), str):
                        reasons.append("round-aggregate-digest-missing")
                matching_results = [
                    item for item in results if item["lineage"]["attemptId"] == attempt_id
                ]
                if len(matching_results) != 1:
                    reasons.append("attempt-result-cardinality-mismatch")
                else:
                    result = matching_results[0]
                    matching_publications = [
                        item for item in publications
                        if item.get("resultId") == result.get("resultId")
                        and item["lineage"]["attemptId"] == attempt_id
                    ]
                    if len(matching_publications) != 1:
                        reasons.append("result-publication-cardinality-mismatch")
                    else:
                        publication = matching_publications[0]
                        if publication.get("state") != "acknowledged":
                            reasons.append("artifact-publication-ack-missing")
                        elif (
                            result.get("artifactDigest") != publication.get("artifactDigest")
                            or result.get("artifactSha256") != publication.get("artifactSha256")
                        ):
                            reasons.append("result-publication-digest-mismatch")
        campaign_state = snapshot["campaign"].get("state")
        operation = snapshot["selectedOperation"]
        operation_state = None if operation is None else operation.get("state")
        terminal = campaign_state in {"succeeded", "failed", "cancelled"} and operation_state in {
            "succeeded", "failed", "cancelled"
        }
        if not terminal:
            reasons.append("execution-not-terminal")
        if terminal and not attempts:
            reasons.append("attempt-evidence-unavailable")
        if unavailable:
            reasons.append("historical-lineage-unavailable")
        reasons = sorted(set(reasons))
        audit_state = (
            "unknown"
            if any("unknown" in item for item in reasons)
            else "complete" if not reasons else "incomplete"
        )
        return {
            "schemaVersion": "jester-trainer-execution-audit-v1",
            "identity": {
                "projectId": None if owner is None else owner.get("projectId"),
                "deploymentId": None if owner is None else owner.get("deploymentId"),
                "campaignId": campaign_id,
                "startOperationId": start_operation_id,
            },
            "outcome": {
                "campaignState": campaign_state,
                "startOperationState": operation_state,
            },
            "auditState": audit_state,
            "proofComplete": audit_state == "complete",
            "reasons": reasons,
            "attempts": sorted(attempts, key=lambda item: item["attemptId"]),
            "rounds": sorted(rounds, key=lambda item: (item["attemptId"], item["index"])),
            "results": sorted(results, key=lambda item: item["resultId"]),
            "publications": sorted(publications, key=lambda item: item["publicationId"]),
            "workerSessions": sorted(sessions, key=lambda item: item["sessionId"]),
            "lineageUnavailable": sorted(unavailable, key=lambda item: (item["kind"], item["id"])),
            "counts": {
                "attempts": len(attempts), "rounds": len(rounds),
                "results": len(results), "publications": len(publications),
                "workerSessions": len(sessions),
            },
        }

    @staticmethod
    def _execution_entity_id(kind: str, item: dict[str, Any]) -> str:
        key = {
            "attempt": "attemptId", "round": "roundId", "result": "resultId",
            "publication": "publicationId", "worker-session": "sessionId",
        }[kind]
        value = item.get(key, item.get("id", "unavailable"))
        return str(value)

    @staticmethod
    def _historical_entity_job_id(
        kind: str, item: dict[str, Any], entities: dict[str, list[dict[str, Any]]]
    ) -> str | None:
        if isinstance(item.get("jobId"), str):
            return str(item["jobId"])
        if kind == "round" and isinstance(item.get("attemptId"), str):
            attempt = next((
                value for value in entities["attempt"]
                if value.get("attemptId") == item["attemptId"]
            ), None)
            return None if attempt is None else attempt.get("jobId")
        if kind == "publication" and isinstance(item.get("resultId"), str):
            result = next((
                value for value in entities["result"]
                if value.get("resultId") == item["resultId"]
            ), None)
            return None if result is None else result.get("jobId")
        return None

    @staticmethod
    def _attempt_audit_projection(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item[key]
            for key in ("attemptId", "jobId", "number", "cohort", "state", "lineage")
            if key in item
        }

    @staticmethod
    def _round_audit_projection(item: dict[str, Any]) -> dict[str, Any]:
        contributions = []
        for value in item.get("contributions", []):
            if isinstance(value, dict):
                contributions.append({
                    key: value[key]
                    for key in ("workerId", "gradientSha256", "supervisedTokens", "examples")
                    if key in value
                })
        return {
            **{
                key: item[key]
                for key in (
                    "roundId", "attemptId", "index", "state", "weightVersion",
                    "selectionSha256", "aggregateSha256", "lineage",
                )
                if key in item
            },
            "contributions": contributions,
        }

    @staticmethod
    def _session_audit_projection(item: dict[str, Any]) -> dict[str, Any]:
        projected = {
            key: item[key]
            for key in (
                "id", "campaignId", "jobId", "attemptId", "workerId", "sessionId",
                "createState", "cleanupState", "lineage",
            )
            if key in item
        }
        for source, keys in (
            ("createAck", ("id", "state", "weightVersion", "weightSha256")),
            ("cleanupAck", ("id", "deleted", "artifactsRetained")),
        ):
            ack = item.get(source)
            if isinstance(ack, dict):
                projected[source] = {key: ack[key] for key in keys if key in ack}
        return projected

    @staticmethod
    def _result_audit_projection(item: dict[str, Any]) -> dict[str, Any]:
        facet = item.get("facetReference")
        return {
            **{
                key: item[key]
                for key in (
                    "resultId", "campaignId", "jobId", "artifactSha256",
                    "artifactDigest", "lineage",
                )
                if key in item
            },
            "facetReferenceSha256": (
                hashlib.sha256(facet.encode("utf-8")).hexdigest()
                if isinstance(facet, str) else None
            ),
        }

    @staticmethod
    def _publication_audit_projection(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item[key]
            for key in (
                "publicationId", "resultId", "artifactDigest", "artifactSha256",
                "state", "descriptorKind", "descriptorSize", "lineage",
            )
            if key in item
        }

    def publication_readback(
        self,
        campaign_id: str,
        result_id: str,
        publication_id: str,
        *,
        operation_id: str,
        expected_project_id: str | None,
        expected_deployment_id: str | None,
    ) -> dict[str, Any]:
        """Return one managed, acknowledged result/publication pair.

        This is intentionally separate from the execution audit: the audit
        retains only a facet-reference hash, while this selected readback may
        release the exact reference after the managed ownership checks pass.
        """
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                campaign_id, result_id, publication_id, operation_id,
                expected_project_id, expected_deployment_id,
            )
        ):
            raise TrainerError(
                "managed publication readback identity is incomplete",
                code="publication_readback_denied",
                status=409,
            )
        execution = self.campaign_execution(
            campaign_id,
            operation_id=operation_id,
            expected_project_id=expected_project_id,
            expected_deployment_id=expected_deployment_id,
        )
        audit = execution.get("audit")
        if not isinstance(audit, dict):
            raise TrainerError(
                "managed publication readback audit is unavailable",
                code="publication_readback_denied",
                status=409,
            )
        identity = audit.get("identity")
        if (
            audit.get("proofComplete") is not True
            or audit.get("auditState") != "complete"
            or audit.get("outcome") != {
                "campaignState": "succeeded", "startOperationState": "succeeded"
            }
            or identity != {
                "projectId": expected_project_id,
                "deploymentId": expected_deployment_id,
                "campaignId": campaign_id,
                "startOperationId": operation_id,
            }
        ):
            raise TrainerError(
                "managed publication readback is not a completed success",
                code="publication_readback_denied",
                status=409,
            )
        results = audit.get("results")
        publications = audit.get("publications")
        attempts = audit.get("attempts")
        if not all(isinstance(value, list) for value in (results, publications, attempts)):
            raise TrainerError(
                "managed publication readback audit is malformed",
                code="publication_readback_denied",
                status=409,
            )
        selected_results = [
            item for item in results
            if isinstance(item, dict) and item.get("resultId") == result_id
        ]
        selected_publications = [
            item for item in publications
            if isinstance(item, dict) and item.get("publicationId") == publication_id
        ]
        if len(selected_results) != 1 or len(selected_publications) != 1:
            raise TrainerError(
                "managed publication readback pair is missing or ambiguous",
                code="publication_readback_denied",
                status=409,
            )
        result_projection = selected_results[0]
        publication_projection = selected_publications[0]
        lineage = result_projection.get("lineage")
        if not isinstance(lineage, dict):
            raise TrainerError(
                "managed publication readback lineage is malformed",
                code="publication_readback_denied",
                status=409,
            )
        try:
            normalized_lineage = ExecutionLineage.model_validate(lineage)
        except ValueError as exc:
            raise TrainerError(
                "managed publication readback lineage is malformed",
                code="publication_readback_denied",
                status=409,
            ) from exc
        if (
            normalized_lineage.api_dict() != lineage
            or normalized_lineage.project_id != expected_project_id
            or normalized_lineage.deployment_id != expected_deployment_id
            or normalized_lineage.campaign_id != campaign_id
            or normalized_lineage.start_operation_id != operation_id
        ):
            raise TrainerError(
                "managed publication readback lineage authority differs",
                code="publication_readback_denied",
                status=409,
            )
        matching_attempts = [
            item for item in attempts
            if isinstance(item, dict)
            and item.get("attemptId") == normalized_lineage.attempt_id
            and item.get("jobId") == normalized_lineage.job_id
            and item.get("lineage") == lineage
            and item.get("state") == "succeeded"
        ]
        if (
            len(matching_attempts) != 1
            or publication_projection.get("resultId") != result_id
            or publication_projection.get("state") != "acknowledged"
            or publication_projection.get("lineage") != lineage
        ):
            raise TrainerError(
                "managed publication readback pair is not consumable",
                code="publication_readback_denied",
                status=409,
            )
        try:
            result = TrainingResult.model_validate(self.get_result(result_id))
            publication = ArtifactPublication.model_validate(
                self.store.get_entity("publication", publication_id)
            )
        except (NotFoundError, ValueError) as exc:
            raise TrainerError(
                "managed publication readback pair is unavailable",
                code="publication_readback_denied",
                status=409,
            ) from exc
        facet_reference = result.facet_reference
        if (
            result.result_id != result_id
            or result.campaign_id != campaign_id
            or result.lineage is None
            or result.lineage.api_dict() != lineage
            or result.job_id != normalized_lineage.job_id
            or publication.publication_id != publication_id
            or publication.result_id != result_id
            or publication.lineage.api_dict() != lineage
            or publication.state != "acknowledged"
            or not self._canonical_sha256(result.artifact_sha256)
            or not self._canonical_digest(result.artifact_digest)
            or not self._canonical_sha256(publication.artifact_sha256)
            or not self._canonical_digest(publication.artifact_digest)
            or result.artifact_sha256 != publication.artifact_sha256
            or result.artifact_digest != publication.artifact_digest
            or result_projection.get("artifactSha256") != result.artifact_sha256
            or result_projection.get("artifactDigest") != result.artifact_digest
            or publication_projection.get("artifactSha256") != publication.artifact_sha256
            or publication_projection.get("artifactDigest") != publication.artifact_digest
            or result_projection.get("facetReferenceSha256")
            != hashlib.sha256(facet_reference.encode("utf-8")).hexdigest()
        ):
            raise TrainerError(
                "managed publication readback authority differs",
                code="publication_readback_denied",
                status=409,
            )
        return {
            "schemaVersion": "jester-trainer-publication-readback-v1",
            "identity": identity,
            "result": {
                "resultId": result.result_id,
                "jobId": result.job_id,
                "lineage": lineage,
                "facetReference": facet_reference,
                "artifactSha256": result.artifact_sha256,
                "artifactDigest": result.artifact_digest,
            },
            "publication": {
                "publicationId": publication.publication_id,
                "resultId": publication.result_id,
                "state": publication.state,
                "lineage": publication.lineage.api_dict(),
                "artifactSha256": publication.artifact_sha256,
                "artifactDigest": publication.artifact_digest,
                "descriptorKind": publication.descriptor_kind,
                "descriptorSize": publication.descriptor_size,
            },
        }

    @staticmethod
    def _canonical_sha256(value: Any) -> bool:
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    @staticmethod
    def _canonical_digest(value: Any) -> bool:
        return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None

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
            return self.store.put_idempotent(scope, key, request_hash, result)

    def _run_campaign(self, operation_id: str, campaign_id: str) -> None:
        operation = Operation.model_validate(self.store.get_operation(operation_id))
        try:
            acquired_running = self.store.mark_campaign_start_running(
                campaign_id,
                operation_id,
                controller_id=self._controller_lease.instance_id,
            )
            if not acquired_running:
                if self.get_campaign(campaign_id).state == CampaignState.CANCELLING:
                    raise TrainerError(
                        "Campaign was cancelled", code="campaign_cancelled", status=409
                    )
                return
            self._control_point(campaign_id)
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
                errors: list[tuple[str, BaseException]] = []
                with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
                    runner = self._run_job_with_retries
                    runner_parameters = inspect.signature(runner).parameters
                    futures = {
                        executor.submit(
                            runner,
                            campaign,
                            job,
                            cohort,
                            **(
                                {"start_operation_id": operation_id}
                                if "start_operation_id" in runner_parameters else {}
                            ),
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
                            errors.append((job.job_id, exc))
                if errors:
                    raise self._campaign_batch_error(errors)
            self._control_point(campaign_id)
            result = {
                "campaignId": campaign_id,
                "results": self.campaign_results(campaign_id),
            }
            if not self.store.settle_campaign_succeeded(
                campaign_id,
                operation_id,
                result,
                controller_id=self._controller_lease.instance_id,
            ):
                self._control_point(campaign_id)
                raise TrainerError(
                    "Campaign cancellation won the terminal race",
                    code="campaign_cancelled",
                    status=409,
                )
            self._event("CAMPAIGN_SUCCEEDED", campaign_id=campaign_id)
        except BaseException as exc:
            if getattr(exc, "code", None) in {
                "remote_session_outcome_unknown",
                "artifact_publication_outcome_unknown",
            }:
                reservation = self.store.campaign_cancel_reservation(campaign_id)
                reason = f"{type(exc).__name__}: {exc}"
                details = {
                    "jobErrors": list(getattr(exc, "fields", {}).get("jobErrors", []))
                }
                self.store.mark_campaign_start_reconciliation_required(
                    campaign_id,
                    operation_id,
                    controller_id=self._controller_lease.instance_id,
                    reason=reason,
                    details=details,
                )
                self._event(
                    "CAMPAIGN_RECONCILIATION_REQUIRED",
                    campaign_id=campaign_id,
                    severity="error",
                    data={
                        "code": getattr(exc, "code", "remote_session_outcome_unknown"),
                        "detail": reason,
                        **details,
                    },
                )
                return
            if getattr(exc, "code", None) == "campaign_cancelled":
                if self.store.settle_campaign_cancelled(
                    campaign_id,
                    operation_id,
                    controller_id=self._controller_lease.instance_id,
                ):
                    self._event("CAMPAIGN_CANCELLED", campaign_id=campaign_id)
                    return
            error = {
                "code": getattr(exc, "code", "campaign_failed"),
                "detail": f"{type(exc).__name__}: {exc}",
            }
            job_errors = getattr(exc, "fields", {}).get("jobErrors")
            if job_errors:
                error["jobErrors"] = list(job_errors)
            applied = self.store.settle_campaign_failed(
                campaign_id,
                operation_id,
                error,
                controller_id=self._controller_lease.instance_id,
            )
            if not applied:
                return
            self._event(
                "CAMPAIGN_FAILED",
                campaign_id=campaign_id,
                severity="error",
                data=error,
            )

    def _run_campaign_guarded(self, operation_id: str, campaign_id: str) -> None:
        """Fence an exception that escapes the normal settlement path."""

        try:
            self._run_campaign(operation_id, campaign_id)
        except BaseException as exc:
            self.store.mark_campaign_start_reconciliation_required(
                campaign_id,
                operation_id,
                controller_id=self._controller_lease.instance_id,
                reason=(
                    "Campaign runner exited before it could prove a terminal "
                    f"receipt: {type(exc).__name__}: {exc}"
                ),
                details={
                    "causeCode": "campaign_runner_outcome_unknown",
                    "exceptionType": type(exc).__name__,
                },
            )
            raise

    @staticmethod
    def _campaign_batch_error(
        errors: list[tuple[str, BaseException]],
    ) -> TrainerError:
        def precedence(item: tuple[str, BaseException]) -> tuple[int, str]:
            code = getattr(item[1], "code", "campaign_failed")
            if code in {
                "remote_session_outcome_unknown",
                "artifact_publication_outcome_unknown",
            }:
                rank = 0
            elif code == "campaign_cancelled":
                rank = 2
            else:
                rank = 1
            return rank, item[0]

        ordered = sorted(errors, key=precedence)
        _job_id, primary = ordered[0]
        job_errors = [
            {
                "jobId": job_id,
                "code": getattr(error, "code", "campaign_failed"),
                "detail": f"{type(error).__name__}: {error}",
            }
            for job_id, error in sorted(errors, key=lambda item: item[0])
        ]
        return TrainerError(
            str(primary),
            code=getattr(primary, "code", "campaign_failed"),
            status=int(getattr(primary, "status", 500)),
            fields={"jobErrors": job_errors},
        )

    def _run_job_with_retries(
        self,
        campaign: Campaign,
        job: TrainingJob,
        workers: list[WorkerClient],
        *,
        start_operation_id: str | None = None,
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
                arguments: dict[str, Any] = {"attempt_number": attempt_number}
                if start_operation_id is not None:
                    arguments["start_operation_id"] = start_operation_id
                return self._run_job(campaign, job, workers, **arguments)
            except BaseException as exc:
                last_error = exc
                if getattr(exc, "code", None) in {
                    "campaign_cancelled",
                    "remote_session_outcome_unknown",
                    "artifact_publication_outcome_unknown",
                }:
                    raise
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
        start_operation_id: str | None = None,
    ) -> TrainingResult:
        owner = self.store.campaign_owner(campaign.metadata.id)
        if owner is not None and not start_operation_id:
            raise TrainerError(
                "Managed training requires its start operation binding",
                code="execution_lineage_missing",
                status=409,
            )
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
        lineage = None if owner is None else ExecutionLineage(
            projectId=owner[0],
            deploymentId=owner[1],
            campaignId=campaign.metadata.id,
            startOperationId=str(start_operation_id),
            jobId=job.job_id,
            attemptId=attempt_id,
        )
        attempt = TrainingAttempt(
            attemptId=attempt_id,
            jobId=job.job_id,
            number=attempt_number,
            cohort=[item.descriptor.worker_id for item in workers],
            lineage=lineage,
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
            self._managed_path(campaign.spec.checkpoint.output_directory)
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
        result: Any | None = None
        execution_error: BaseException | None = None
        remote_execution_completed = False
        try:
            result = server.run()
            remote_execution_completed = True
            # A managed retry/start retains each immutable result under the
            # attempt that produced it.  Unmanaged campaigns preserve their
            # historical lookup identity.
            result_id = (
                f"{attempt_id}-result" if lineage is not None
                else f"{job.job_id}-result"
            )
            training_result = TrainingResult(
                resultId=result_id,
                campaignId=campaign.metadata.id,
                jobId=job.job_id,
                lineage=lineage,
                facetReference=result.facet_reference,
                artifactPath=str(result.artifact_path),
                artifactSha256=result.artifact_sha256,
                artifactDigest=None,
                metrics=result.metrics,
            )
            if self.rm.available:
                if lineage is None:
                    raise TrainerError(
                        "Managed artifact publication has no durable execution lineage",
                        code="execution_lineage_missing",
                        status=409,
                    )
                weights_path = Path(result.artifact_path) / "facet.safetensors"
                observed_sha256 = self._sha256_file(weights_path)
                if observed_sha256 != result.artifact_sha256:
                    raise TrainerError(
                        "Trainer weights changed before publication reservation",
                        code="artifact_publication_source_changed",
                        status=409,
                    )
                relative_path = self._relative_to_instance_root(Path(result.artifact_path))
                publication_id = f"{attempt_id}-publication"
                request = PublicationRegistrationRequest(
                    relativePath=relative_path,
                    publicationId=publication_id,
                    resultId=result_id,
                    lineage=lineage,
                    artifactSha256=result.artifact_sha256,
                )
                publication = ArtifactPublication(
                    publicationId=publication_id,
                    resultId=result_id,
                    lineage=lineage,
                    artifactSha256=result.artifact_sha256,
                )
                source_expectation = self._publication_source_expectation()
                registration = PublicationRegistration(
                    registrationId=publication_id,
                    publicationId=publication_id,
                    resultId=result_id,
                    lineage=lineage,
                    relativePath=relative_path,
                    artifactSha256=result.artifact_sha256,
                    idempotencyKey="p" + secrets.token_urlsafe(47),
                    requestSha256=sha256_json({
                        "topologyId": self.rm.topology_id,
                        **request.api_dict(exclude_none=False),
                    }),
                    request=request,
                    sourceExpectation=source_expectation,
                    startControllerId=self._controller_lease.instance_id,
                )
                _stored_result, _stored_publication, stored_registration = (
                    self.store.reserve_publication_draft(
                        result=training_result.api_dict(),
                        publication=publication.api_dict(),
                        registration=registration.api_dict(exclude_none=False),
                    )
                )
                acknowledged = self._register_reserved_publication(
                    PublicationRegistration.model_validate(stored_registration)
                )
                if not self.store.settle_acknowledged_publication(
                    acknowledged.registration_id
                ):
                    raise TrainerError(
                        "Campaign cancellation won after publication intent",
                        code="campaign_cancelled",
                        status=409,
                    )
                stored_result = TrainingResult.model_validate(
                    self.store.get_entity("result", result_id)
                )
            else:
                succeeded_attempt = attempt.model_copy(
                    update={"state": "succeeded", "completed_at": utc_now()}
                )
                self.store.put_entity(
                    "attempt", attempt_id, succeeded_attempt.api_dict(),
                    expected_generation=attempt.generation,
                )
                stored_result = TrainingResult.model_validate(
                    self.store.put_entity("result", result_id, training_result.api_dict())
                )
                current = TrainingJob.model_validate(self.get_job(job.job_id))
                succeeded = current.model_copy(
                    update={"state": "succeeded", "result_id": result_id}
                )
                self.store.put_entity(
                    "job", job.job_id, succeeded.api_dict(),
                    expected_generation=current.generation,
                )
            self._event(
                "JOB_SUCCEEDED",
                campaign_id=campaign.metadata.id,
                job_id=job.job_id,
                data={"resultId": result_id, "facet": result.facet_reference},
            )
            result = stored_result
        except BaseException as exc:
            execution_error = exc
        cleanup_unknown: list[dict[str, Any]] = []
        for worker in workers:
            worker_id = worker.descriptor.worker_id
            session_id = server.session_ids.get(worker_id)
            if not session_id:
                continue
            try:
                response = worker.json("DELETE", f"/v1/sessions/{session_id}")
                observation = server.record_session_cleanup(
                    worker_id, response=response
                )
            except BaseException as cleanup_error:
                try:
                    observation = server.record_session_cleanup(
                        worker_id, error=cleanup_error
                    )
                except BaseException as persist_error:
                    observation = {
                        "workerId": worker_id,
                        "sessionId": session_id,
                        "cleanupState": "outcome-unknown",
                        "cleanupError": {
                            "code": "worker_session_observation_write_failed",
                            "detail": f"{type(persist_error).__name__}: {persist_error}",
                        },
                    }
            if observation.get("cleanupState") != "acknowledged":
                cleanup_unknown.append(observation)

        current_attempt = TrainingAttempt.model_validate(
            self.store.get_entity("attempt", attempt_id)
        )
        safe_frontier = (
            (
                getattr(execution_error, "code", None) == "campaign_cancelled"
                and bool(getattr(execution_error, "fields", {}).get("safeBoundary"))
            )
            or remote_execution_completed
        )
        if cleanup_unknown and not safe_frontier:
            if current_attempt.state == "running":
                cause = None
                if execution_error is not None:
                    cause = {
                        "code": getattr(execution_error, "code", "job_failed"),
                        "detail": f"{type(execution_error).__name__}: {execution_error}",
                    }
                uncertain_attempt = current_attempt.model_copy(
                    update={
                        "state": "reconciliation-required",
                        "error": {
                            "code": "remote_session_outcome_unknown",
                            "detail": "One or more worker sessions lack an exact deletion acknowledgement",
                            "cause": cause,
                            "sessions": [
                                {
                                    "workerId": item.get("workerId"),
                                    "sessionId": item.get("sessionId"),
                                    "cleanupError": item.get("cleanupError"),
                                }
                                for item in cleanup_unknown
                            ],
                        },
                    }
                )
                self.store.put_entity(
                    "attempt",
                    attempt_id,
                    uncertain_attempt.api_dict(),
                    expected_generation=current_attempt.generation,
                )
            raise TrainerError(
                "Worker session termination could not be proven",
                code="remote_session_outcome_unknown",
                status=503,
                fields={
                    "sessions": [item.get("sessionId") for item in cleanup_unknown]
                },
            )

        if execution_error is not None:
            if current_attempt.state == "running":
                cancelled = getattr(execution_error, "code", None) == "campaign_cancelled"
                uncertain_publication = (
                    getattr(execution_error, "code", None)
                    == "artifact_publication_outcome_unknown"
                )
                completed_attempt = current_attempt.model_copy(
                    update={
                        "state": (
                            "cancelled" if cancelled else
                            "reconciliation-required" if uncertain_publication else
                            "failed"
                        ),
                        "completed_at": None if uncertain_publication else utc_now(),
                        "error": None
                        if cancelled
                        else {
                            "code": getattr(execution_error, "code", "job_failed"),
                            "detail": f"{type(execution_error).__name__}: {execution_error}",
                        },
                    }
                )
                self.store.put_entity(
                    "attempt",
                    attempt_id,
                    completed_attempt.api_dict(),
                    expected_generation=current_attempt.generation,
                )
            raise execution_error

        if not isinstance(result, TrainingResult):
            raise TrainerError(
                "Training attempt completed without a result",
                code="training_result_missing",
                status=500,
            )
        return result

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
        if (
            controls["cancel"].is_set()
            or self.get_campaign(campaign_id).state
            in {CampaignState.CANCELLING, CampaignState.CANCELLED}
        ):
            raise TrainerError(
                "Campaign was cancelled",
                code="campaign_cancelled",
                status=409,
                fields={"safeBoundary": True},
            )
        while controls["pause"].is_set():
            if controls["cancel"].wait(0.25):
                raise TrainerError(
                    "Campaign was cancelled",
                    code="campaign_cancelled",
                    status=409,
                    fields={"safeBoundary": True},
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

    @staticmethod
    def _sha256_file(path: Path) -> str:
        if not path.is_file():
            raise TrainerError(
                "Trainer publication requires facet.safetensors",
                code="artifact_publication_source_missing",
                status=409,
            )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _publication_source_expectation(self) -> PublicationSourceExpectation:
        lease = self.settings.lease
        instance_id = lease.get("instanceId")
        lease_id = lease.get("leaseId")
        host_id = lease.get("hostId")
        if not all(isinstance(value, str) and value for value in (instance_id, lease_id, host_id)):
            raise TrainerError(
                "Trainer lease omits publication source identity",
                code="publication_source_identity_missing",
                status=409,
            )
        generation = lease.get("instanceGeneration")
        if generation is not None and (type(generation) is not int or generation < 1):
            raise TrainerError(
                "Trainer lease has an invalid instance generation",
                code="publication_source_identity_invalid",
                status=409,
            )
        host_epoch = lease.get("hostEpoch")
        if host_epoch is not None and (not isinstance(host_epoch, str) or not host_epoch):
            raise TrainerError(
                "Trainer lease has an invalid host epoch",
                code="publication_source_identity_invalid",
                status=409,
            )
        return PublicationSourceExpectation(
            trainerInstanceId=instance_id,
            trainerInstanceGeneration=generation,
            trainerLeaseId=lease_id,
            hostId=host_id,
            hostEpoch=host_epoch,
        )

    def _validate_publication_authority(
        self,
        registration: PublicationRegistration,
        payload: dict[str, Any],
    ) -> PublicationAuthority:
        try:
            authority = PublicationAuthority.model_validate(payload)
        except ValidationError as exc:
            raise TrainerError(
                "RM publication authority is malformed",
                code="artifact_publication_invalid_ack",
                status=502,
            ) from exc
        lineage = registration.lineage.api_dict(exclude_none=False)
        expected_source = registration.source_expectation
        identity = authority.identity
        source = authority.source
        if (
            identity.project_id != registration.lineage.project_id
            or identity.deployment_id != registration.lineage.deployment_id
            or identity.topology_id != self.rm.topology_id
            or authority.result.result_id != registration.result_id
            or authority.result.lineage.api_dict(exclude_none=False) != lineage
            or authority.publication.publication_id != registration.publication_id
            or authority.publication.artifact_sha256 != registration.artifact_sha256
            or source.trainer_instance_id != expected_source.trainer_instance_id
            or source.trainer_lease_id != expected_source.trainer_lease_id
            or source.host_id != expected_source.host_id
            or (
                expected_source.trainer_instance_generation is not None
                and source.trainer_instance_generation
                != expected_source.trainer_instance_generation
            )
            or (
                expected_source.host_epoch is not None
                and source.host_epoch != expected_source.host_epoch
            )
        ):
            raise TrainerError(
                "RM publication authority differs from the durable Trainer intent",
                code="artifact_publication_authority_mismatch",
                status=409,
            )
        return authority

    def _register_reserved_publication(
        self,
        registration: PublicationRegistration,
        *,
        timeout_seconds: float = 3600,
    ) -> PublicationRegistration:
        expected_hash = sha256_json({
            "topologyId": self.rm.topology_id,
            **registration.request.api_dict(exclude_none=False),
        })
        if expected_hash != registration.request_sha256:
            raise TrainerError(
                "Durable publication request changed",
                code="artifact_publication_intent_changed",
                status=409,
            )
        try:
            response = self.rm.register_trainer_publication(
                registration.request.api_dict(exclude_none=False),
                idempotency_key=registration.idempotency_key,
                timeout_seconds=timeout_seconds,
            )
            authority = self._validate_publication_authority(registration, response)
            _result, _publication, stored = self.store.acknowledge_publication_authority(
                registration_id=registration.registration_id,
                expected_generation=registration.generation,
                authority=authority.api_dict(exclude_none=False),
            )
            return PublicationRegistration.model_validate(stored)
        except BaseException as exc:
            code = str(getattr(exc, "code", "rm_unavailable"))[:128]
            try:
                self.store.mark_publication_reconciliation_required(
                    registration.registration_id,
                    error={
                        "code": code,
                        "detail": "Publication registration has no accepted RM authority",
                    },
                )
            except BaseException:
                pass
            raise TrainerError(
                "Trainer cannot prove the RM publication outcome",
                code="artifact_publication_outcome_unknown",
                status=503,
                fields={"registrationId": registration.registration_id},
            ) from exc

    def _recover_publications(self) -> None:
        """Perform one bounded startup pass without launching training work."""

        payloads = self.store.recoverable_publication_registrations(limit=256)
        external_replays = 0
        for payload in payloads:
            registration: PublicationRegistration | None = None
            registration_id = payload.get("registrationId")
            recovery_complete = False
            recovery_examined = False
            deferred_code = "publication_recovery_deferred"
            try:
                registration = PublicationRegistration.model_validate(payload)
                if (
                    self._publication_source_expectation()
                    != registration.source_expectation
                ):
                    recovery_examined = True
                    raise TrainerError(
                        "Trainer publication source changed before recovery",
                        code="artifact_publication_authority_mismatch",
                        status=409,
                    )
                if registration.state == "acknowledged":
                    recovery_examined = True
                    if registration.authority is None:
                        raise TrainerError(
                            "Acknowledged publication omits its authority",
                            code="artifact_publication_authority_mismatch",
                            status=409,
                        )
                    self._validate_publication_authority(
                        registration, registration.authority.api_dict(exclude_none=False)
                    )
                    acknowledged = registration
                else:
                    if not self.rm.available or external_replays >= 1:
                        continue
                    recovery_examined = True
                    external_replays += 1
                    acknowledged = self._register_reserved_publication(
                        registration, timeout_seconds=5
                    )
                if not self.store.settle_acknowledged_publication(
                    acknowledged.registration_id
                ):
                    continue
                recovery_complete = self._settle_recovered_campaign(acknowledged)
            except BaseException as exc:
                # The durable registration remains the recovery receipt. Startup
                # must not turn an uncertain external effect into retraining.
                deferred_code = str(getattr(exc, "code", deferred_code))[:128]
            finally:
                if (
                    recovery_examined
                    and not recovery_complete
                    and isinstance(registration_id, str)
                ):
                    try:
                        self.store.mark_publication_recovery_deferred(
                            registration_id,
                            error={
                                "code": deferred_code,
                                "detail": "Publication still requires local reconciliation",
                            },
                        )
                    except BaseException:
                        pass

    def _settle_recovered_campaign(
        self, registration: PublicationRegistration
    ) -> bool:
        lineage = registration.lineage
        jobs = self.campaign_jobs(lineage.campaign_id)
        if not jobs or any(item.get("state") != "succeeded" for item in jobs):
            return False
        result = {
            "campaignId": lineage.campaign_id,
            "results": self.campaign_results(lineage.campaign_id),
        }
        return self.store.settle_campaign_succeeded(
            lineage.campaign_id,
            lineage.start_operation_id,
            result,
            controller_id=registration.start_controller_id,
            reconciled_publication_registration_id=registration.registration_id,
        )

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
        for reservation in self.store.incomplete_campaign_cancels():
            if self._controller_lease.owner_is_active(reservation["controller_id"]):
                continue
            self.store.mark_campaign_cancel_reconciliation_required(
                reservation["campaign_id"],
                reservation["cancel_operation_id"],
                controller_id=reservation["controller_id"],
                reason=(
                    "The controller that signalled cancellation is no longer active; "
                    "Trainer cannot prove that all launched workloads are terminal"
                ),
            )
        reservations = self.store.incomplete_campaign_starts()
        reserved_campaigns = {item["campaign_id"] for item in reservations}
        for reservation in reservations:
            if self._controller_lease.owner_is_active(reservation["controller_id"]):
                continue
            self.store.mark_campaign_start_reconciliation_required(
                reservation["campaign_id"],
                reservation["operation_id"],
                controller_id=reservation["controller_id"],
                reason=(
                    "The controller that reserved this start is no longer active; "
                    "Trainer will not launch replacement work automatically"
                ),
            )
        for payload in self.store.list_entities("campaign"):
            if (
                payload.get("state") in {"running", "recovering"}
                and payload.get("metadata", {}).get("id") not in reserved_campaigns
            ):
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
