from __future__ import annotations

import copy
import json
import math
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors.torch import load_file, save_file

from .errors import ConflictError, TrainerError
from .models import (
    Campaign,
    RoundContribution,
    RoundState,
    TrainingAttempt,
    TrainingJob,
    TrainingRound,
)
from .util import atomic_write_json, sha256_file, sha256_json, utc_now
from .workers import WorkerClient


PersistEntity = Callable[[str, str, dict[str, Any], int | None], dict[str, Any]]
EmitEvent = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ParameterServerResult:
    artifact_path: Path
    artifact_sha256: str
    facet_reference: str
    metrics: dict[str, Any]


class ParameterServer:
    def __init__(
        self,
        *,
        campaign: Campaign,
        job: TrainingJob,
        attempt: TrainingAttempt,
        workers: list[WorkerClient],
        shard_plan: dict[str, Any],
        mi_config: dict[str, Any],
        output_directory: Path,
        persist: PersistEntity,
        emit: EmitEvent,
        cancel_check: Callable[[], None],
    ):
        self.campaign = campaign
        self.job = job
        self.attempt = attempt
        self.workers = workers
        self.shard_plan = shard_plan
        self.mi_config = copy.deepcopy(mi_config)
        self.output = output_directory
        self.persist = persist
        self.emit = emit
        self.cancel_check = cancel_check
        self.parameters: dict[str, torch.nn.Parameter] = {}
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any | None = None
        self.current_version = "init-0"
        self.start_round = 0
        self.session_ids: dict[str, str] = {}
        self.session_summaries: dict[str, dict[str, Any]] = {}
        self.losses: list[float] = []
        self.round_durations: list[float] = []
        self.worker_wait_seconds: dict[str, float] = {
            worker.descriptor.worker_id: 0.0 for worker in workers
        }
        self.total_supervised_tokens = 0
        self.artifact_bindings: list[dict[str, Any]] = []

    def run(self) -> ParameterServerResult:
        self.output.mkdir(parents=True, exist_ok=True)
        self._validate_transport()
        self._create_sessions()
        initial_path = self.output / "coordinator" / "initial.safetensors"
        first = self.workers[0]
        first_session = self.session_ids[first.descriptor.worker_id]
        first_summary = self.session_summaries[first.descriptor.worker_id]
        first.download(
            f"/v1/sessions/{first_session}/weights/{first_summary['weightVersion']}",
            initial_path,
            expected_sha256=str(first_summary["weightSha256"]),
            compression=self.campaign.spec.training.compression,
        )
        try:
            self._initialize_parameters(initial_path)
        finally:
            initial_path.unlink(missing_ok=True)
        self._build_optimizer()
        self._resume_if_requested()
        current_path = self._save_coordinator_weights(self.current_version)
        self._publish_weights(current_path, self.current_version)

        for round_index in range(
            self.start_round, self.campaign.spec.training.max_steps
        ):
            self.cancel_check()
            started = time.perf_counter()
            round_id = f"{self.attempt.attempt_id}-r{round_index:08d}"
            round_model = TrainingRound(
                roundId=round_id,
                attemptId=self.attempt.attempt_id,
                index=round_index,
                state=RoundState.PREPARING,
                weightVersion=self.current_version,
            )
            stored_round = self.persist(
                "round", round_id, round_model.api_dict(), None
            )
            round_model = TrainingRound.model_validate(stored_round)
            self.emit(
                "ROUND_STARTED",
                campaign_id=self.campaign.metadata.id,
                job_id=self.job.job_id,
                round_id=round_id,
                data={"weightVersion": self.current_version},
            )
            try:
                collecting = round_model.model_copy(
                    update={"state": RoundState.COLLECTING}
                )
                round_model = TrainingRound.model_validate(
                    self.persist(
                        "round",
                        round_id,
                        collecting.api_dict(),
                        round_model.generation,
                    )
                )
                contributions, gradients, timings = self._collect_round(
                    round_index
                )
                aggregate, aggregate_sha = self._aggregate(
                    contributions, gradients
                )
                gradient_norm = self._apply_update(aggregate)
                next_version = f"r{round_index + 1:08d}"
                previous_path = current_path
                current_path = self._save_coordinator_weights(next_version)
                self.current_version = next_version
                self.losses.append(
                    sum(
                        item.mean_loss * item.supervised_tokens
                        for item in contributions
                    )
                    / sum(item.supervised_tokens for item in contributions)
                )
                self.total_supervised_tokens += sum(
                    item.supervised_tokens for item in contributions
                )
                duration = time.perf_counter() - started
                self.round_durations.append(duration)
                slowest = max(timings.values()) if timings else 0.0
                for worker_id, elapsed in timings.items():
                    self.worker_wait_seconds[worker_id] += max(0.0, slowest - elapsed)
                committed = round_model.model_copy(
                    update={
                        "state": RoundState.COMMITTED,
                        "contributions": contributions,
                        "aggregate_sha256": aggregate_sha,
                        "committed_at": utc_now(),
                    }
                )
                round_model = TrainingRound.model_validate(
                    self.persist(
                        "round",
                        round_id,
                        committed.api_dict(),
                        round_model.generation,
                    )
                )
                if (
                    (round_index + 1)
                    % self.campaign.spec.checkpoint.save_every_rounds
                    == 0
                    or round_index + 1 == self.campaign.spec.training.max_steps
                ):
                    self._checkpoint(round_index + 1, current_path)
                self._ack_rounds(round_index, contributions)
                self._cleanup_downloaded_gradients(gradients)
                self.emit(
                    "ROUND_COMMITTED",
                    campaign_id=self.campaign.metadata.id,
                    job_id=self.job.job_id,
                    round_id=round_id,
                    data={
                        "weightVersion": self.current_version,
                        "loss": self.losses[-1],
                        "gradientNorm": gradient_norm,
                        "durationSeconds": duration,
                    },
                )
                if round_index + 1 < self.campaign.spec.training.max_steps:
                    self._publish_weights(current_path, self.current_version)
                if previous_path != current_path:
                    previous_path.unlink(missing_ok=True)
            except BaseException as exc:
                aborted = round_model.model_copy(
                    update={
                        "state": RoundState.ABORTED,
                        "error": {
                            "code": getattr(exc, "code", "round_failed"),
                            "detail": f"{type(exc).__name__}: {exc}",
                        },
                    }
                )
                self.persist(
                    "round",
                    round_id,
                    aborted.api_dict(),
                    round_model.generation,
                )
                raise

        final_weights = self._save_coordinator_weights("final")
        metrics = self._metrics()
        try:
            artifact_path, digest, reference = self._write_mi_artifact(
                final_weights, metrics
            )
        finally:
            final_weights.unlink(missing_ok=True)
            current_path.unlink(missing_ok=True)
        return ParameterServerResult(
            artifact_path=artifact_path,
            artifact_sha256=digest,
            facet_reference=reference,
            metrics=metrics,
        )

    def _create_sessions(self) -> None:
        config = self._worker_training_config()
        for worker in self.workers:
            worker_id = worker.descriptor.worker_id
            session_id = f"{self.attempt.attempt_id}-{worker_id}"
            payload = {
                "schemaVersion": "mi-worker-task-v1",
                "sessionId": session_id,
                "trainingConfig": config,
                "weightVersion": "worker-init",
                "requestedBackend": worker.descriptor.requested_backend,
                "effectiveBackend": worker.descriptor.effective_backend,
                "fallbackReason": worker.descriptor.fallback_reason,
                "allowCpuFallback": self.campaign.spec.allow_cpu_fallback,
                "artifactBindings": self.artifact_bindings,
                "transport": {
                    "compression": self.campaign.spec.training.compression,
                    "dtype": self.campaign.spec.training.transport_dtype,
                },
            }
            summary = worker.json("POST", "/v1/sessions", payload)
            self.session_ids[worker_id] = session_id
            self.session_summaries[worker_id] = summary
        tensor_sets = {
            sha256_json(summary["engine"]["tensorSpecs"])
            for summary in self.session_summaries.values()
        }
        base_sets = {
            sha256_json(summary["engine"]["base"]) for summary in self.session_summaries.values()
        }
        if len(tensor_sets) != 1 or len(base_sets) != 1:
            raise ConflictError(
                "Synchronous cohort workers disagree on base binding or child tensor structure",
                code="heterogeneous_cohort",
            )

    def _worker_training_config(self) -> dict[str, Any]:
        config = copy.deepcopy(self.mi_config)
        config["schemaVersion"] = "mi-training-v1"
        bindings: list[dict[str, Any]] = []
        base = config.setdefault("baseModel", {})
        base_digest = base.pop("artifactDigest", None)
        base_logical = base.pop("artifactLogicalPath", None)
        if base_digest:
            # Preserve an identity that is stable across the remote CAS
            # checkout and a later local checkout of the exact same model.
            base.setdefault("canonicalId", base_digest)
            bindings.append(
                {
                    "target": "/baseModel/id",
                    "digest": base_digest,
                    "logicalPath": base_logical,
                }
            )
            base["id"] = "rm-artifact-pending"
        registry_digest = config.pop("registryArtifactDigest", None)
        if registry_digest:
            bindings.append(
                {
                    "target": "/registry",
                    "digest": registry_digest,
                }
            )
            config["registry"] = "rm-artifact-pending"
        data = config.setdefault("data", {})
        data["train"] = []
        for index, source in enumerate(self.shard_plan["sources"]):
            source_config = {"path": source["path"]}
            if source.get("logicalPath"):
                source_config["logicalPath"] = source["logicalPath"]
            data["train"].append(source_config)
            if source.get("artifactDigest"):
                bindings.append(
                    {
                        "target": f"/data/train/{index}/path",
                        "digest": source["artifactDigest"],
                        "logicalPath": source.get("logicalPath"),
                    }
                )
        data["shuffle"] = False
        data["packing"] = False
        data["numWorkers"] = 0
        training = config.setdefault("training", {})
        training.update(
            {
                "precision": self.campaign.spec.training.precision,
                "batchSize": self.campaign.spec.training.batch_size,
                "gradientAccumulationSteps": 1,
                "maxSteps": self.campaign.spec.training.max_steps,
                "identityCheck": False,
            }
        )
        config["checkpoint"] = {
            "outputDirectory": str(self.output / "worker-unused"),
            "overwrite": True,
            "saveOptimizer": False,
        }
        self.artifact_bindings = bindings
        return config

    def _initialize_parameters(self, path: Path) -> None:
        state = load_file(str(path), device="cpu")
        initial = self.campaign.spec.checkpoint.initial_weights
        if initial:
            migrated = load_file(str(Path(initial).expanduser().resolve()), device="cpu")
            if set(migrated) != set(state):
                raise ConflictError(
                    "Migrated initial weight tensor set differs from the worker child facet",
                    code="initial_weight_tensor_mismatch",
                )
            for name, tensor in migrated.items():
                if tuple(tensor.shape) != tuple(state[name].shape) or not torch.isfinite(tensor).all():
                    raise ConflictError(
                        f"Migrated initial tensor {name} has invalid shape or values",
                        code="initial_weight_tensor_mismatch",
                    )
            state = migrated
        if not state:
            raise TrainerError(
                "Worker returned an empty child weight set",
                code="empty_weight_set",
                status=409,
            )
        self.parameters = {
            name: torch.nn.Parameter(tensor.detach().float().clone(), requires_grad=True)
            for name, tensor in state.items()
        }

    def _build_optimizer(self) -> None:
        spec = self.campaign.spec.optimizer
        parameters = list(self.parameters.values())
        if spec.type == "adamw":
            self.optimizer = torch.optim.AdamW(
                parameters,
                lr=spec.learning_rate,
                weight_decay=spec.weight_decay,
                betas=spec.betas,
                eps=spec.eps,
            )
        else:
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=spec.learning_rate,
                weight_decay=spec.weight_decay,
                momentum=spec.momentum,
            )
        total = self.campaign.spec.training.max_steps
        warmup = self.campaign.spec.scheduler.warmup_steps
        scheduler_type = self.campaign.spec.scheduler.type

        def factor(step: int) -> float:
            if warmup and step < warmup:
                return max(1e-12, (step + 1) / warmup)
            progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
            if scheduler_type == "constant":
                return 1.0
            if scheduler_type == "linear":
                return max(0.0, 1.0 - progress)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)

    def _collect_round(
        self, round_index: int
    ) -> tuple[list[RoundContribution], dict[str, Path], dict[str, float]]:
        examples = self.campaign.spec.training.examples_per_worker
        requests: dict[Any, tuple[WorkerClient, float]] = {}
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            for worker in self.workers:
                worker_id = worker.descriptor.worker_id
                assigned = list(self.shard_plan["workers"].get(worker_id, []))
                if not assigned:
                    raise ConflictError(
                        f"Worker {worker_id} received an empty deterministic shard",
                        code="empty_worker_shard",
                    )
                offset = (round_index * examples) % len(assigned)
                selected = [
                    assigned[(offset + index) % len(assigned)]
                    for index in range(min(examples, len(assigned)))
                ]
                session_id = self.session_ids[worker_id]
                payload = {
                    "weightVersion": self.current_version,
                    "indices": selected,
                    "batchSize": self.campaign.spec.training.batch_size,
                    "seed": self.campaign.spec.training.seed + round_index,
                }
                started = time.perf_counter()
                future = executor.submit(
                    worker.json,
                    "POST",
                    f"/v1/sessions/{session_id}/rounds/{round_index}/compute",
                    payload,
                )
                requests[future] = (worker, started)
            records: dict[str, dict[str, Any]] = {}
            timings: dict[str, float] = {}
            for future in as_completed(requests):
                worker, started = requests[future]
                worker_id = worker.descriptor.worker_id
                records[worker_id] = future.result()
                timings[worker_id] = time.perf_counter() - started

        contributions: list[RoundContribution] = []
        gradients: dict[str, Path] = {}
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            downloads: dict[Any, tuple[str, dict[str, Any]]] = {}
            for worker in self.workers:
                worker_id = worker.descriptor.worker_id
                record = records[worker_id]
                destination = (
                    self.output
                    / "gradients"
                    / f"round-{round_index:08d}"
                    / f"{worker_id}.safetensors"
                )
                session_id = self.session_ids[worker_id]
                future = executor.submit(
                    worker.download,
                    f"/v1/sessions/{session_id}/rounds/{round_index}/gradient",
                    destination,
                    expected_sha256=str(record["gradientSha256"]),
                    compression=self.campaign.spec.training.compression,
                )
                downloads[future] = (worker_id, record)
            for future in as_completed(downloads):
                worker_id, record = downloads[future]
                gradients[worker_id] = future.result()
                contributions.append(
                    RoundContribution(
                        workerId=worker_id,
                        gradientSha256=record["gradientSha256"],
                        supervisedTokens=record["supervisedTokens"],
                        examples=record["examples"],
                        meanLoss=record["meanLoss"],
                    )
                )
        contributions.sort(key=lambda item: item.worker_id)
        return contributions, gradients, timings

    def _aggregate(
        self,
        contributions: list[RoundContribution],
        paths: dict[str, Path],
    ) -> tuple[dict[str, torch.Tensor], str]:
        total_tokens = sum(item.supervised_tokens for item in contributions)
        aggregate = {
            name: torch.zeros_like(parameter.detach(), dtype=torch.float32)
            for name, parameter in self.parameters.items()
        }
        seen: dict[str, str] = {}
        for contribution in contributions:
            previous = seen.get(contribution.worker_id)
            if previous is not None:
                if previous == contribution.gradient_sha256:
                    continue
                raise ConflictError(
                    f"Worker {contribution.worker_id} contributed two hashes",
                    code="gradient_idempotency_conflict",
                )
            seen[contribution.worker_id] = contribution.gradient_sha256
            state = load_file(str(paths[contribution.worker_id]), device="cpu")
            missing = sorted(set(self.parameters).difference(state))
            unexpected = sorted(set(state).difference(self.parameters))
            if missing or unexpected:
                raise ConflictError(
                    f"Gradient tensor set mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}",
                    code="gradient_tensor_set_mismatch",
                )
            weight = contribution.supervised_tokens / total_tokens
            for name, parameter in self.parameters.items():
                tensor = state[name].float()
                if tuple(tensor.shape) != tuple(parameter.shape):
                    raise ConflictError(
                        f"Gradient {name} shape mismatch",
                        code="gradient_shape_mismatch",
                    )
                if not torch.isfinite(tensor).all():
                    raise ConflictError(
                        f"Gradient {name} contains NaN or Inf",
                        code="non_finite_gradient",
                    )
                aggregate[name].add_(tensor, alpha=weight)
        digest = sha256_json(
            {
                name: {
                    "shape": list(tensor.shape),
                    "sum": float(tensor.double().sum().item()),
                    "norm": float(tensor.double().norm().item()),
                }
                for name, tensor in aggregate.items()
            }
        )
        return aggregate, digest

    def _apply_update(self, aggregate: dict[str, torch.Tensor]) -> float:
        assert self.optimizer is not None and self.scheduler is not None
        self.optimizer.zero_grad(set_to_none=True)
        for name, parameter in self.parameters.items():
            parameter.grad = aggregate[name].to(dtype=parameter.dtype)
        maximum = self.campaign.spec.training.max_grad_norm
        if maximum > 0:
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    list(self.parameters.values()), maximum
                ).item()
            )
        else:
            norm = float(
                torch.sqrt(
                    sum(
                        parameter.grad.detach().float().pow(2).sum()
                        for parameter in self.parameters.values()
                        if parameter.grad is not None
                    )
                ).item()
            )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return norm

    def _save_coordinator_weights(self, version: str) -> Path:
        path = self.output / "coordinator" / f"{version}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                name: parameter.detach().cpu().contiguous()
                for name, parameter in self.parameters.items()
            },
            str(path),
        )
        return path

    def _publish_weights(self, path: Path, version: str) -> None:
        transport_path = self._transport_weight_path(path)
        try:
            with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
                futures = {
                    executor.submit(
                        worker.upload_weights,
                        self.session_ids[worker.descriptor.worker_id],
                        version,
                        transport_path,
                        compression=self.campaign.spec.training.compression,
                    ): worker.descriptor.worker_id
                    for worker in self.workers
                }
                for future in as_completed(futures):
                    future.result()
        finally:
            if transport_path != path:
                transport_path.unlink(missing_ok=True)
                transport_path.with_suffix(transport_path.suffix + ".zst").unlink(
                    missing_ok=True
                )

    def _transport_weight_path(self, path: Path) -> Path:
        dtype_name = self.campaign.spec.training.transport_dtype
        if dtype_name == "float32":
            return path
        target_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(dtype_name)
        if target_dtype is None:
            raise TrainerError(
                f"Unsupported weight transport dtype: {dtype_name}",
                code="unsupported_transport_dtype",
                status=422,
            )
        state = load_file(str(path), device="cpu")
        destination = path.with_name(f"{path.stem}.{dtype_name}{path.suffix}")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            save_file(
                {
                    name: tensor.to(dtype=target_dtype).contiguous()
                    for name, tensor in state.items()
                },
                str(temporary),
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def _ack_rounds(
        self, round_index: int, contributions: list[RoundContribution]
    ) -> None:
        by_worker = {item.worker_id: item for item in contributions}
        for worker in self.workers:
            worker_id = worker.descriptor.worker_id
            worker.json(
                "POST",
                f"/v1/sessions/{self.session_ids[worker_id]}/rounds/{round_index}/ack",
                {"gradientSha256": by_worker[worker_id].gradient_sha256},
            )

    @staticmethod
    def _cleanup_downloaded_gradients(paths: dict[str, Path]) -> None:
        parents: set[Path] = set()
        for path in paths.values():
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".zst").unlink(missing_ok=True)
            parents.add(path.parent)
        for parent in parents:
            try:
                parent.rmdir()
            except OSError:
                pass

    def _checkpoint(self, next_round: int, weights_path: Path) -> Path:
        assert self.optimizer is not None and self.scheduler is not None
        checkpoints = self.output / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        destination = checkpoints / f"round-{next_round:08d}"
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=checkpoints)
        )
        try:
            shutil.copyfile(weights_path, temporary / "weights.safetensors")
            torch.save(
                {
                    "schemaVersion": "jester-trainer-state-v1",
                    "nextRound": next_round,
                    "weightVersion": self.current_version,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "sampler": {
                        "seed": self.campaign.spec.training.seed,
                        "examplesPerWorker": self.campaign.spec.training.examples_per_worker,
                    },
                    "cohort": [
                        worker.descriptor.worker_id for worker in self.workers
                    ],
                    "datasetHashes": [
                        source["digest"] for source in self.shard_plan["sources"]
                    ],
                    "lastConfirmedRound": next_round - 1,
                    "metrics": self._checkpoint_metrics(),
                },
                temporary / "trainer-state.pt",
            )
            atomic_write_json(
                temporary / "checkpoint.json",
                {
                    "schemaVersion": "jester-trainer-checkpoint-v1",
                    "nextRound": next_round,
                    "weightVersion": self.current_version,
                    "weightsSha256": sha256_file(
                        temporary / "weights.safetensors"
                    ),
                },
            )
            if destination.exists():
                existing = json.loads(
                    (destination / "checkpoint.json").read_text(encoding="utf-8")
                )
                if existing["weightsSha256"] == sha256_file(
                    temporary / "weights.safetensors"
                ):
                    shutil.rmtree(temporary)
                    return destination
                raise ConflictError(
                    f"Checkpoint {destination} already exists with another hash",
                    code="checkpoint_conflict",
                )
            os.replace(temporary, destination)
            return destination
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _resume_if_requested(self) -> None:
        source = self.campaign.spec.checkpoint.resume_from
        if not source:
            return
        checkpoint = Path(source).expanduser().resolve()
        weights = checkpoint / "weights.safetensors"
        state_path = checkpoint / "trainer-state.pt"
        if not weights.is_file() or not state_path.is_file():
            raise TrainerError(
                f"Incomplete checkpoint: {checkpoint}",
                code="checkpoint_incomplete",
                status=409,
            )
        loaded = load_file(str(weights), device="cpu")
        if set(loaded) != set(self.parameters):
            raise ConflictError(
                "Checkpoint tensor set differs from the current child facet",
                code="checkpoint_tensor_mismatch",
            )
        with torch.no_grad():
            for name, parameter in self.parameters.items():
                if tuple(loaded[name].shape) != tuple(parameter.shape):
                    raise ConflictError(
                        f"Checkpoint tensor {name} shape differs",
                        code="checkpoint_tensor_mismatch",
                    )
                parameter.copy_(loaded[name].float())
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")
        assert self.optimizer is not None and self.scheduler is not None
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        cohort = [worker.descriptor.worker_id for worker in self.workers]
        if state.get("cohort") != cohort:
            raise ConflictError(
                "Checkpoint cohort differs; synchronous v1 resumes the complete cohort",
                code="checkpoint_cohort_mismatch",
            )
        if state.get("datasetHashes") != [
            source["digest"] for source in self.shard_plan["sources"]
        ]:
            raise ConflictError(
                "Checkpoint dataset hashes differ",
                code="checkpoint_dataset_mismatch",
            )
        self.start_round = int(state["nextRound"])
        self.current_version = str(state["weightVersion"])
        self._restore_checkpoint_metrics(state)

    def _checkpoint_metrics(self) -> dict[str, Any]:
        return {
            "losses": list(self.losses),
            "roundDurations": list(self.round_durations),
            "workerWaitSeconds": dict(self.worker_wait_seconds),
            "supervisedTokens": self.total_supervised_tokens,
        }

    def _restore_checkpoint_metrics(self, state: dict[str, Any]) -> None:
        """Restore cumulative metrics without breaking legacy v1 checkpoints."""

        raw = state.get("metrics")
        if raw is None:
            return
        if not isinstance(raw, dict):
            raise ConflictError(
                "Checkpoint metrics state is invalid",
                code="checkpoint_metrics_invalid",
            )
        try:
            losses = [float(item) for item in raw["losses"]]
            durations = [float(item) for item in raw["roundDurations"]]
            waits = {
                str(worker_id): float(value)
                for worker_id, value in raw["workerWaitSeconds"].items()
            }
            supervised_tokens = int(raw["supervisedTokens"])
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ConflictError(
                "Checkpoint metrics state is invalid",
                code="checkpoint_metrics_invalid",
            ) from exc
        if (
            len(losses) != len(durations)
            or len(losses) != self.start_round
            or any(duration < 0 for duration in durations)
            or any(value < 0 for value in waits.values())
            or supervised_tokens < 0
            or set(waits) != set(self.worker_wait_seconds)
        ):
            raise ConflictError(
                "Checkpoint metrics do not match the confirmed rounds or cohort",
                code="checkpoint_metrics_invalid",
            )
        self.losses = losses
        self.round_durations = durations
        self.worker_wait_seconds = waits
        self.total_supervised_tokens = supervised_tokens

    def _write_mi_artifact(
        self, weights_path: Path, metrics: dict[str, Any]
    ) -> tuple[Path, str, str]:
        first_summary = next(iter(self.session_summaries.values()))
        manifest = copy.deepcopy(first_summary["engine"]["facet"])
        converted: dict[str, torch.Tensor] = {}
        state = load_file(str(weights_path), device="cpu")
        for slot in manifest["module"]["slots"]:
            slot_id = slot["id"]
            for index, block in enumerate(slot["blocks"]):
                source_prefix = f"{slot_id}.{index}."
                checkpoint_prefix = f"slots.{slot_id}.blocks.{index:04d}"
                block["checkpointPrefix"] = checkpoint_prefix
                matched = False
                for name, tensor in state.items():
                    if name.startswith(source_prefix):
                        converted[
                            f"{checkpoint_prefix}.{name[len(source_prefix):]}"
                        ] = tensor
                        matched = True
                if not matched:
                    raise ConflictError(
                        f"Final state has no tensors for {slot_id}[{index}]",
                        code="artifact_tensor_missing",
                    )
        artifact_root = self.output / "result"
        artifact_root.mkdir(parents=True, exist_ok=True)
        artifact_weights = artifact_root / "facet.safetensors"
        save_file(converted, str(artifact_weights))
        digest = sha256_file(artifact_weights)
        manifest["status"] = "trained"
        manifest["artifact"] = {
            "weights": "facet.safetensors",
            "format": "safetensors",
            "sha256": digest,
        }
        manifest.setdefault("metadata", {}).update(
            {
                "trainer": "Jester Trainer 5/5",
                "campaignId": self.campaign.metadata.id,
                "jobId": self.job.job_id,
                "completedAt": utc_now(),
                "metrics": metrics,
            }
        )
        atomic_write_json(artifact_root / "facet.mi.json", manifest)
        atomic_write_json(
            artifact_root / "training-result.json",
            {
                "schemaVersion": "jester-training-result-v1",
                "status": "succeeded",
                "facet": f"{manifest['id']}@{manifest['version']}",
                "artifactSha256": digest,
                "metrics": metrics,
            },
        )
        return artifact_root, digest, f"{manifest['id']}@{manifest['version']}"

    def _metrics(self) -> dict[str, Any]:
        total_tokens = self.total_supervised_tokens
        elapsed = sum(self.round_durations)
        return {
            "rounds": len(self.round_durations),
            "meanTrainingLoss": sum(self.losses) / max(1, len(self.losses)),
            "lastTrainingLoss": self.losses[-1] if self.losses else None,
            "supervisedTokens": total_tokens,
            "elapsedSeconds": elapsed,
            "tokensPerSecond": total_tokens / elapsed if elapsed else 0.0,
            "meanRoundSeconds": elapsed / max(1, len(self.round_durations)),
            "workerWaitSeconds": self.worker_wait_seconds,
            "cohort": [worker.descriptor.worker_id for worker in self.workers],
            "optimizer": self.campaign.spec.optimizer.api_dict(),
            "scheduler": self.campaign.spec.scheduler.api_dict(),
        }

    def _validate_transport(self) -> None:
        ssh = [
            worker.descriptor.worker_id
            for worker in self.workers
            if worker.descriptor.transport == "ssh-relay"
        ]
        if ssh and not self.campaign.spec.accept_ssh_data_plane:
            raise TrainerError(
                f"Parameter-server data plane crosses SSH for workers {ssh}; set acceptSshDataPlane=true",
                code="ssh_data_plane_confirmation_required",
                status=409,
            )
        if ssh:
            missing_datasets = [
                source["path"]
                for source in self.shard_plan.get("sources", [])
                if not source.get("artifactDigest")
            ]
            if missing_datasets:
                raise TrainerError(
                    "Remote workers require every gradient dataset source in the RM CAS",
                    code="remote_dataset_requires_cas",
                    status=409,
                    fields={"localPaths": missing_datasets},
                )
            base = self.mi_config.get("baseModel", {})
            if not isinstance(base, dict) or not base.get("artifactDigest"):
                raise TrainerError(
                    "Remote workers require baseModel.artifactDigest from the RM CAS",
                    code="remote_base_model_requires_cas",
                    status=409,
                )
