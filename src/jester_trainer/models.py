from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .util import safe_identifier, utc_now


API_VERSION = "jester.trainer/v1"


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", use_enum_values=True)

    def api_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        return self.model_dump(by_alias=True, mode="json", exclude_none=exclude_none)


class StrictAPIModel(APIModel):
    model_config = ConfigDict(
        populate_by_name=True, extra="forbid", use_enum_values=True, strict=True
    )


class CampaignState(str, Enum):
    DRAFT = "draft"
    VALIDATING = "validating"
    PLANNED = "planned"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"


class WorkerState(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    DRAINING = "draining"
    UNAVAILABLE = "unavailable"


class RoundState(str, Enum):
    PREPARING = "preparing"
    COLLECTING = "collecting"
    COMMITTED = "committed"
    ABORTED = "aborted"


class OperationState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECONCILIATION_REQUIRED = "reconciliation-required"


class Metadata(APIModel):
    id: str
    display_name: str = Field(default="", alias="displayName")
    labels: dict[str, str] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        return safe_identifier(value)


class DatasetFile(APIModel):
    split: Literal["pretraining", "train", "validation", "test"]
    digest: str
    path: str | None = None
    artifact_digest: str | None = Field(default=None, alias="artifactDigest")
    logical_path: str | None = Field(default=None, alias="logicalPath")
    records: int | None = Field(default=None, ge=0)

    @field_validator("digest", "artifact_digest")
    @classmethod
    def digest_format(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower().removeprefix("sha256:")
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError("digest must be a SHA-256 value")
        return f"sha256:{normalized}"

    @model_validator(mode="after")
    def path_or_artifact(self) -> "DatasetFile":
        if bool(self.path) == bool(self.artifact_digest):
            raise ValueError("dataset file requires exactly one of path or artifactDigest")
        return self


class DatasetManifest(APIModel):
    api_version: Literal[API_VERSION] = Field(default=API_VERSION, alias="apiVersion")
    kind: Literal["DatasetManifest"] = "DatasetManifest"
    metadata: Metadata
    files: list[DatasetFile]
    semantic_case_id_field: str = Field(default="semanticCaseId", alias="semanticCaseIdField")
    seed: int = 42
    generation: int = Field(default=1, ge=1)
    created_at: str = Field(default_factory=utc_now, alias="createdAt")
    updated_at: str = Field(default_factory=utc_now, alias="updatedAt")

    @model_validator(mode="after")
    def unique_files(self) -> "DatasetManifest":
        keys = [(item.split, item.digest) for item in self.files]
        if len(keys) != len(set(keys)):
            raise ValueError("dataset files must be unique by split and digest")
        if not self.files:
            raise ValueError("dataset manifest must contain at least one file")
        return self


class DatasetUse(APIModel):
    dataset_id: str = Field(alias="datasetId")
    split: Literal["pretraining", "train", "validation"] = "train"
    weight: float = Field(default=1.0, gt=0)

    @field_validator("dataset_id")
    @classmethod
    def safe_dataset_id(cls, value: str) -> str:
        return safe_identifier(value)


class CandidateSpec(APIModel):
    id: str
    overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def safe_candidate_id(cls, value: str) -> str:
        return safe_identifier(value)


class FacetNodeSpec(APIModel):
    id: str
    depends_on: list[str] = Field(default_factory=list, alias="dependsOn")
    datasets: list[DatasetUse] = Field(default_factory=list)
    mi_config: dict[str, Any] = Field(alias="miConfig")
    candidates: list[CandidateSpec] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def safe_facet_id(cls, value: str) -> str:
        return safe_identifier(value)

    @model_validator(mode="after")
    def default_candidate(self) -> "FacetNodeSpec":
        if not self.candidates:
            self.candidates = [CandidateSpec(id="default")]
        candidate_ids = [candidate.id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"facet {self.id} candidate IDs must be unique")
        dataset_keys = [(item.dataset_id, item.split) for item in self.datasets]
        if len(dataset_keys) != len(set(dataset_keys)):
            raise ValueError(
                f"facet {self.id} dataset uses must be unique by datasetId and split"
            )
        return self


class CohortSpec(APIModel):
    worker_ids: list[str] = Field(default_factory=list, alias="workerIds")
    size: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def selection_mode(self) -> "CohortSpec":
        if self.worker_ids and self.size is not None:
            raise ValueError("cohort accepts workerIds or size, not both")
        if len(self.worker_ids) != len(set(self.worker_ids)):
            raise ValueError("cohort workerIds must be unique")
        return self


class OptimizerSpec(APIModel):
    type: Literal["adamw", "sgd"] = "adamw"
    learning_rate: float = Field(default=2e-4, alias="learningRate", gt=0)
    weight_decay: float = Field(default=0.0, alias="weightDecay", ge=0)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = Field(default=1e-8, gt=0)
    momentum: float = Field(default=0.0, ge=0, lt=1)


class SchedulerSpec(APIModel):
    type: Literal["constant", "linear", "cosine"] = "constant"
    warmup_steps: int = Field(default=0, alias="warmupSteps", ge=0)


class TrainingSpec(APIModel):
    max_steps: int = Field(alias="maxSteps", gt=0)
    examples_per_worker: int = Field(default=1, alias="examplesPerWorker", gt=0)
    batch_size: int = Field(default=1, alias="batchSize", gt=0)
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    max_grad_norm: float = Field(default=1.0, alias="maxGradNorm", ge=0)
    seed: int = 42
    transport_dtype: Literal["float32", "float16", "bfloat16"] = Field(
        default="float32", alias="transportDtype"
    )
    compression: Literal["none", "zstd"] = "none"


class CheckpointSpec(APIModel):
    output_directory: str = Field(alias="outputDirectory")
    save_every_rounds: int = Field(default=1, alias="saveEveryRounds", ge=1)
    max_retries: int = Field(default=1, alias="maxRetries", ge=0, le=20)
    resume_from: str | None = Field(default=None, alias="resumeFrom")
    initial_weights: str | None = Field(default=None, alias="initialWeights")


class SelectionSpec(APIModel):
    metric: str
    mode: Literal["min", "max"] = "min"


class CampaignSpec(APIModel):
    topology_id: str = Field(alias="topologyId")
    datasets: list[DatasetUse]
    facets: list[FacetNodeSpec]
    mode: Literal["independent", "parameter-server"] = "parameter-server"
    cohort: CohortSpec = Field(default_factory=CohortSpec)
    optimizer: OptimizerSpec = Field(default_factory=OptimizerSpec)
    scheduler: SchedulerSpec = Field(default_factory=SchedulerSpec)
    training: TrainingSpec
    checkpoint: CheckpointSpec
    selection: SelectionSpec | None = None
    accept_ssh_data_plane: bool = Field(default=False, alias="acceptSshDataPlane")
    allow_cpu_fallback: bool = Field(default=False, alias="allowCpuFallback")

    @model_validator(mode="after")
    def validate_dag(self) -> "CampaignSpec":
        if not self.facets:
            raise ValueError("campaign requires at least one facet node")
        missing_datasets = [
            facet.id for facet in self.facets if not facet.datasets and not self.datasets
        ]
        if missing_datasets:
            raise ValueError(
                "every facet requires datasets or a campaign-level dataset fallback: "
                f"{missing_datasets}"
            )
        ids = [facet.id for facet in self.facets]
        if len(ids) != len(set(ids)):
            raise ValueError("campaign facet IDs must be unique")
        known = set(ids)
        dependencies = {facet.id: set(facet.depends_on) for facet in self.facets}
        for facet_id, required in dependencies.items():
            unknown = sorted(required.difference(known))
            if unknown:
                raise ValueError(f"facet {facet_id} depends on unknown facets: {unknown}")
            if facet_id in required:
                raise ValueError(f"facet {facet_id} cannot depend on itself")
        pending = {key: set(value) for key, value in dependencies.items()}
        while pending:
            ready = [key for key, value in pending.items() if not value]
            if not ready:
                raise ValueError("campaign facet graph must be acyclic")
            for key in ready:
                pending.pop(key)
                for value in pending.values():
                    value.discard(key)
        return self


class Campaign(APIModel):
    api_version: Literal[API_VERSION] = Field(default=API_VERSION, alias="apiVersion")
    kind: Literal["Campaign"] = "Campaign"
    metadata: Metadata
    spec: CampaignSpec
    state: CampaignState = CampaignState.DRAFT
    generation: int = Field(default=1, ge=1)
    created_at: str = Field(default_factory=utc_now, alias="createdAt")
    updated_at: str = Field(default_factory=utc_now, alias="updatedAt")


class TrainingJob(APIModel):
    api_version: Literal[API_VERSION] = Field(default=API_VERSION, alias="apiVersion")
    kind: Literal["TrainingJob"] = "TrainingJob"
    job_id: str = Field(alias="jobId")
    campaign_id: str = Field(alias="campaignId")
    facet_id: str = Field(alias="facetId")
    candidate_id: str = Field(alias="candidateId")
    depends_on: list[str] = Field(default_factory=list, alias="dependsOn")
    state: Literal["queued", "preparing", "running", "succeeded", "failed", "cancelled"] = "queued"
    worker_ids: list[str] = Field(default_factory=list, alias="workerIds")
    attempt: int = 0
    result_id: str | None = Field(default=None, alias="resultId")
    error: dict[str, Any] | None = None
    generation: int = Field(default=1, ge=1)
    created_at: str = Field(default_factory=utc_now, alias="createdAt")
    updated_at: str = Field(default_factory=utc_now, alias="updatedAt")


class ExecutionLineage(APIModel):
    schema_version: Literal["jester-trainer-execution-lineage-v1"] = Field(
        default="jester-trainer-execution-lineage-v1", alias="schemaVersion"
    )
    project_id: str = Field(alias="projectId")
    deployment_id: str = Field(alias="deploymentId")
    campaign_id: str = Field(alias="campaignId")
    start_operation_id: str = Field(alias="startOperationId")
    job_id: str = Field(alias="jobId")
    attempt_id: str = Field(alias="attemptId")


class TrainingAttempt(APIModel):
    api_version: Literal[API_VERSION] = Field(default=API_VERSION, alias="apiVersion")
    kind: Literal["TrainingAttempt"] = "TrainingAttempt"
    attempt_id: str = Field(alias="attemptId")
    job_id: str = Field(alias="jobId")
    number: int = Field(ge=1)
    cohort: list[str]
    lineage: ExecutionLineage | None = None
    state: Literal[
        "running", "succeeded", "failed", "cancelled", "reconciliation-required"
    ] = "running"
    started_at: str = Field(default_factory=utc_now, alias="startedAt")
    completed_at: str | None = Field(default=None, alias="completedAt")
    error: dict[str, Any] | None = None
    generation: int = Field(default=1, ge=1)
    created_at: str = Field(default_factory=utc_now, alias="createdAt")
    updated_at: str = Field(default_factory=utc_now, alias="updatedAt")


class RoundContribution(APIModel):
    worker_id: str = Field(alias="workerId")
    gradient_sha256: str = Field(alias="gradientSha256")
    supervised_tokens: int = Field(alias="supervisedTokens", gt=0)
    examples: int = Field(gt=0)
    mean_loss: float = Field(alias="meanLoss")


class TrainingRound(APIModel):
    api_version: Literal[API_VERSION] = Field(default=API_VERSION, alias="apiVersion")
    kind: Literal["TrainingRound"] = "TrainingRound"
    round_id: str = Field(alias="roundId")
    attempt_id: str = Field(alias="attemptId")
    lineage: ExecutionLineage | None = None
    index: int = Field(ge=0)
    state: RoundState = RoundState.PREPARING
    weight_version: str = Field(alias="weightVersion")
    selection_sha256: str | None = Field(default=None, alias="selectionSha256")
    contributions: list[RoundContribution] = Field(default_factory=list)
    aggregate_sha256: str | None = Field(default=None, alias="aggregateSha256")
    started_at: str = Field(default_factory=utc_now, alias="startedAt")
    committed_at: str | None = Field(default=None, alias="committedAt")
    error: dict[str, Any] | None = None
    generation: int = Field(default=1, ge=1)
    created_at: str = Field(default_factory=utc_now, alias="createdAt")
    updated_at: str = Field(default_factory=utc_now, alias="updatedAt")


class TrainingResult(APIModel):
    api_version: Literal[API_VERSION] = Field(default=API_VERSION, alias="apiVersion")
    kind: Literal["TrainingResult"] = "TrainingResult"
    result_id: str = Field(alias="resultId")
    campaign_id: str = Field(alias="campaignId")
    job_id: str = Field(alias="jobId")
    lineage: ExecutionLineage | None = None
    facet_reference: str = Field(alias="facetReference")
    artifact_path: str = Field(alias="artifactPath")
    artifact_sha256: str = Field(alias="artifactSha256")
    artifact_digest: str | None = Field(default=None, alias="artifactDigest")
    metrics: dict[str, Any] = Field(default_factory=dict)
    promoted: bool = False
    generation: int = Field(default=1, ge=1)
    created_at: str = Field(default_factory=utc_now, alias="createdAt")
    updated_at: str = Field(default_factory=utc_now, alias="updatedAt")


class ArtifactPublication(APIModel):
    schema_version: Literal["jester-trainer-artifact-publication-v1"] = Field(
        default="jester-trainer-artifact-publication-v1", alias="schemaVersion"
    )
    kind: Literal["ArtifactPublication"] = "ArtifactPublication"
    publication_id: str = Field(alias="publicationId")
    result_id: str = Field(alias="resultId")
    lineage: ExecutionLineage
    state: Literal["dispatched", "acknowledged"] = "dispatched"
    artifact_digest: str | None = Field(default=None, alias="artifactDigest")
    artifact_sha256: str = Field(alias="artifactSha256")
    descriptor_kind: str | None = Field(default=None, alias="descriptorKind")
    descriptor_size: int | None = Field(default=None, alias="descriptorSize", ge=0)
    generation: int = Field(default=1, ge=1)
    created_at: str = Field(default_factory=utc_now, alias="createdAt")
    updated_at: str = Field(default_factory=utc_now, alias="updatedAt")


class PublicationSourceExpectation(StrictAPIModel):
    trainer_instance_id: str = Field(alias="trainerInstanceId")
    trainer_instance_generation: int | None = Field(
        default=None, alias="trainerInstanceGeneration", ge=1
    )
    trainer_lease_id: str = Field(alias="trainerLeaseId")
    host_id: str = Field(alias="hostId")
    host_epoch: str | None = Field(default=None, alias="hostEpoch")


class PublicationRegistrationRequest(StrictAPIModel):
    relative_path: str = Field(alias="relativePath", min_length=1, max_length=1024)
    publication_id: str = Field(alias="publicationId")
    result_id: str = Field(alias="resultId")
    lineage: ExecutionLineage
    artifact_sha256: str = Field(alias="artifactSha256", pattern=r"[0-9a-f]{64}")

    @field_validator("artifact_sha256")
    @classmethod
    def exact_artifact_sha256(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("artifactSha256 must be exactly 64 lowercase hex characters")
        return value


class PublicationAuthorityIdentity(StrictAPIModel):
    project_id: str = Field(alias="projectId")
    deployment_id: str = Field(alias="deploymentId")
    deployment_generation: int = Field(alias="deploymentGeneration", ge=1)
    topology_id: str = Field(alias="topologyId")
    topology_generation: int = Field(alias="topologyGeneration", ge=1)


class PublicationAuthoritySource(StrictAPIModel):
    trainer_instance_id: str = Field(alias="trainerInstanceId")
    trainer_instance_generation: int = Field(alias="trainerInstanceGeneration", ge=1)
    trainer_lease_id: str = Field(alias="trainerLeaseId")
    host_id: str = Field(alias="hostId")
    host_epoch: str = Field(alias="hostEpoch")


class PublicationAuthorityResult(StrictAPIModel):
    result_id: str = Field(alias="resultId")
    lineage: ExecutionLineage


class PublicationAuthorityArtifact(StrictAPIModel):
    publication_id: str = Field(alias="publicationId")
    state: Literal["acknowledged"]
    artifact_sha256: str = Field(alias="artifactSha256", pattern=r"[0-9a-f]{64}")
    artifact_digest: str = Field(
        alias="artifactDigest", pattern=r"sha256:[0-9a-f]{64}"
    )
    descriptor_kind: Literal["directory"] = Field(alias="descriptorKind")
    descriptor_size: None = Field(default=None, alias="descriptorSize")

    @field_validator("artifact_sha256")
    @classmethod
    def exact_artifact_sha256(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("artifactSha256 must be exactly 64 lowercase hex characters")
        return value

    @field_validator("artifact_digest")
    @classmethod
    def exact_artifact_digest(cls, value: str) -> str:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
            raise ValueError("artifactDigest must be one canonical sha256 digest")
        return value


class PublicationAuthority(StrictAPIModel):
    schema_version: Literal["jester-rm-trainer-publication-authority-v1"] = Field(
        alias="schemaVersion"
    )
    authority_id: str = Field(alias="authorityId")
    identity: PublicationAuthorityIdentity
    source: PublicationAuthoritySource
    result: PublicationAuthorityResult
    publication: PublicationAuthorityArtifact


class PublicationRegistrationError(StrictAPIModel):
    code: str = Field(min_length=1, max_length=128)
    detail: str = Field(min_length=1, max_length=1024)


class PublicationRegistration(StrictAPIModel):
    schema_version: Literal["jester-trainer-publication-registration-v1"] = Field(
        default="jester-trainer-publication-registration-v1", alias="schemaVersion"
    )
    kind: Literal["PublicationRegistration"] = "PublicationRegistration"
    registration_id: str = Field(alias="registrationId")
    publication_id: str = Field(alias="publicationId")
    result_id: str = Field(alias="resultId")
    lineage: ExecutionLineage
    relative_path: str = Field(alias="relativePath", min_length=1, max_length=1024)
    artifact_sha256: str = Field(alias="artifactSha256", pattern=r"[0-9a-f]{64}")
    idempotency_key: str = Field(
        alias="idempotencyKey", min_length=32, max_length=256,
        pattern=r"[A-Za-z0-9][A-Za-z0-9._~-]{31,255}",
    )
    request_sha256: str = Field(alias="requestSha256", pattern=r"[0-9a-f]{64}")
    request: PublicationRegistrationRequest
    source_expectation: PublicationSourceExpectation = Field(alias="sourceExpectation")
    start_controller_id: str = Field(alias="startControllerId")
    state: Literal["pending", "acknowledged", "reconciliation-required"] = "pending"
    authority: PublicationAuthority | None = None
    error: PublicationRegistrationError | None = None
    generation: int = Field(default=1, ge=1)
    created_at: str = Field(default_factory=utc_now, alias="createdAt")
    updated_at: str = Field(default_factory=utc_now, alias="updatedAt")

    @field_validator("artifact_sha256", "request_sha256")
    @classmethod
    def exact_sha256(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("SHA-256 values must be exactly 64 lowercase hex characters")
        return value

    @field_validator("idempotency_key")
    @classmethod
    def exact_idempotency_key(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._~-]{31,255}", value) is None:
            raise ValueError("idempotencyKey must use the canonical token alphabet")
        return value


class Operation(APIModel):
    operation_id: str = Field(alias="operationId")
    action: str
    target_id: str = Field(alias="targetId")
    state: OperationState = OperationState.QUEUED
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: str = Field(default_factory=utc_now, alias="createdAt")
    updated_at: str = Field(default_factory=utc_now, alias="updatedAt")


class Event(APIModel):
    event_id: str = Field(alias="eventId")
    sequence: int = 0
    timestamp: str = Field(default_factory=utc_now)
    type: str
    campaign_id: str | None = Field(default=None, alias="campaignId")
    job_id: str | None = Field(default=None, alias="jobId")
    round_id: str | None = Field(default=None, alias="roundId")
    severity: Literal["debug", "info", "warning", "error"] = "info"
    data: dict[str, Any] = Field(default_factory=dict)
