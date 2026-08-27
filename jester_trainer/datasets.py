from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .errors import ConflictError, TrainerError
from .models import DatasetFile, DatasetManifest, DatasetUse
from .util import sha256_file


class DatasetManager:
    def __init__(self, artifact_manifest_path: str | Path | None = None):
        self.artifact_manifest_path = (
            Path(artifact_manifest_path).resolve()
            if artifact_manifest_path
            else None
        )
        self._artifact_entries = self._load_artifacts()

    def validate(self, manifest: DatasetManifest) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        total = 0
        semantic_cases: set[str] = set()
        warnings: list[str] = []
        for item in manifest.files:
            path = self.resolve(item)
            actual = sha256_file(path)
            expected = item.digest.removeprefix("sha256:")
            if actual != expected:
                raise TrainerError(
                    f"Dataset hash mismatch for {path}: {actual} != {expected}",
                    code="dataset_hash_mismatch",
                    status=409,
                )
            records = list(self._records(path))
            for index, record in enumerate(records):
                semantic = self._semantic_id(
                    record, manifest.semantic_case_id_field, path, index
                )
                semantic_cases.add(semantic)
            total += len(records)
            if item.split != "test" and path.name.lower() in {
                "test.json",
                "test.jsonl",
            }:
                warnings.append(
                    f"{path.name} is marked {item.split}; it will still be rejected as a gradient source"
                )
            files.append(
                {
                    "split": item.split,
                    "path": str(path),
                    "digest": item.digest,
                    "records": len(records),
                }
            )
        return {
            "datasetId": manifest.metadata.id,
            "valid": True,
            "files": files,
            "records": total,
            "semanticCases": len(semantic_cases),
            "warnings": warnings,
        }

    def shard_plan(
        self,
        manifests: list[tuple[DatasetManifest, DatasetUse]],
        worker_ids: list[str],
        *,
        seed: int,
    ) -> dict[str, Any]:
        if not worker_ids:
            raise ConflictError("Cannot shard a dataset over an empty cohort")
        assignments: dict[str, list[int]] = {worker_id: [] for worker_id in worker_ids}
        semantic_groups: dict[str, list[int]] = {}
        sources: list[dict[str, Any]] = []
        global_index = 0
        selected_records = 0
        for manifest, use in manifests:
            for item in manifest.files:
                if item.split != use.split:
                    continue
                path = self.resolve(item)
                if item.split == "test" or path.name.lower() in {
                    "test.json",
                    "test.jsonl",
                }:
                    raise TrainerError(
                        f"Test split cannot supply gradients: {path}",
                        code="test_split_gradient_forbidden",
                        status=422,
                    )
                records = list(self._records(path))
                sources.append(
                    {
                        "path": str(path),
                        "digest": item.digest,
                        "artifactDigest": item.artifact_digest,
                        "logicalPath": item.logical_path,
                        "records": len(records),
                        "datasetId": manifest.metadata.id,
                        "split": item.split,
                    }
                )
                for index, record in enumerate(records):
                    semantic = self._semantic_id(
                        record, manifest.semantic_case_id_field, path, index
                    )
                    group_key = f"{use.split}:{semantic}"
                    semantic_groups.setdefault(group_key, []).append(global_index)
                    global_index += 1
                    selected_records += 1
        if not selected_records:
            raise TrainerError(
                "No records matched campaign dataset uses",
                code="empty_training_dataset",
                status=422,
            )
        owner: dict[str, str] = {}
        worker_groups: dict[str, list[str]] = {item: [] for item in worker_ids}
        for group_key in sorted(semantic_groups):
            digest = hashlib.sha256(
                f"{seed}:{group_key}".encode("utf-8")
            ).digest()
            worker = worker_ids[
                int.from_bytes(digest[:8], "big") % len(worker_ids)
            ]
            owner[group_key] = worker
            worker_groups[worker].append(group_key)
        empty = [worker for worker in worker_ids if not worker_groups[worker]]
        if empty and len(semantic_groups) < len(worker_ids):
            raise ConflictError(
                "Cohort has more workers than independent semantic cases",
                code="insufficient_semantic_shards",
            )
        for worker in empty:
            donor = max(
                worker_ids,
                key=lambda item: (len(worker_groups[item]), item),
            )
            moved = sorted(worker_groups[donor])[-1]
            worker_groups[donor].remove(moved)
            worker_groups[worker].append(moved)
            owner[moved] = worker
        for group_key, indices in semantic_groups.items():
            assignments[owner[group_key]].extend(indices)
        for values in assignments.values():
            values.sort()
        return {
            "schemaVersion": "jester-trainer-shard-plan-v1",
            "seed": seed,
            "workers": assignments,
            "sources": sources,
            "records": selected_records,
            "semanticCases": len(semantic_groups),
        }

    def resolve(self, item: DatasetFile) -> Path:
        if item.path:
            path = Path(item.path).expanduser().resolve()
            if not path.is_file():
                raise TrainerError(
                    f"Dataset file does not exist: {path}",
                    code="dataset_file_not_found",
                    status=404,
                )
            return path
        digest = str(item.artifact_digest)
        entry = self._artifact_entries.get(digest)
        if entry is None:
            raise TrainerError(
                f"RM artifact {digest} is not present in the Trainer lease",
                code="dataset_artifact_not_leased",
                status=409,
            )
        if entry.get("kind") == "file":
            return Path(entry.get("checkoutPath") or entry.get("objectPath", "")).resolve()
        logical = item.logical_path
        if not logical:
            raise TrainerError(
                "logicalPath is required for a directory artifact",
                code="dataset_logical_path_required",
                status=422,
            )
        for file in entry.get("files", []):
            if file.get("path") == logical:
                return Path(
                    file.get("checkoutPath") or file.get("objectPath", "")
                ).resolve()
        raise TrainerError(
            f"Artifact {digest} has no logical path {logical}",
            code="dataset_logical_path_not_found",
            status=404,
        )

    def _load_artifacts(self) -> dict[str, dict[str, Any]]:
        path = self.artifact_manifest_path
        if path is None or not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainerError(
                f"Cannot load RM artifact manifest: {exc}",
                code="invalid_rm_artifact_manifest",
                status=409,
            ) from exc
        return {
            str(item.get("digest")): item
            for item in payload.get("entries", [])
            if isinstance(item, dict) and item.get("digest")
        }

    @staticmethod
    def _records(path: Path) -> Iterable[dict[str, Any]]:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise TrainerError(
                            f"Invalid dataset JSONL {path}:{line_number}: {exc.msg}",
                            code="invalid_dataset_json",
                            status=422,
                        ) from exc
                    if not isinstance(item, dict):
                        raise TrainerError(
                            f"Dataset record {path}:{line_number} is not an object",
                            code="invalid_dataset_record",
                            status=422,
                        )
                    yield item
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TrainerError(
                f"Invalid dataset JSON {path}: {exc.msg}",
                code="invalid_dataset_json",
                status=422,
            ) from exc
        if isinstance(payload, dict):
            for key in ("records", "items", "data"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
            else:
                payload = [payload]
        if not isinstance(payload, list):
            raise TrainerError(
                f"Dataset root must be an array or object: {path}",
                code="invalid_dataset_json",
                status=422,
            )
        for item in payload:
            if not isinstance(item, dict):
                raise TrainerError(
                    f"Dataset item is not an object: {path}",
                    code="invalid_dataset_record",
                    status=422,
                )
            yield item

    @staticmethod
    def _semantic_id(
        record: dict[str, Any], field: str, path: Path, index: int
    ) -> str:
        value = record.get(field)
        if value is None:
            metadata = record.get("metadata")
            if isinstance(metadata, dict):
                value = metadata.get(field)
        if value is None:
            value = record.get("id") or record.get("record_id")
        return str(value) if value is not None else f"{path.name}:{index}"
