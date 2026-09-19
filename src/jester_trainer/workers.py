from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConflictError, NotFoundError, TrainerError
from .util import sha256_file


@dataclass(frozen=True)
class WorkerDescriptor:
    worker_id: str
    node_id: str
    instance_id: str
    base_url: str
    transport: str
    channel_id: str
    protocol: str
    resumable: bool
    requested_backend: str = "cpu"
    effective_backend: str = "cpu"
    fallback_reason: str | None = None

    def api_dict(self) -> dict[str, Any]:
        return {
            "workerId": self.worker_id,
            "nodeId": self.node_id,
            "instanceId": self.instance_id,
            "baseUrl": self.base_url,
            "transport": self.transport,
            "channelId": self.channel_id,
            "protocol": self.protocol,
            "resumable": self.resumable,
            "requestedBackend": self.requested_backend,
            "effectiveBackend": self.effective_backend,
            "fallbackReason": self.fallback_reason,
        }


class WorkerClient:
    def __init__(self, descriptor: WorkerDescriptor, *, timeout: float = 3600):
        self.descriptor = descriptor
        self.timeout = timeout

    def json(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.descriptor.base_url + path,
            data=data,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                problem = json.loads(raw).get("error", {})
            except json.JSONDecodeError:
                problem = {"code": "worker_http_error", "detail": raw}
            raise TrainerError(
                f"Worker {self.descriptor.worker_id}: {problem.get('detail') or problem.get('message') or exc.reason}",
                code=str(problem.get("code", "worker_http_error")),
                status=exc.code,
            ) from exc
        except (OSError, urllib.error.URLError) as exc:
            raise TrainerError(
                f"Worker {self.descriptor.worker_id} is unavailable: {exc}",
                code="worker_unavailable",
                status=503,
            ) from exc
        if not isinstance(value, dict):
            raise TrainerError(
                "Worker returned a non-object JSON response",
                code="invalid_worker_response",
                status=502,
            )
        return value

    def upload_weights(
        self,
        session_id: str,
        version: str,
        path: Path,
        *,
        chunk_bytes: int = 4 * 1024 * 1024,
        retries: int = 4,
        compression: str = "none",
    ) -> dict[str, Any]:
        digest = sha256_file(path)
        transfer_path = self._encoded_path(path, compression)
        total = transfer_path.stat().st_size
        offset = 0
        result: dict[str, Any] = {}
        failures = 0
        with transfer_path.open("rb") as handle:
            while offset < total:
                handle.seek(offset)
                body = handle.read(min(chunk_bytes, total - offset))
                end = offset + len(body) - 1
                request = urllib.request.Request(
                    self.descriptor.base_url
                    + f"/v1/sessions/{session_id}/weights/{version}",
                    data=body,
                    method="PUT",
                    headers={
                        "Content-Type": "application/octet-stream",
                        "Content-Length": str(len(body)),
                        "Content-Range": f"bytes {offset}-{end}/{total}",
                        "X-Content-SHA256": digest,
                        "Content-Encoding": compression
                        if compression != "none"
                        else "identity",
                    },
                )
                while True:
                    try:
                        with urllib.request.urlopen(
                            request, timeout=self.timeout
                        ) as response:
                            result = json.loads(response.read().decode("utf-8"))
                        break
                    except urllib.error.HTTPError as exc:
                        raw = exc.read().decode("utf-8", errors="replace")
                        try:
                            error = json.loads(raw).get("error", {})
                        except json.JSONDecodeError:
                            error = {}
                        if error.get("code") == "upload_offset_conflict":
                            result = self.json(
                                "GET",
                                f"/v1/sessions/{session_id}/weights/{version}/upload",
                            )
                            if result.get("complete"):
                                if result.get("sha256") != digest:
                                    raise ConflictError(
                                        f"Worker upload version conflict for {session_id}"
                                    ) from exc
                                return result
                            break
                        raise TrainerError(
                            error.get("detail", raw or str(exc)),
                            code=error.get("code", "worker_upload_failed"),
                            status=exc.code,
                        ) from exc
                    except (OSError, urllib.error.URLError):
                        failures += 1
                        if failures > retries:
                            raise TrainerError(
                                f"Weight upload to {self.descriptor.worker_id} could not resume",
                                code="worker_transfer_failed",
                                status=503,
                            )
                        time.sleep(min(2 ** (failures - 1), 5))
                        try:
                            result = self.json(
                                "GET",
                                f"/v1/sessions/{session_id}/weights/{version}/upload",
                            )
                        except TrainerError:
                            continue
                        if result.get("complete"):
                            if result.get("sha256") != digest:
                                raise ConflictError(
                                    f"Worker upload version conflict for {session_id}"
                                )
                            return result
                        break
                offset = int(result.get("nextOffset", end + 1))
        if not result.get("complete") or result.get("sha256") != digest:
            raise TrainerError(
                "Worker did not confirm the complete weight hash",
                code="worker_weight_confirmation_failed",
                status=502,
            )
        return result

    def download(
        self,
        path: str,
        destination: Path,
        *,
        expected_sha256: str,
        retries: int = 4,
        compression: str = "none",
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(f".{destination.name}.part")
        for attempt in range(retries + 1):
            offset = staging.stat().st_size if staging.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            if compression == "zstd":
                headers["Accept-Encoding"] = "zstd"
            request = urllib.request.Request(
                self.descriptor.base_url + path, headers=headers, method="GET"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    if offset and response.status != 206:
                        raise TrainerError(
                            "Worker ignored Range while resuming a transfer",
                            code="range_resume_not_supported",
                            status=502,
                        )
                    content_encoding = response.headers.get("Content-Encoding", "identity").lower()
                    content_range = response.headers.get("Content-Range", "")
                    if content_range:
                        total = int(content_range.rsplit("/", 1)[-1])
                    else:
                        total = offset + int(response.headers.get("Content-Length", "0"))
                    mode = "ab" if offset else "wb"
                    with staging.open(mode) as handle:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                if staging.stat().st_size < total:
                    if attempt >= retries:
                        raise TrainerError(
                            "Artifact stream ended before Content-Length was satisfied",
                            code="incomplete_worker_transfer",
                            status=503,
                        )
                    continue
                decoded = staging
                temporary_decoded: Path | None = None
                if content_encoding == "zstd":
                    try:
                        import zstandard
                    except ImportError as exc:
                        raise TrainerError(
                            "zstandard is required to decode worker artifacts",
                            code="compression_unavailable",
                            status=500,
                        ) from exc
                    descriptor, name = tempfile.mkstemp(
                        prefix=f".{destination.name}.", suffix=".decoded", dir=destination.parent
                    )
                    os.close(descriptor)
                    temporary_decoded = Path(name)
                    with staging.open("rb") as source, temporary_decoded.open("wb") as target:
                        zstandard.ZstdDecompressor().copy_stream(source, target)
                    decoded = temporary_decoded
                actual = sha256_file(decoded)
                if actual != expected_sha256:
                    if temporary_decoded is not None:
                        temporary_decoded.unlink(missing_ok=True)
                    raise TrainerError(
                        f"Downloaded hash {actual} differs from {expected_sha256}",
                        code="gradient_hash_mismatch",
                        status=409,
                    )
                if temporary_decoded is not None:
                    os.replace(temporary_decoded, destination)
                    staging.unlink(missing_ok=True)
                else:
                    os.replace(staging, destination)
                return destination
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                try:
                    error = json.loads(raw).get("error", {})
                except json.JSONDecodeError:
                    error = {}
                raise TrainerError(
                    error.get("detail", raw or str(exc)),
                    code=error.get("code", "worker_download_failed"),
                    status=exc.code,
                ) from exc
            except TrainerError:
                raise
            except (OSError, urllib.error.URLError):
                if attempt >= retries:
                    raise TrainerError(
                        f"Transfer from {self.descriptor.worker_id} failed after resume attempts",
                        code="worker_transfer_failed",
                        status=503,
                    )
                time.sleep(min(2**attempt, 5))
        raise AssertionError("unreachable")

    @staticmethod
    def _encoded_path(path: Path, compression: str) -> Path:
        if compression == "none":
            return path
        if compression != "zstd":
            raise TrainerError(
                f"Unsupported transfer compression: {compression}",
                code="unsupported_compression",
                status=422,
            )
        try:
            import zstandard
        except ImportError as exc:
            raise TrainerError(
                "zstandard is required for compressed transfers",
                code="compression_unavailable",
                status=500,
            ) from exc
        encoded = path.with_suffix(path.suffix + ".zst")
        if encoded.is_file() and encoded.stat().st_mtime_ns >= path.stat().st_mtime_ns:
            return encoded
        descriptor, name = tempfile.mkstemp(
            prefix=f".{encoded.name}.", suffix=".tmp", dir=encoded.parent
        )
        os.close(descriptor)
        temporary = Path(name)
        try:
            with path.open("rb") as source, temporary.open("wb") as target:
                zstandard.ZstdCompressor(level=3).copy_stream(source, target)
            os.replace(temporary, encoded)
        finally:
            temporary.unlink(missing_ok=True)
        return encoded


class WorkerPool:
    def __init__(self, topology: dict[str, Any]):
        nodes = {
            str(item.get("nodeId")): item
            for item in topology.get("nodes", [])
            if isinstance(item, dict)
        }
        descriptors: dict[str, WorkerDescriptor] = {}
        for channel in topology.get("channels", []):
            if not isinstance(channel, dict):
                continue
            destination = str(channel.get("destinationNodeId", ""))
            node = nodes.get(destination, {})
            if node and node.get("role") != "mi.worker":
                continue
            endpoint = channel.get("endpoint", {})
            port = int(endpoint.get("port", 0) or 0)
            if not destination or not port:
                continue
            worker_id = destination
            descriptor = WorkerDescriptor(
                worker_id=worker_id,
                node_id=destination,
                instance_id=str(node.get("instanceId") or destination),
                base_url=f"http://127.0.0.1:{port}",
                transport=str(channel.get("transport", "local-loopback")),
                channel_id=str(channel.get("channelId", "")),
                protocol=str(channel.get("protocol", "http-json")),
                resumable=bool(channel.get("resumable", False)),
                requested_backend=str(node.get("requestedBackend", "cpu")),
                effective_backend=str(node.get("effectiveBackend", "cpu")),
                fallback_reason=node.get("fallbackReason"),
            )
            descriptors[worker_id] = descriptor
        self.descriptors = descriptors
        self.clients = {
            worker_id: WorkerClient(descriptor)
            for worker_id, descriptor in descriptors.items()
        }

    def select(
        self,
        worker_ids: list[str],
        size: int | None,
        *,
        retain_pool: bool = False,
    ) -> list[WorkerClient]:
        available = sorted(self.clients)
        selected = worker_ids or available
        missing = sorted(set(selected).difference(self.clients))
        if missing:
            raise NotFoundError("worker", ",".join(missing))
        if not selected:
            raise ConflictError(
                "The RM topology exposes no mi.worker channels",
                code="empty_worker_pool",
            )
        if size is not None and size > len(selected):
            raise ConflictError(
                f"Cohort size {size} exceeds available pool size {len(selected)}",
                code="cohort_too_large",
            )
        if size is not None and not retain_pool:
            selected = selected[:size]
        return [self.clients[item] for item in selected]

    def status(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for worker_id, client in self.clients.items():
            descriptor = self.descriptors[worker_id]
            try:
                health = client.json("GET", "/v1/health")
                state = "idle" if health.get("status") == "ok" else "unavailable"
                result.append({**descriptor.api_dict(), "state": state, "health": health})
            except TrainerError as exc:
                result.append(
                    {
                        **descriptor.api_dict(),
                        "state": "unavailable",
                        "error": {"code": exc.code, "detail": str(exc)},
                    }
                )
        return result
