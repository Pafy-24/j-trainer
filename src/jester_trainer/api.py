from __future__ import annotations

import asyncio
import json
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Body, Depends, FastAPI, Header, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .errors import AuthenticationError, TrainerError
from .channel_sessions import ChannelSessionDenied, ManagedChannelGuard
from .models import Campaign, DatasetManifest
from .service import TrainerService


def create_app(
    service: TrainerService | None = None, *,
    managed_channel_guard: ManagedChannelGuard | None = None,
    require_managed_session: bool = False,
) -> FastAPI:
    trainer = service or TrainerService()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            trainer.close()

    app = FastAPI(
        title="Jester Trainer",
        description="Component 5/5: RM-managed campaigns, datasets, parameter server and MI workers",
        version="0.1.0",
        openapi_url="/v1/openapi.json",
        docs_url="/v1/docs",
        lifespan=lifespan,
    )
    app.state.trainer = trainer

    @app.middleware("http")
    async def request_id(request: Request, call_next: Any) -> Response:
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        if require_managed_session and request.url.path != "/v1/health":
            operation = None
            campaign_id = None
            if request.method == "POST" and request.url.path == "/v1/campaigns":
                operation = "trainer.campaign:create"
            elif request.method == "GET" and request.url.path.startswith("/v1/campaigns/"):
                operation = "trainer.campaign:status"
                campaign_id = request.url.path.rsplit("/", 1)[-1]
            try:
                authorize_managed_channel(request, operation=operation or "", campaign_id=campaign_id)
                if operation is None:
                    raise ChannelSessionDenied("managed channel operation is not allowed")
            except (ChannelSessionDenied, TrainerError):
                return JSONResponse(status_code=401, content={
                    "code": "channel_session_denied", "detail": "managed channel session is not authorized",
                    "requestId": request.state.request_id,
                })
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(TrainerError)
    async def trainer_error(request: Request, error: TrainerError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status,
            content={
                "type": f"https://jester.local/problems/{error.code}",
                "status": error.status,
                "code": error.code,
                "title": "Trainer request failed",
                "detail": str(error),
                "requestId": request.state.request_id,
                "fields": error.fields,
            },
            media_type="application/problem+json",
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "type": "https://jester.local/problems/validation_error",
                "status": 422,
                "code": "validation_error",
                "title": "Trainer request validation failed",
                "detail": "One or more request fields are invalid",
                "requestId": request.state.request_id,
                "fields": {"errors": error.errors()},
            },
            media_type="application/problem+json",
        )

    def auth(request: Request, authorization: str | None = Header(default=None, alias="Authorization")) -> None:
        if require_managed_session:
            # The managed middleware below validates every non-health route;
            # never fall back to the legacy administrative token there.
            return
        expected = f"Bearer {trainer.token}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise AuthenticationError()

    secured = [Depends(auth)]

    def authorize_managed_channel(request: Request, *, operation: str, campaign_id: str | None = None) -> None:
        if not require_managed_session:
            return
        if managed_channel_guard is None:
            raise TrainerError("managed channel session is not authorized", code="channel_session_denied", status=401)
        if campaign_id is not None:
            allowed = getattr(managed_channel_guard, "allowed_campaign_ids", None)
            if not allowed or campaign_id not in allowed:
                raise TrainerError("managed channel session is not authorized", code="channel_session_denied", status=403)
        try:
            managed_channel_guard.authorize(request.headers, operation=operation, method=request.method)
        except ChannelSessionDenied as exc:
            raise TrainerError("managed channel session is not authorized", code="channel_session_denied", status=401) from exc

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return trainer.health()

    @app.get("/v1/pool", dependencies=secured)
    def pool() -> dict[str, Any]:
        return trainer.pool_status()

    @app.get("/v1/workers", dependencies=secured)
    def workers() -> dict[str, Any]:
        return {"items": trainer.pool.status()}

    @app.get("/v1/workers/{worker_id}", dependencies=secured)
    def worker(worker_id: str) -> dict[str, Any]:
        return trainer.worker(worker_id)

    @app.get("/v1/pool/metrics", dependencies=secured)
    def pool_metrics() -> dict[str, Any]:
        return trainer.pool_metrics()

    @app.get("/v1/datasets", dependencies=secured)
    def datasets() -> dict[str, Any]:
        return {"items": trainer.list_datasets()}

    @app.post("/v1/datasets", status_code=201, dependencies=secured)
    def create_dataset(
        manifest: DatasetManifest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return trainer.idempotent(
            "datasets.create",
            idempotency_key,
            _idempotency_payload(manifest),
            lambda: trainer.put_dataset(manifest),
        )

    @app.get("/v1/datasets/{dataset_id}", dependencies=secured)
    def dataset(dataset_id: str) -> dict[str, Any]:
        return trainer.get_dataset(dataset_id).api_dict()

    @app.post("/v1/datasets/{dataset_id}/validate", dependencies=secured)
    def validate_dataset(dataset_id: str) -> dict[str, Any]:
        return trainer.validate_dataset(dataset_id)

    @app.post("/v1/datasets/{dataset_id}/shard-plan", dependencies=secured)
    def shard_plan(
        dataset_id: str, payload: dict[str, Any] = Body(default_factory=dict)
    ) -> dict[str, Any]:
        workers = payload.get("workerIds")
        if workers is not None and (
            not isinstance(workers, list)
            or not all(isinstance(item, str) for item in workers)
        ):
            raise TrainerError(
                "workerIds must be an array of strings",
                code="validation_error",
                status=422,
            )
        return trainer.dataset_shard_plan(
            dataset_id,
            split=str(payload.get("split", "train")),
            workers=workers,
            seed=int(payload.get("seed", 42)),
        )

    @app.get("/v1/campaigns", dependencies=secured)
    def campaigns() -> dict[str, Any]:
        return {"items": trainer.list_campaigns()}

    @app.post("/v1/campaigns", status_code=201, dependencies=secured)
    def create_campaign(
        campaign: Campaign,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        authorize_managed_channel(request, operation="trainer.campaign:create")
        return trainer.idempotent(
            "campaigns.create",
            idempotency_key,
            _idempotency_payload(campaign),
            lambda: trainer.put_campaign(campaign),
        )

    @app.get("/v1/campaigns/{campaign_id}", dependencies=secured)
    def campaign(campaign_id: str, request: Request) -> dict[str, Any]:
        authorize_managed_channel(request, operation="trainer.campaign:status", campaign_id=campaign_id)
        return trainer.get_campaign(campaign_id).api_dict()

    @app.patch("/v1/campaigns/{campaign_id}", dependencies=secured)
    def patch_campaign(
        campaign_id: str,
        patch: dict[str, Any] = Body(...),
        if_match: str | None = Header(default=None, alias="If-Match"),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        generation = _generation(if_match)
        return trainer.idempotent(
            f"campaigns.patch:{campaign_id}",
            idempotency_key,
            {"patch": patch, "generation": generation},
            lambda: trainer.patch_campaign(campaign_id, patch, generation),
        )

    @app.delete("/v1/campaigns/{campaign_id}", dependencies=secured)
    def delete_campaign(
        campaign_id: str,
        if_match: str | None = Header(default=None, alias="If-Match"),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        generation = _generation(if_match)
        return trainer.idempotent(
            f"campaigns.delete:{campaign_id}",
            idempotency_key,
            {"generation": generation},
            lambda: trainer.delete_campaign(
                campaign_id, expected_generation=generation
            ),
        )

    @app.post("/v1/campaigns/{campaign_id}/plan", dependencies=secured)
    def plan_campaign(
        campaign_id: str,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return trainer.idempotent(
            f"campaigns.plan:{campaign_id}",
            idempotency_key,
            {},
            lambda: trainer.plan_campaign(campaign_id),
        )

    def campaign_action(
        campaign_id: str,
        action: str,
        payload: dict[str, Any],
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        actions = {
            "start": trainer.start_campaign,
            "pause": trainer.pause_campaign,
            "resume": trainer.resume_campaign,
            "cancel": trainer.cancel_campaign,
        }
        return trainer.idempotent(
            f"campaigns.{action}:{campaign_id}",
            idempotency_key,
            payload,
            lambda: actions[action](campaign_id),
        )

    @app.post("/v1/campaigns/{campaign_id}/actions/start", status_code=202, dependencies=secured)
    def start_campaign(
        campaign_id: str,
        payload: dict[str, Any] = Body(default_factory=dict),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return campaign_action(campaign_id, "start", payload, idempotency_key)

    @app.post("/v1/campaigns/{campaign_id}/actions/pause", status_code=202, dependencies=secured)
    def pause_campaign(
        campaign_id: str,
        payload: dict[str, Any] = Body(default_factory=dict),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return campaign_action(campaign_id, "pause", payload, idempotency_key)

    @app.post("/v1/campaigns/{campaign_id}/actions/resume", status_code=202, dependencies=secured)
    def resume_campaign(
        campaign_id: str,
        payload: dict[str, Any] = Body(default_factory=dict),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return campaign_action(campaign_id, "resume", payload, idempotency_key)

    @app.post("/v1/campaigns/{campaign_id}/actions/cancel", status_code=202, dependencies=secured)
    def cancel_campaign(
        campaign_id: str,
        payload: dict[str, Any] = Body(default_factory=dict),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return campaign_action(campaign_id, "cancel", payload, idempotency_key)

    @app.get("/v1/campaigns/{campaign_id}/jobs", dependencies=secured)
    def campaign_jobs(campaign_id: str) -> dict[str, Any]:
        trainer.get_campaign(campaign_id)
        return {"items": trainer.campaign_jobs(campaign_id)}

    @app.get("/v1/campaigns/{campaign_id}/metrics", dependencies=secured)
    def campaign_metrics(campaign_id: str) -> dict[str, Any]:
        return trainer.campaign_metrics(campaign_id)

    @app.get("/v1/campaigns/{campaign_id}/events", dependencies=secured)
    def campaign_events(
        campaign_id: str,
        after_sequence: int = Query(default=0, alias="afterSequence", ge=0),
    ) -> dict[str, Any]:
        return {
            "items": trainer.campaign_events(
                campaign_id, after_sequence=after_sequence
            )
        }

    @app.get("/v1/campaigns/{campaign_id}/results", dependencies=secured)
    def campaign_results(campaign_id: str) -> dict[str, Any]:
        return {"items": trainer.campaign_results(campaign_id)}

    @app.get("/v1/jobs/{job_id}", dependencies=secured)
    def job(job_id: str) -> dict[str, Any]:
        return trainer.get_job(job_id)

    @app.post("/v1/jobs/{job_id}/actions/retry", status_code=202, dependencies=secured)
    def retry_job(
        job_id: str,
        payload: dict[str, Any] = Body(default_factory=dict),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return trainer.idempotent(
            f"jobs.retry:{job_id}",
            idempotency_key,
            payload,
            lambda: trainer.retry_job(job_id),
        )

    @app.get("/v1/operations/{operation_id}", dependencies=secured)
    def operation(operation_id: str) -> dict[str, Any]:
        return trainer.get_operation(operation_id)

    @app.get("/v1/results", dependencies=secured)
    def results() -> dict[str, Any]:
        return {"items": trainer.list_results()}

    @app.get("/v1/results/{result_id}", dependencies=secured)
    def result(result_id: str) -> dict[str, Any]:
        return trainer.get_result(result_id)

    @app.post("/v1/results/{result_id}/actions/promote", dependencies=secured)
    def promote_result(result_id: str) -> dict[str, Any]:
        return trainer.promote_result(result_id)

    @app.post("/v1/results/{result_id}/actions/export", dependencies=secured)
    def export_result(result_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        destination = str(payload.get("destination", "")).strip()
        if not destination:
            raise TrainerError("destination is required", code="validation_error", status=422)
        return trainer.export_result(result_id, destination)

    @app.get("/v1/events/stream", dependencies=secured)
    async def event_stream(
        campaign_id: str | None = Query(default=None, alias="campaignId"),
        after_sequence: int = Query(default=0, alias="afterSequence", ge=0),
    ) -> StreamingResponse:
        async def generate() -> AsyncIterator[str]:
            sequence = after_sequence
            while True:
                items = await asyncio.to_thread(
                    trainer.store.list_events,
                    campaign_id=campaign_id,
                    after_sequence=sequence,
                )
                for item in items:
                    sequence = max(sequence, int(item.get("sequence", 0)))
                    yield f"id: {sequence}\nevent: {item.get('type', 'event')}\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
                if not items:
                    yield ": keepalive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app


def _generation(value: str | None) -> int:
    if value is None:
        raise TrainerError(
            "If-Match generation is required",
            code="if_match_required",
            status=428,
        )
    try:
        return int(value.strip().strip('"'))
    except ValueError as exc:
        raise TrainerError(
            "If-Match must contain an integer generation",
            code="invalid_if_match",
            status=422,
        ) from exc


def _idempotency_payload(model: DatasetManifest | Campaign) -> dict[str, Any]:
    """Hash the client-authored creation payload, not generated timestamps."""
    payload = model.api_dict()
    generated = {
        "generation": "generation",
        "created_at": "createdAt",
        "updated_at": "updatedAt",
    }
    for field_name, alias in generated.items():
        if field_name not in model.model_fields_set:
            payload.pop(alias, None)
    return payload
