from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import typer

from .config import Settings
from .errors import TrainerError
from .models import Campaign, DatasetManifest
from .run import serve
from .util import atomic_write_json


app = typer.Typer(no_args_is_help=True, help="Jester Trainer, component 5/5")
dataset_app = typer.Typer(no_args_is_help=True)
campaign_app = typer.Typer(no_args_is_help=True)
pool_app = typer.Typer(no_args_is_help=True)
job_app = typer.Typer(no_args_is_help=True)
result_app = typer.Typer(no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")
app.add_typer(campaign_app, name="campaign")
app.add_typer(pool_app, name="pool")
app.add_typer(job_app, name="job")
app.add_typer(result_app, name="result")


class APIClient:
    def __init__(self, base: str, token: str):
        self.token = token
        self.base = base.rstrip("/")

    @classmethod
    def from_environment(cls) -> "APIClient":
        url = os.environ.get("JESTER_TRAINER_URL")
        token = os.environ.get("JESTER_TRAINER_TOKEN")
        token_file = os.environ.get("JESTER_TRAINER_TOKEN_FILE")
        if token is None and token_file:
            path = Path(token_file).expanduser().resolve()
            try:
                token = path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise TrainerError(
                    f"Cannot read Trainer token file {path}: {exc}",
                    code="trainer_token_unavailable",
                    status=409,
                ) from exc
        if url and token:
            return cls(url, token)
        if bool(url) != bool(token):
            raise TrainerError(
                "JESTER_TRAINER_URL and a Trainer token must be configured together",
                code="trainer_client_configuration_incomplete",
                status=409,
            )
        settings = Settings.from_rm()
        return cls(
            f"http://{settings.api_host}:{settings.api_port}",
            settings.initialize(),
        )

    def call(
        self,
        method: str,
        path: str,
        payload: Any = None,
        *,
        generation: int | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if generation is not None:
            headers["If-Match"] = str(generation)
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        request = urllib.request.Request(
            self.base + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                problem = json.loads(raw)
            except json.JSONDecodeError:
                problem = {"detail": raw, "code": "http_error"}
            raise TrainerError(
                str(problem.get("detail", exc.reason)),
                code=str(problem.get("code", "http_error")),
                status=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise TrainerError(
                f"Trainer API unavailable at {self.base}: {exc.reason}",
                code="trainer_unavailable",
                status=503,
            ) from exc


@app.command("serve")
def serve_command(host: str = "127.0.0.1", port: int = 8140) -> None:
    serve(host, port)


@dataset_app.command("register")
def dataset_register(manifest: Path) -> None:
    payload = DatasetManifest.model_validate(_load(manifest)).api_dict()
    _print(
        _api().call(
            "POST",
            "/v1/datasets",
            payload,
            idempotency_key=f"dataset:{payload['metadata']['id']}",
        )
    )


@dataset_app.command("validate")
def dataset_validate(dataset_id: str) -> None:
    _print(_api().call("POST", f"/v1/datasets/{_quote(dataset_id)}/validate", {}))


@dataset_app.command("shard-plan")
def dataset_shard_plan(
    dataset_id: str,
    split: str = "train",
    worker: list[str] = typer.Option([], "--worker"),
    seed: int = 42,
) -> None:
    _print(
        _api().call(
            "POST",
            f"/v1/datasets/{_quote(dataset_id)}/shard-plan",
            {"split": split, "workerIds": worker or None, "seed": seed},
        )
    )


@dataset_app.command("inspect")
def dataset_inspect(dataset_id: str) -> None:
    _print(_api().call("GET", f"/v1/datasets/{_quote(dataset_id)}"))


@campaign_app.command("apply")
def campaign_apply(spec: Path) -> None:
    payload = Campaign.model_validate(_load(spec)).api_dict()
    campaign_id = payload["metadata"]["id"]
    client = _api()
    try:
        current = client.call("GET", f"/v1/campaigns/{_quote(campaign_id)}")
    except TrainerError as exc:
        if exc.status != 404:
            raise
        result = client.call(
            "POST",
            "/v1/campaigns",
            payload,
            idempotency_key=f"campaign-create:{campaign_id}",
        )
    else:
        result = client.call(
            "PATCH",
            f"/v1/campaigns/{_quote(campaign_id)}",
            {"metadata": payload["metadata"], "spec": payload["spec"]},
            generation=int(current["generation"]),
            idempotency_key=f"campaign-apply:{campaign_id}:{current['generation']}",
        )
    _print(result)


@campaign_app.command("plan")
def campaign_plan(campaign_id: str) -> None:
    _print(_api().call("POST", f"/v1/campaigns/{_quote(campaign_id)}/plan", {}))


def _campaign_action(campaign_id: str, action: str) -> None:
    _print(
        _api().call(
            "POST",
            f"/v1/campaigns/{_quote(campaign_id)}/actions/{action}",
            {},
            idempotency_key=f"campaign-{action}:{campaign_id}:{int(time.time())}",
        )
    )


@campaign_app.command("start")
def campaign_start(campaign_id: str) -> None:
    _campaign_action(campaign_id, "start")


@campaign_app.command("pause")
def campaign_pause(campaign_id: str) -> None:
    _campaign_action(campaign_id, "pause")


@campaign_app.command("resume")
def campaign_resume(campaign_id: str) -> None:
    _campaign_action(campaign_id, "resume")


@campaign_app.command("cancel")
def campaign_cancel(campaign_id: str) -> None:
    _campaign_action(campaign_id, "cancel")


@campaign_app.command("status")
def campaign_status(campaign_id: str) -> None:
    _print(_api().call("GET", f"/v1/campaigns/{_quote(campaign_id)}"))


@campaign_app.command("inspect")
def campaign_inspect(campaign_id: str) -> None:
    client = _api()
    _print(
        {
            "campaign": client.call("GET", f"/v1/campaigns/{_quote(campaign_id)}"),
            "jobs": client.call("GET", f"/v1/campaigns/{_quote(campaign_id)}/jobs"),
            "results": client.call("GET", f"/v1/campaigns/{_quote(campaign_id)}/results"),
        }
    )


@pool_app.command("status")
def pool_status() -> None:
    _print(_api().call("GET", "/v1/pool"))


@pool_app.command("workers")
def pool_workers() -> None:
    _print(_api().call("GET", "/v1/workers"))


@pool_app.command("metrics")
def pool_metrics() -> None:
    _print(_api().call("GET", "/v1/pool/metrics"))


@pool_app.command("watch")
def pool_watch(interval: float = 1.0) -> None:
    while True:
        _print(_api().call("GET", "/v1/pool/metrics"))
        time.sleep(max(0.2, interval))


@job_app.command("list")
def job_list(campaign_id: str) -> None:
    _print(_api().call("GET", f"/v1/campaigns/{_quote(campaign_id)}/jobs"))


@job_app.command("inspect")
def job_inspect(job_id: str) -> None:
    _print(_api().call("GET", f"/v1/jobs/{_quote(job_id)}"))


@job_app.command("retry")
def job_retry(job_id: str) -> None:
    _print(_api().call("POST", f"/v1/jobs/{_quote(job_id)}/actions/retry", {}))


@result_app.command("list")
def result_list(campaign_id: str | None = None) -> None:
    path = f"/v1/campaigns/{_quote(campaign_id)}/results" if campaign_id else "/v1/results"
    _print(_api().call("GET", path))


@result_app.command("inspect")
def result_inspect(result_id: str) -> None:
    _print(_api().call("GET", f"/v1/results/{_quote(result_id)}"))


@result_app.command("promote")
def result_promote(result_id: str) -> None:
    _print(_api().call("POST", f"/v1/results/{_quote(result_id)}/actions/promote", {}))


@result_app.command("export")
def result_export(result_id: str, destination: Path) -> None:
    _print(
        _api().call(
            "POST",
            f"/v1/results/{_quote(result_id)}/actions/export",
            {"destination": str(destination.resolve())},
        )
    )


@app.command("migrate-mi-config")
def migrate_mi_config(
    source: Path,
    output: Path,
    topology_id: str = typer.Option(..., "--topology-id"),
    dataset_id: str = typer.Option(..., "--dataset-id"),
    campaign_id: str | None = typer.Option(None, "--campaign-id"),
) -> None:
    config = _load(source)
    if config.get("schemaVersion") != "mi-training-v1":
        raise TrainerError(
            "Input must be an mi-training-v1 document",
            code="invalid_mi_config",
            status=422,
        )
    base = config.setdefault("baseModel", {})
    previous_device = base.get("device")
    base["device"] = "managed"
    optimizer = config.pop("optimizer", {})
    scheduler = config.pop("scheduler", {})
    training = config.get("training", {})
    checkpoint = config.get("checkpoint", {})
    facet = config.get("facet", {})
    identifier = campaign_id or f"train-{facet.get('id', 'facet')}"
    payload = {
        "apiVersion": "jester.trainer/v1",
        "kind": "Campaign",
        "metadata": {"id": identifier, "displayName": f"Migrated {facet.get('id', 'facet')}"},
        "spec": {
            "topologyId": topology_id,
            "datasets": [{"datasetId": dataset_id, "split": "train"}],
            "facets": [{"id": facet.get("id", "facet"), "miConfig": config}],
            "mode": "parameter-server",
            "cohort": {},
            "optimizer": {
                "type": optimizer.get("type", "adamw"),
                "learningRate": optimizer.get("learningRate", 0.0002),
                "weightDecay": optimizer.get("weightDecay", 0.0),
                "betas": optimizer.get("betas", [0.9, 0.999]),
                "eps": optimizer.get("eps", 1e-8),
                "momentum": optimizer.get("momentum", 0.0),
            },
            "scheduler": {
                "type": scheduler.get("type", "constant"),
                "warmupSteps": scheduler.get("warmupSteps", 0),
            },
            "training": {
                "maxSteps": training.get("maxSteps", 1),
                "examplesPerWorker": training.get("batchSize", 1),
                "batchSize": training.get("batchSize", 1),
                "precision": training.get("precision", "float32"),
                "maxGradNorm": training.get("maxGradNorm", 1.0),
                "seed": training.get("seed", 42),
                "transportDtype": "float32",
                "compression": "none",
            },
            "checkpoint": {
                "outputDirectory": checkpoint.get("outputDirectory", "artifacts/trainer-results"),
                "saveEveryRounds": checkpoint.get("saveEverySteps", 1) or 1,
                "maxRetries": 1,
                "resumeFrom": checkpoint.get("resumeFrom"),
            },
        },
    }
    validated = Campaign.model_validate(payload).api_dict()
    atomic_write_json(output.resolve(), validated)
    _print(
        {
            "migrated": True,
            "source": str(source.resolve()),
            "output": str(output.resolve()),
            "previousDevice": previous_device,
            "device": "managed",
            "removedFromMI": ["optimizer", "scheduler", "job lifecycle"],
        }
    )


def _api() -> APIClient:
    return APIClient.from_environment()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainerError(
            f"Invalid JSON {path}: {exc.msg}", code="invalid_json", status=422
        ) from exc
    if not isinstance(value, dict):
        raise TrainerError("JSON root must be an object", code="invalid_json", status=422)
    return value


def _quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def _print(value: Any) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> int:
    try:
        app()
        return 0
    except TrainerError as exc:
        typer.echo(
            json.dumps(
                {"error": {"code": exc.code, "detail": str(exc), "status": exc.status}},
                ensure_ascii=False,
            ),
            err=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
