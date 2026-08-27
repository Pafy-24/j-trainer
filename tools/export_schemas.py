from __future__ import annotations

import json
import tempfile
from pathlib import Path

from jester_trainer.api import create_app
from jester_trainer.config import Settings
from jester_trainer.models import (
    Campaign,
    DatasetManifest,
    TrainingAttempt,
    TrainingJob,
    TrainingResult,
    TrainingRound,
)
from jester_trainer.service import TrainerService


MODELS = {
    "dataset-manifest.schema.json": DatasetManifest,
    "campaign.schema.json": Campaign,
    "training-job.schema.json": TrainingJob,
    "training-attempt.schema.json": TrainingAttempt,
    "training-round.schema.json": TrainingRound,
    "training-result.schema.json": TrainingResult,
}


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "schemas"
    root.mkdir(parents=True, exist_ok=True)
    for filename, model in MODELS.items():
        schema = model.model_json_schema(by_alias=True)
        schema["$id"] = f"https://jester.local/schemas/trainer/v1/{filename}"
        (root / filename).write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    with tempfile.TemporaryDirectory() as temporary:
        service = TrainerService(Settings.for_test(temporary))
        try:
            openapi = create_app(service).openapi()
        finally:
            service.close()
    (root / "openapi.json").write_text(
        json.dumps(openapi, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
