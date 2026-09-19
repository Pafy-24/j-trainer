from __future__ import annotations

import uvicorn

from .api import create_app
from .config import Settings
from .service import TrainerService


def serve(host: str | None = None, port: int | None = None) -> None:
    settings = Settings.from_rm()
    service = TrainerService(settings)
    uvicorn.run(
        create_app(service),
        host=host or settings.api_host,
        port=port or settings.api_port,
        log_level="info",
    )
