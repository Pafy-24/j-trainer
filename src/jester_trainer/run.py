from __future__ import annotations

import os
import uvicorn

from .api import create_app
from .config import Settings
from .channel_sessions import ChannelSessionDenied, guard_from_managed_environment
from .service import TrainerService


def serve(host: str | None = None, port: int | None = None) -> None:
    settings = Settings.from_rm()
    service = TrainerService(settings)
    managed = os.environ.get("JESTER_RM_REQUIRED", "").lower() in {"1", "true", "yes", "on"}
    try:
        guard = guard_from_managed_environment() if managed else None
    except ChannelSessionDenied as exc:
        service.close()
        raise RuntimeError("managed channel context is required") from exc
    uvicorn.run(
        create_app(service, managed_channel_guard=guard, require_managed_session=managed),
        host=host or settings.api_host,
        port=port or settings.api_port,
        log_level="info",
    )
