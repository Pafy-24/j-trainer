from __future__ import annotations

from typing import Any


class TrainerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "trainer_error",
        status: int = 400,
        fields: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.status = status
        self.fields = fields or {}


class NotFoundError(TrainerError):
    def __init__(self, kind: str, entity_id: str):
        super().__init__(
            f"{kind} {entity_id!r} was not found",
            code=f"{kind.replace('-', '_')}_not_found",
            status=404,
        )


class ConflictError(TrainerError):
    def __init__(self, message: str, *, code: str = "conflict"):
        super().__init__(message, code=code, status=409)


class AuthenticationError(TrainerError):
    def __init__(self):
        super().__init__("Invalid or missing Trainer bearer token", code="unauthorized", status=401)
