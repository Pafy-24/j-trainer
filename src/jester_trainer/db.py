from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .errors import ConflictError, NotFoundError
from .util import canonical_json, utc_now


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(
            self.path, timeout=30, isolation_level=None, check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self._migrate()

    def _migrate(self) -> None:
        with self._lock, self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    kind TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(kind, entity_id)
                );
                CREATE TABLE IF NOT EXISTS operations (
                    operation_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    timestamp TEXT NOT NULL,
                    campaign_id TEXT,
                    job_id TEXT,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_campaign_idx
                    ON events(campaign_id, sequence);
                CREATE TABLE IF NOT EXISTS campaign_owners (
                    campaign_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    deployment_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS idempotency (
                    scope TEXT NOT NULL,
                    idem_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(scope, idem_key)
                );
                """
            )

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def bind_campaign_owner(self, campaign_id: str, *, project_id: str, deployment_id: str) -> None:
        """Atomically reserve immutable managed ownership before campaign mutation."""
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                row = self.connection.execute("SELECT project_id, deployment_id FROM campaign_owners WHERE campaign_id=?", (campaign_id,)).fetchone()
                if row is not None and (row["project_id"], row["deployment_id"]) != (project_id, deployment_id):
                    raise ConflictError("campaign is owned by another deployment", code="campaign_owner_conflict")
                if row is None:
                    self.connection.execute("INSERT INTO campaign_owners(campaign_id, project_id, deployment_id, created_at) VALUES(?,?,?,?)", (campaign_id, project_id, deployment_id, utc_now()))
                self.connection.commit()
            except BaseException:
                self.connection.rollback(); raise

    def campaign_owner(self, campaign_id: str) -> tuple[str, str] | None:
        with self._lock:
            row=self.connection.execute("SELECT project_id, deployment_id FROM campaign_owners WHERE campaign_id=?", (campaign_id,)).fetchone()
        return (str(row["project_id"]), str(row["deployment_id"])) if row else None

    def put_entity(
        self,
        kind: str,
        entity_id: str,
        payload: dict[str, Any],
        *,
        expected_generation: int | None = None,
        archived: bool = False,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._lock, self.connection:
            row = self.connection.execute(
                "SELECT generation, created_at FROM entities WHERE kind=? AND entity_id=?",
                (kind, entity_id),
            ).fetchone()
            if row is None:
                if expected_generation not in {None, 0}:
                    raise ConflictError(
                        f"{kind} {entity_id} does not exist at generation {expected_generation}",
                        code="generation_conflict",
                    )
                generation = 1
                created = now
            else:
                current = int(row["generation"])
                if expected_generation is not None and current != expected_generation:
                    raise ConflictError(
                        f"{kind} {entity_id} generation is {current}, not {expected_generation}",
                        code="generation_conflict",
                    )
                generation = current + 1
                created = str(row["created_at"])
            stored = dict(payload)
            stored.update(
                {"generation": generation, "createdAt": created, "updatedAt": now}
            )
            self.connection.execute(
                """
                INSERT INTO entities(kind, entity_id, generation, payload, archived, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(kind, entity_id) DO UPDATE SET
                    generation=excluded.generation,
                    payload=excluded.payload,
                    archived=excluded.archived,
                    updated_at=excluded.updated_at
                """,
                (
                    kind,
                    entity_id,
                    generation,
                    canonical_json(stored),
                    int(archived),
                    created,
                    now,
                ),
            )
            return stored

    def get_entity(
        self, kind: str, entity_id: str, *, include_archived: bool = False
    ) -> dict[str, Any]:
        with self._lock:
            row = self.connection.execute(
                "SELECT payload, archived FROM entities WHERE kind=? AND entity_id=?",
                (kind, entity_id),
            ).fetchone()
        if row is None or (row["archived"] and not include_archived):
            raise NotFoundError(kind, entity_id)
        return json.loads(row["payload"])

    def list_entities(
        self, kind: str, *, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        query = "SELECT payload FROM entities WHERE kind=?"
        if not include_archived:
            query += " AND archived=0"
        query += " ORDER BY entity_id"
        with self._lock:
            rows = self.connection.execute(query, (kind,)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def archive_entity(
        self, kind: str, entity_id: str, *, expected_generation: int
    ) -> dict[str, Any]:
        current = self.get_entity(kind, entity_id)
        return self.put_entity(
            kind,
            entity_id,
            current,
            expected_generation=expected_generation,
            archived=True,
        )

    def put_operation(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self._lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO operations(operation_id, state, payload, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(operation_id) DO UPDATE SET
                    state=excluded.state, payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (
                    payload["operationId"],
                    payload["state"],
                    canonical_json(payload),
                    payload.get("createdAt", now),
                    now,
                ),
            )
        return payload

    def get_operation(self, operation_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.connection.execute(
                "SELECT payload FROM operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError("operation", operation_id)
        return json.loads(row["payload"])

    def add_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO events(event_id, timestamp, campaign_id, job_id, payload)
                VALUES(?, ?, ?, ?, ?)
                """,
                (
                    payload["eventId"],
                    payload["timestamp"],
                    payload.get("campaignId"),
                    payload.get("jobId"),
                    canonical_json(payload),
                ),
            )
            result = {**payload, "sequence": int(cursor.lastrowid)}
            self.connection.execute(
                "UPDATE events SET payload=? WHERE sequence=?",
                (canonical_json(result), cursor.lastrowid),
            )
            return result

    def list_events(
        self,
        *,
        after_sequence: int = 0,
        campaign_id: str | None = None,
        job_id: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        clauses = ["sequence>?"]
        values: list[Any] = [after_sequence]
        if campaign_id:
            clauses.append("campaign_id=?")
            values.append(campaign_id)
        if job_id:
            clauses.append("job_id=?")
            values.append(job_id)
        values.append(max(1, min(limit, 100000)))
        with self._lock:
            rows = self.connection.execute(
                "SELECT payload FROM events WHERE "
                + " AND ".join(clauses)
                + " ORDER BY sequence LIMIT ?",
                values,
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def get_idempotent(
        self, scope: str, key: str, request_hash: str
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT request_hash, response FROM idempotency WHERE scope=? AND idem_key=?",
                (scope, key),
            ).fetchone()
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise ConflictError(
                "Idempotency-Key was already used with another request",
                code="idempotency_conflict",
            )
        return json.loads(row["response"])

    def put_idempotent(
        self,
        scope: str,
        key: str,
        request_hash: str,
        response: dict[str, Any],
    ) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT INTO idempotency(scope, idem_key, request_hash, response, created_at) VALUES(?, ?, ?, ?, ?)",
                (scope, key, request_hash, canonical_json(response), utc_now()),
            )
