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
                CREATE TABLE IF NOT EXISTS campaign_start_reservations (
                    campaign_id TEXT PRIMARY KEY,
                    operation_id TEXT NOT NULL UNIQUE,
                    controller_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS campaign_cancel_reservations (
                    campaign_id TEXT PRIMARY KEY,
                    cancel_operation_id TEXT NOT NULL UNIQUE,
                    start_operation_id TEXT NOT NULL,
                    controller_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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
        """Atomically reserve immutable managed ownership before campaign mutation.

        A managed caller cannot adopt an existing legacy campaign without an
        explicit owner record. A reservation remains after a later create
        failure so its original owner can retry while another deployment
        cannot take over the campaign id.
        """
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                row = self.connection.execute("SELECT project_id, deployment_id FROM campaign_owners WHERE campaign_id=?", (campaign_id,)).fetchone()
                if row is not None and (row["project_id"], row["deployment_id"]) != (project_id, deployment_id):
                    raise ConflictError("campaign is owned by another deployment", code="campaign_owner_conflict")
                if row is None:
                    legacy = self.connection.execute(
                        "SELECT 1 FROM entities WHERE kind='campaign' AND entity_id=? AND archived=0",
                        (campaign_id,),
                    ).fetchone()
                    if legacy is not None:
                        raise ConflictError(
                            "existing unmanaged campaign cannot be claimed by a managed deployment",
                            code="campaign_owner_missing",
                        )
                    self.connection.execute("INSERT INTO campaign_owners(campaign_id, project_id, deployment_id, created_at) VALUES(?,?,?,?)", (campaign_id, project_id, deployment_id, utc_now()))
                self.connection.commit()
            except BaseException:
                self.connection.rollback(); raise

    def campaign_owner(self, campaign_id: str) -> tuple[str, str] | None:
        with self._lock:
            row=self.connection.execute("SELECT project_id, deployment_id FROM campaign_owners WHERE campaign_id=?", (campaign_id,)).fetchone()
        return (str(row["project_id"]), str(row["deployment_id"])) if row else None

    def reserve_campaign_start(
        self,
        campaign_id: str,
        operation: dict[str, Any],
        *,
        controller_id: str,
    ) -> tuple[dict[str, Any], bool]:
        """Reserve one campaign start and its operation in one SQLite write.

        The transaction ends before planning or thread launch. The boolean is
        true only for the caller that may continue toward that launch.
        """
        operation_id = str(operation["operationId"])
        now = utc_now()
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                campaign_row = self.connection.execute(
                    "SELECT payload FROM entities WHERE kind='campaign' AND entity_id=? AND archived=0",
                    (campaign_id,),
                ).fetchone()
                if campaign_row is None:
                    raise NotFoundError("campaign", campaign_id)
                campaign = json.loads(campaign_row["payload"])
                existing_operation = self.connection.execute(
                    "SELECT payload FROM operations WHERE operation_id=?",
                    (operation_id,),
                ).fetchone()
                if existing_operation is not None:
                    payload = json.loads(existing_operation["payload"])
                    if (
                        payload.get("action") != "campaign.start"
                        or payload.get("targetId") != campaign_id
                    ):
                        raise ConflictError(
                            "Operation identifier belongs to another campaign",
                            code="operation_identity_conflict",
                        )
                    self.connection.commit()
                    return payload, False

                reservation = self.connection.execute(
                    "SELECT operation_id, phase FROM campaign_start_reservations WHERE campaign_id=?",
                    (campaign_id,),
                ).fetchone()
                if reservation is not None:
                    previous = self.connection.execute(
                        "SELECT payload FROM operations WHERE operation_id=?",
                        (reservation["operation_id"],),
                    ).fetchone()
                    previous_payload = (
                        json.loads(previous["payload"]) if previous is not None else None
                    )
                    if reservation["operation_id"] == operation_id:
                        if previous_payload is None:
                            raise ConflictError(
                                "Campaign start reservation has no operation receipt",
                                code="campaign_start_receipt_missing",
                            )
                        self.connection.commit()
                        return previous_payload, False
                    if reservation["phase"] != "terminal":
                        raise ConflictError(
                            "Campaign already has a reserved start operation",
                            code="campaign_start_reserved",
                        )

                if campaign.get("state") in {"running", "recovering"}:
                    raise ConflictError(
                        "Campaign is already running",
                        code="campaign_running",
                    )
                self.connection.execute(
                    """
                    INSERT INTO operations(operation_id, state, payload, created_at, updated_at)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (
                        operation_id,
                        operation["state"],
                        canonical_json(operation),
                        operation.get("createdAt", now),
                        now,
                    ),
                )
                self.connection.execute(
                    """
                    INSERT INTO campaign_start_reservations(
                        campaign_id, operation_id, controller_id, phase, created_at, updated_at
                    ) VALUES(?, ?, ?, 'reserved', ?, ?)
                    ON CONFLICT(campaign_id) DO UPDATE SET
                        operation_id=excluded.operation_id,
                        controller_id=excluded.controller_id,
                        phase=excluded.phase,
                        created_at=excluded.created_at,
                        updated_at=excluded.updated_at
                    """,
                    (campaign_id, operation_id, controller_id, now, now),
                )
                self.connection.commit()
                return operation, True
            except BaseException:
                self.connection.rollback()
                raise

    def require_campaign_start_reservation(
        self, campaign_id: str, operation_id: str, *, controller_id: str
    ) -> None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT operation_id, controller_id, phase
                FROM campaign_start_reservations WHERE campaign_id=?
                """,
                (campaign_id,),
            ).fetchone()
        if row is None or (
            row["operation_id"], row["controller_id"], row["phase"]
        ) != (operation_id, controller_id, "reserved"):
            raise ConflictError(
                "Campaign start reservation is no longer owned by this controller",
                code="campaign_start_reservation_lost",
            )

    def reserve_campaign_cancel(
        self,
        campaign_id: str,
        start_operation_id: str,
        operation: dict[str, Any],
        *,
        controller_id: str,
    ) -> tuple[dict[str, Any], bool]:
        """Persist an exact cancel intent before the in-process signal is set."""
        cancel_operation_id = str(operation["operationId"])
        now = utc_now()
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                campaign_row = self.connection.execute(
                    "SELECT generation,payload FROM entities WHERE kind='campaign' AND entity_id=? AND archived=0",
                    (campaign_id,),
                ).fetchone()
                if campaign_row is None:
                    raise NotFoundError("campaign", campaign_id)
                campaign = json.loads(campaign_row["payload"])

                start_reservation = self.connection.execute(
                    "SELECT operation_id,controller_id,phase FROM campaign_start_reservations WHERE campaign_id=?",
                    (campaign_id,),
                ).fetchone()
                start_row = self.connection.execute(
                    "SELECT payload FROM operations WHERE operation_id=?",
                    (start_operation_id,),
                ).fetchone()
                existing = self.connection.execute(
                    "SELECT cancel_operation_id,start_operation_id,controller_id,phase FROM campaign_cancel_reservations WHERE campaign_id=?",
                    (campaign_id,),
                ).fetchone()
                if existing is not None:
                    previous = self.connection.execute(
                        "SELECT payload FROM operations WHERE operation_id=?",
                        (existing["cancel_operation_id"],),
                    ).fetchone()
                    if (
                        existing["cancel_operation_id"] == cancel_operation_id
                        and existing["start_operation_id"] == start_operation_id
                        and existing["phase"] == "terminal"
                        and previous is not None
                    ):
                        self.connection.commit()
                        return json.loads(previous["payload"]), False

                if start_reservation is None or start_row is None or (
                    start_reservation["operation_id"] != start_operation_id
                    or start_reservation["controller_id"] != controller_id
                ):
                    raise ConflictError(
                        "Campaign start is not actively owned by this controller",
                        code="campaign_cancel_start_not_active",
                    )
                start = json.loads(start_row["payload"])
                if start.get("action") != "campaign.start" or start.get("targetId") != campaign_id:
                    raise ConflictError(
                        "Cancel does not identify the campaign start operation",
                        code="campaign_cancel_start_mismatch",
                    )
                if start.get("state") in {"succeeded", "failed", "cancelled"} or campaign.get("state") in {
                    "succeeded", "failed", "cancelled"
                }:
                    raise ConflictError(
                        "Campaign start is already terminal",
                        code="campaign_cancel_too_late",
                    )
                if start_reservation["phase"] not in {"launch-committed", "running"}:
                    raise ConflictError(
                        "Campaign start is not actively owned by this controller",
                        code="campaign_cancel_start_not_active",
                    )

                prior_operation = self.connection.execute(
                    "SELECT payload FROM operations WHERE operation_id=?",
                    (cancel_operation_id,),
                ).fetchone()
                if prior_operation is not None:
                    prior = json.loads(prior_operation["payload"])
                    if (
                        prior.get("action") != "campaign.cancel"
                        or prior.get("targetId") != campaign_id
                        or (prior.get("result") or {}).get("startOperationId") != start_operation_id
                    ):
                        raise ConflictError(
                            "Operation identifier belongs to another cancel intent",
                            code="operation_identity_conflict",
                        )
                    self.connection.commit()
                    return prior, False

                if existing is not None:
                    previous = self.connection.execute(
                        "SELECT payload FROM operations WHERE operation_id=?",
                        (existing["cancel_operation_id"],),
                    ).fetchone()
                    if (
                        existing["cancel_operation_id"] == cancel_operation_id
                        and existing["start_operation_id"] == start_operation_id
                        and previous is not None
                    ):
                        self.connection.commit()
                        return json.loads(previous["payload"]), False
                    raise ConflictError(
                        "Campaign already has a different cancel operation",
                        code="campaign_cancel_reserved",
                    )

                stored_operation = dict(operation)
                stored_operation["result"] = {"startOperationId": start_operation_id}
                self.connection.execute(
                    "INSERT INTO operations(operation_id,state,payload,created_at,updated_at) VALUES(?,?,?,?,?)",
                    (
                        cancel_operation_id,
                        stored_operation["state"],
                        canonical_json(stored_operation),
                        stored_operation.get("createdAt", now),
                        now,
                    ),
                )
                self.connection.execute(
                    """INSERT INTO campaign_cancel_reservations(
                           campaign_id,cancel_operation_id,start_operation_id,controller_id,phase,created_at,updated_at
                       ) VALUES(?,?,?,?, 'requested', ?, ?)""",
                    (campaign_id, cancel_operation_id, start_operation_id, controller_id, now, now),
                )
                campaign.update(
                    {
                        "state": "cancelling",
                        "generation": int(campaign_row["generation"]) + 1,
                        "updatedAt": now,
                    }
                )
                self.connection.execute(
                    "UPDATE entities SET generation=?,payload=?,updated_at=? WHERE kind='campaign' AND entity_id=?",
                    (campaign["generation"], canonical_json(campaign), now, campaign_id),
                )
                self.connection.commit()
                return stored_operation, True
            except BaseException:
                self.connection.rollback()
                raise

    def mark_campaign_cancel_signal_committed(
        self, campaign_id: str, cancel_operation_id: str, *, controller_id: str
    ) -> dict[str, Any]:
        now = utc_now()
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                reservation = self.connection.execute(
                    """SELECT phase FROM campaign_cancel_reservations
                       WHERE campaign_id=? AND cancel_operation_id=? AND controller_id=?""",
                    (campaign_id, cancel_operation_id, controller_id),
                ).fetchone()
                row = self.connection.execute(
                    "SELECT payload FROM operations WHERE operation_id=?",
                    (cancel_operation_id,),
                ).fetchone()
                if reservation is None or row is None:
                    raise ConflictError(
                        "Campaign cancel reservation is no longer signalable",
                        code="campaign_cancel_reservation_lost",
                    )
                operation = json.loads(row["payload"])
                if reservation["phase"] == "terminal":
                    start_id = (operation.get("result") or {}).get("startOperationId")
                    start_row = self.connection.execute(
                        "SELECT payload FROM operations WHERE operation_id=?",
                        (start_id,),
                    ).fetchone()
                    campaign_row = self.connection.execute(
                        "SELECT payload FROM entities WHERE kind='campaign' AND entity_id=?",
                        (campaign_id,),
                    ).fetchone()
                    start = json.loads(start_row["payload"]) if start_row is not None else {}
                    campaign = json.loads(campaign_row["payload"]) if campaign_row is not None else {}
                    if (
                        operation.get("state") == "succeeded"
                        and operation.get("action") == "campaign.cancel"
                        and operation.get("targetId") == campaign_id
                        and start.get("state") == "cancelled"
                        and campaign.get("state") == "cancelled"
                    ):
                        self.connection.commit()
                        return operation
                    raise ConflictError(
                        "Campaign cancel terminal outcome is not cancelled",
                        code="campaign_cancel_terminal_conflict",
                    )
                if reservation["phase"] == "signal-committed" and operation.get("state") == "running":
                    self.connection.commit()
                    return operation
                if reservation["phase"] != "requested" or operation.get("state") != "queued":
                    raise ConflictError(
                        "Campaign cancel reservation is no longer signalable",
                        code="campaign_cancel_reservation_lost",
                    )
                operation.update({"state": "running", "updatedAt": now})
                self.connection.execute(
                    "UPDATE operations SET state='running',payload=?,updated_at=? WHERE operation_id=? AND state='queued'",
                    (canonical_json(operation), now, cancel_operation_id),
                )
                cursor = self.connection.execute(
                    """UPDATE campaign_cancel_reservations
                       SET phase='signal-committed',updated_at=?
                       WHERE campaign_id=? AND cancel_operation_id=? AND controller_id=? AND phase='requested'""",
                    (now, campaign_id, cancel_operation_id, controller_id),
                )
                if cursor.rowcount != 1:
                    raise ConflictError(
                        "Campaign cancel reservation is no longer signalable",
                        code="campaign_cancel_reservation_lost",
                    )
                self.connection.commit()
                return operation
            except BaseException:
                self.connection.rollback()
                raise

    def settle_campaign_cancelled(
        self,
        campaign_id: str,
        start_operation_id: str,
        *,
        controller_id: str,
    ) -> bool:
        """Atomically publish cancellation after the owning worker has unwound."""
        now = utc_now()
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                reservation = self.connection.execute(
                    """SELECT cancel_operation_id,phase FROM campaign_cancel_reservations
                       WHERE campaign_id=? AND start_operation_id=? AND controller_id=?""",
                    (campaign_id, start_operation_id, controller_id),
                ).fetchone()
                if reservation is None or reservation["phase"] == "reconciliation-required":
                    self.connection.commit()
                    return False
                if reservation["phase"] == "terminal":
                    cancel_row = self.connection.execute(
                        "SELECT payload FROM operations WHERE operation_id=?",
                        (reservation["cancel_operation_id"],),
                    ).fetchone()
                    start_row = self.connection.execute(
                        "SELECT payload FROM operations WHERE operation_id=?",
                        (start_operation_id,),
                    ).fetchone()
                    campaign_row = self.connection.execute(
                        "SELECT payload FROM entities WHERE kind='campaign' AND entity_id=?",
                        (campaign_id,),
                    ).fetchone()
                    cancel = json.loads(cancel_row["payload"]) if cancel_row is not None else {}
                    start = json.loads(start_row["payload"]) if start_row is not None else {}
                    campaign = json.loads(campaign_row["payload"]) if campaign_row is not None else {}
                    self.connection.commit()
                    return bool(
                        cancel.get("state") == "succeeded"
                        and cancel.get("action") == "campaign.cancel"
                        and cancel.get("targetId") == campaign_id
                        and (cancel.get("result") or {}).get("startOperationId") == start_operation_id
                        and start.get("state") == "cancelled"
                        and campaign.get("state") == "cancelled"
                    )
                campaign_row = self.connection.execute(
                    "SELECT generation,payload FROM entities WHERE kind='campaign' AND entity_id=?",
                    (campaign_id,),
                ).fetchone()
                if campaign_row is None:
                    raise NotFoundError("campaign", campaign_id)
                campaign = json.loads(campaign_row["payload"])
                if campaign.get("state") != "cancelling":
                    self.connection.commit()
                    return False

                owned_job_ids = {
                    str(row["entity_id"])
                    for row in self.connection.execute(
                        "SELECT entity_id,payload FROM entities WHERE kind='job' AND archived=0"
                    ).fetchall()
                    if json.loads(row["payload"]).get("campaignId") == campaign_id
                }
                for kind in ("job", "attempt"):
                    rows = self.connection.execute(
                        "SELECT entity_id,generation,payload FROM entities WHERE kind=? AND archived=0",
                        (kind,),
                    ).fetchall()
                    for row in rows:
                        payload = json.loads(row["payload"])
                        belongs = (
                            payload.get("campaignId") == campaign_id
                            if kind == "job"
                            else payload.get("jobId") in owned_job_ids
                        )
                        if not belongs or payload.get("state") not in {"queued", "preparing", "running"}:
                            continue
                        payload.update(
                            {
                                "state": "cancelled",
                                "generation": int(row["generation"]) + 1,
                                "updatedAt": now,
                            }
                        )
                        if kind == "attempt":
                            payload["completedAt"] = now
                        self.connection.execute(
                            "UPDATE entities SET generation=?,payload=?,updated_at=? WHERE kind=? AND entity_id=?",
                            (payload["generation"], canonical_json(payload), now, kind, row["entity_id"]),
                        )

                campaign.update(
                    {
                        "state": "cancelled",
                        "generation": int(campaign_row["generation"]) + 1,
                        "updatedAt": now,
                    }
                )
                self.connection.execute(
                    "UPDATE entities SET generation=?,payload=?,updated_at=? WHERE kind='campaign' AND entity_id=?",
                    (campaign["generation"], canonical_json(campaign), now, campaign_id),
                )
                for operation_id, state, result in (
                    (start_operation_id, "cancelled", {"campaignId": campaign_id}),
                    (
                        reservation["cancel_operation_id"],
                        "succeeded",
                        {"campaignId": campaign_id, "startOperationId": start_operation_id},
                    ),
                ):
                    row = self.connection.execute(
                        "SELECT payload FROM operations WHERE operation_id=?", (operation_id,)
                    ).fetchone()
                    if row is None:
                        raise ConflictError("Cancellation operation receipt is missing", code="campaign_cancel_receipt_missing")
                    payload = json.loads(row["payload"])
                    payload.update({"state": state, "result": result, "updatedAt": now})
                    self.connection.execute(
                        "UPDATE operations SET state=?,payload=?,updated_at=? WHERE operation_id=?",
                        (state, canonical_json(payload), now, operation_id),
                    )
                self.connection.execute(
                    "UPDATE campaign_start_reservations SET phase='terminal',updated_at=? WHERE campaign_id=? AND operation_id=?",
                    (now, campaign_id, start_operation_id),
                )
                self.connection.execute(
                    "UPDATE campaign_cancel_reservations SET phase='terminal',updated_at=? WHERE campaign_id=?",
                    (now, campaign_id),
                )
                self.connection.commit()
                return True
            except BaseException:
                self.connection.rollback()
                raise

    def campaign_cancel_reservation(self, campaign_id: str) -> dict[str, str] | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM campaign_cancel_reservations WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def active_campaign_start(self, campaign_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                """SELECT r.operation_id,r.controller_id,r.phase,o.payload
                   FROM campaign_start_reservations r JOIN operations o
                   ON o.operation_id=r.operation_id WHERE r.campaign_id=?""",
                (campaign_id,),
            ).fetchone()
        if row is None or row["phase"] not in {
            "launch-committed",
            "running",
            "reconciliation-required",
        }:
            return None
        return {
            "operationId": row["operation_id"],
            "controllerId": row["controller_id"],
            "phase": row["phase"],
            "operation": json.loads(row["payload"]),
        }

    def cancel_unstarted_campaign(self, campaign_id: str) -> dict[str, Any]:
        """Preserve the legacy admin cancel for work that was never launched."""
        now = utc_now()
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                start = self.connection.execute(
                    "SELECT phase FROM campaign_start_reservations WHERE campaign_id=?",
                    (campaign_id,),
                ).fetchone()
                if start is not None and start["phase"] != "terminal":
                    raise ConflictError(
                        "Active campaign cancellation requires exact operation identifiers",
                        code="cancel_operation_ids_required",
                    )
                row = self.connection.execute(
                    "SELECT generation,payload FROM entities WHERE kind='campaign' AND entity_id=? AND archived=0",
                    (campaign_id,),
                ).fetchone()
                if row is None:
                    raise NotFoundError("campaign", campaign_id)
                campaign = json.loads(row["payload"])
                if campaign.get("state") in {"succeeded", "failed", "cancelled"}:
                    self.connection.commit()
                    return campaign
                campaign.update(
                    {
                        "state": "cancelled",
                        "generation": int(row["generation"]) + 1,
                        "updatedAt": now,
                    }
                )
                self.connection.execute(
                    "UPDATE entities SET generation=?,payload=?,updated_at=? WHERE kind='campaign' AND entity_id=?",
                    (campaign["generation"], canonical_json(campaign), now, campaign_id),
                )
                self.connection.commit()
                return campaign
            except BaseException:
                self.connection.rollback()
                raise

    def settle_campaign_succeeded(
        self,
        campaign_id: str,
        start_operation_id: str,
        result: dict[str, Any],
        *,
        controller_id: str,
    ) -> bool:
        """Commit natural success only if a cancel intent did not win first."""
        now = utc_now()
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                cancel = self.connection.execute(
                    "SELECT phase FROM campaign_cancel_reservations WHERE campaign_id=?",
                    (campaign_id,),
                ).fetchone()
                if cancel is not None and cancel["phase"] != "terminal":
                    self.connection.commit()
                    return False
                campaign_row = self.connection.execute(
                    "SELECT generation,payload FROM entities WHERE kind='campaign' AND entity_id=?",
                    (campaign_id,),
                ).fetchone()
                operation_row = self.connection.execute(
                    "SELECT payload FROM operations WHERE operation_id=?",
                    (start_operation_id,),
                ).fetchone()
                if campaign_row is None or operation_row is None:
                    raise ConflictError("Campaign execution receipt is missing", code="campaign_start_receipt_missing")
                campaign = json.loads(campaign_row["payload"])
                operation = json.loads(operation_row["payload"])
                if campaign.get("state") in {"succeeded", "failed", "cancelled"}:
                    self.connection.commit()
                    return campaign.get("state") == "succeeded"
                campaign.update({"state": "succeeded", "generation": int(campaign_row["generation"]) + 1, "updatedAt": now})
                operation.update({"state": "succeeded", "result": result, "updatedAt": now})
                self.connection.execute(
                    "UPDATE entities SET generation=?,payload=?,updated_at=? WHERE kind='campaign' AND entity_id=?",
                    (campaign["generation"], canonical_json(campaign), now, campaign_id),
                )
                self.connection.execute(
                    "UPDATE operations SET state='succeeded',payload=?,updated_at=? WHERE operation_id=?",
                    (canonical_json(operation), now, start_operation_id),
                )
                cursor = self.connection.execute(
                    "UPDATE campaign_start_reservations SET phase='terminal',updated_at=? WHERE campaign_id=? AND operation_id=? AND controller_id=? AND phase!='reconciliation-required'",
                    (now, campaign_id, start_operation_id, controller_id),
                )
                if cursor.rowcount != 1:
                    raise ConflictError("Campaign start reservation is no longer owned", code="campaign_start_reservation_lost")
                self.connection.commit()
                return True
            except BaseException:
                self.connection.rollback()
                raise

    def settle_campaign_failed(
        self,
        campaign_id: str,
        start_operation_id: str,
        error: dict[str, Any],
        *,
        controller_id: str,
    ) -> bool:
        """Publish the real execution failure and close any losing cancel intent."""
        now = utc_now()
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                start_reservation = self.connection.execute(
                    "SELECT phase FROM campaign_start_reservations WHERE campaign_id=? AND operation_id=? AND controller_id=?",
                    (campaign_id, start_operation_id, controller_id),
                ).fetchone()
                cancel_fence = self.connection.execute(
                    "SELECT phase FROM campaign_cancel_reservations WHERE campaign_id=?",
                    (campaign_id,),
                ).fetchone()
                if start_reservation is not None and start_reservation["phase"] == "terminal":
                    self.connection.commit()
                    return False
                if (
                    start_reservation is None
                    or start_reservation["phase"] == "reconciliation-required"
                    or (cancel_fence is not None and cancel_fence["phase"] == "reconciliation-required")
                ):
                    self.connection.commit()
                    return False
                campaign_row = self.connection.execute(
                    "SELECT generation,payload FROM entities WHERE kind='campaign' AND entity_id=?",
                    (campaign_id,),
                ).fetchone()
                start_row = self.connection.execute(
                    "SELECT payload FROM operations WHERE operation_id=?",
                    (start_operation_id,),
                ).fetchone()
                if campaign_row is None or start_row is None:
                    raise ConflictError("Campaign execution receipt is missing", code="campaign_start_receipt_missing")
                campaign = json.loads(campaign_row["payload"])
                start = json.loads(start_row["payload"])
                if campaign.get("state") in {"succeeded", "failed", "cancelled"} or start.get("state") in {
                    "succeeded", "failed", "cancelled"
                }:
                    self.connection.commit()
                    return False
                campaign.update({"state": "failed", "generation": int(campaign_row["generation"]) + 1, "updatedAt": now})
                start.update({"state": "failed", "error": error, "updatedAt": now})
                self.connection.execute(
                    "UPDATE entities SET generation=?,payload=?,updated_at=? WHERE kind='campaign' AND entity_id=?",
                    (campaign["generation"], canonical_json(campaign), now, campaign_id),
                )
                self.connection.execute(
                    "UPDATE operations SET state='failed',payload=?,updated_at=? WHERE operation_id=?",
                    (canonical_json(start), now, start_operation_id),
                )
                cancel = self.connection.execute(
                    "SELECT cancel_operation_id,phase FROM campaign_cancel_reservations WHERE campaign_id=?",
                    (campaign_id,),
                ).fetchone()
                if cancel is not None and cancel["phase"] != "terminal":
                    row = self.connection.execute(
                        "SELECT payload FROM operations WHERE operation_id=?",
                        (cancel["cancel_operation_id"],),
                    ).fetchone()
                    if row is not None:
                        operation = json.loads(row["payload"])
                        operation.update(
                            {
                                "state": "failed",
                                "updatedAt": now,
                                "error": {
                                    "code": "campaign_failed_before_cancellation",
                                    "detail": error.get("detail", "Campaign failed"),
                                },
                            }
                        )
                        self.connection.execute(
                            "UPDATE operations SET state='failed',payload=?,updated_at=? WHERE operation_id=?",
                            (canonical_json(operation), now, cancel["cancel_operation_id"]),
                        )
                    self.connection.execute(
                        "UPDATE campaign_cancel_reservations SET phase='terminal',updated_at=? WHERE campaign_id=?",
                        (now, campaign_id),
                    )
                self.connection.execute(
                    "UPDATE campaign_start_reservations SET phase='terminal',updated_at=? WHERE campaign_id=? AND operation_id=? AND controller_id=?",
                    (now, campaign_id, start_operation_id, controller_id),
                )
                self.connection.commit()
                return True
            except BaseException:
                self.connection.rollback()
                raise

    def incomplete_campaign_cancels(self) -> list[dict[str, str]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM campaign_cancel_reservations WHERE phase IN ('requested','signal-committed') ORDER BY created_at,campaign_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_campaign_cancel_reconciliation_required(
        self,
        campaign_id: str,
        cancel_operation_id: str,
        *,
        controller_id: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> bool:
        now = utc_now()
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                reservation = self.connection.execute(
                    "SELECT phase FROM campaign_cancel_reservations WHERE campaign_id=? AND cancel_operation_id=? AND controller_id=?",
                    (campaign_id, cancel_operation_id, controller_id),
                ).fetchone()
                if reservation is None or reservation["phase"] not in {"requested", "signal-committed"}:
                    self.connection.commit()
                    return False
                row = self.connection.execute(
                    "SELECT payload FROM operations WHERE operation_id=?", (cancel_operation_id,)
                ).fetchone()
                if row is None:
                    raise ConflictError("Cancellation operation receipt is missing", code="campaign_cancel_receipt_missing")
                operation = json.loads(row["payload"])
                operation.update(
                    {
                        "state": "reconciliation-required",
                        "updatedAt": now,
                        "error": {
                            "code": "campaign_cancel_outcome_unknown",
                            "detail": reason,
                            **(details or {}),
                        },
                    }
                )
                self.connection.execute(
                    "UPDATE operations SET state=?,payload=?,updated_at=? WHERE operation_id=?",
                    (operation["state"], canonical_json(operation), now, cancel_operation_id),
                )
                self.connection.execute(
                    "UPDATE campaign_cancel_reservations SET phase='reconciliation-required',updated_at=? WHERE campaign_id=?",
                    (now, campaign_id),
                )
                self.connection.commit()
                return True
            except BaseException:
                self.connection.rollback()
                raise

    def require_campaign_plan_access(
        self,
        campaign_id: str,
        *,
        operation_id: str | None = None,
        controller_id: str | None = None,
    ) -> None:
        """Fence public planning while a durable start is in progress."""
        with self._lock:
            row = self.connection.execute(
                """
                SELECT operation_id, controller_id, phase
                FROM campaign_start_reservations WHERE campaign_id=?
                """,
                (campaign_id,),
            ).fetchone()
        if row is None or row["phase"] == "terminal":
            return
        if (
            operation_id is not None
            and controller_id is not None
            and (row["operation_id"], row["controller_id"], row["phase"])
            == (operation_id, controller_id, "reserved")
        ):
            return
        raise ConflictError(
            "Campaign planning is fenced by its reserved start operation",
            code="campaign_start_reserved",
        )

    def mark_campaign_start_launch_committed(
        self, campaign_id: str, operation_id: str, *, controller_id: str
    ) -> None:
        with self._lock, self.connection:
            cursor = self.connection.execute(
                """
                UPDATE campaign_start_reservations
                SET phase='launch-committed', updated_at=?
                WHERE campaign_id=? AND operation_id=? AND controller_id=? AND phase='reserved'
                """,
                (utc_now(), campaign_id, operation_id, controller_id),
            )
            if cursor.rowcount != 1:
                raise ConflictError(
                    "Campaign start reservation is no longer launchable",
                    code="campaign_start_reservation_lost",
                )

    def mark_campaign_start_running(
        self, campaign_id: str, operation_id: str, *, controller_id: str
    ) -> bool:
        """Publish running only when no atomic cancel reservation won first."""
        now = utc_now()
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                reservation = self.connection.execute(
                    """SELECT phase FROM campaign_start_reservations
                       WHERE campaign_id=? AND operation_id=? AND controller_id=?""",
                    (campaign_id, operation_id, controller_id),
                ).fetchone()
                campaign_row = self.connection.execute(
                    "SELECT generation,payload FROM entities WHERE kind='campaign' AND entity_id=?",
                    (campaign_id,),
                ).fetchone()
                operation_row = self.connection.execute(
                    "SELECT payload FROM operations WHERE operation_id=?",
                    (operation_id,),
                ).fetchone()
                if reservation is None or campaign_row is None or operation_row is None:
                    raise ConflictError(
                        "Campaign start reservation is no longer owned",
                        code="campaign_start_reservation_lost",
                    )
                if reservation["phase"] == "running":
                    self.connection.commit()
                    return True
                if reservation["phase"] != "launch-committed":
                    self.connection.commit()
                    return False
                campaign = json.loads(campaign_row["payload"])
                operation = json.loads(operation_row["payload"])
                if campaign.get("state") == "cancelling":
                    self.connection.commit()
                    return False
                if campaign.get("state") in {"succeeded", "failed", "cancelled"}:
                    self.connection.commit()
                    return False
                campaign.update(
                    {
                        "state": "running",
                        "generation": int(campaign_row["generation"]) + 1,
                        "updatedAt": now,
                    }
                )
                operation.update({"state": "running", "updatedAt": now})
                self.connection.execute(
                    "UPDATE entities SET generation=?,payload=?,updated_at=? WHERE kind='campaign' AND entity_id=?",
                    (campaign["generation"], canonical_json(campaign), now, campaign_id),
                )
                self.connection.execute(
                    "UPDATE operations SET state='running',payload=?,updated_at=? WHERE operation_id=?",
                    (canonical_json(operation), now, operation_id),
                )
                cursor = self.connection.execute(
                    """UPDATE campaign_start_reservations SET phase='running',updated_at=?
                       WHERE campaign_id=? AND operation_id=? AND controller_id=? AND phase='launch-committed'""",
                    (now, campaign_id, operation_id, controller_id),
                )
                if cursor.rowcount != 1:
                    raise ConflictError(
                        "Campaign start reservation changed concurrently",
                        code="campaign_start_reservation_lost",
                    )
                self.connection.commit()
                return True
            except BaseException:
                self.connection.rollback()
                raise

    def commit_campaign_plan(
        self,
        campaign_id: str,
        campaign: dict[str, Any],
        jobs: list[dict[str, Any]],
        *,
        expected_generation: int,
        operation_id: str | None = None,
        controller_id: str | None = None,
    ) -> dict[str, Any]:
        """Commit plan entities only if no competing start acquired the fence.

        Dataset and worker validation happen before this short local SQLite
        transaction. This final guard closes the check/validation/commit race
        without holding a database transaction over filesystem or network I/O.
        """
        now = utc_now()
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                reservation = self.connection.execute(
                    """
                    SELECT operation_id, controller_id, phase
                    FROM campaign_start_reservations WHERE campaign_id=?
                    """,
                    (campaign_id,),
                ).fetchone()
                allowed = reservation is None or reservation["phase"] == "terminal"
                if reservation is not None and (
                    operation_id is not None
                    and controller_id is not None
                    and (
                        reservation["operation_id"],
                        reservation["controller_id"],
                        reservation["phase"],
                    )
                    == (operation_id, controller_id, "reserved")
                ):
                    allowed = True
                if not allowed:
                    raise ConflictError(
                        "Campaign planning is fenced by its reserved start operation",
                        code="campaign_start_reserved",
                    )

                campaign_row = self.connection.execute(
                    """
                    SELECT generation, created_at FROM entities
                    WHERE kind='campaign' AND entity_id=? AND archived=0
                    """,
                    (campaign_id,),
                ).fetchone()
                if campaign_row is None:
                    raise NotFoundError("campaign", campaign_id)
                current_generation = int(campaign_row["generation"])
                if current_generation != expected_generation:
                    raise ConflictError(
                        f"campaign {campaign_id} generation is {current_generation}, not {expected_generation}",
                        code="generation_conflict",
                    )

                for payload in jobs:
                    job_id = str(payload["jobId"])
                    row = self.connection.execute(
                        """
                        SELECT generation, created_at FROM entities
                        WHERE kind='job' AND entity_id=?
                        """,
                        (job_id,),
                    ).fetchone()
                    generation = 1 if row is None else int(row["generation"]) + 1
                    created = now if row is None else str(row["created_at"])
                    stored_job = dict(payload)
                    stored_job.update(
                        {
                            "generation": generation,
                            "createdAt": created,
                            "updatedAt": now,
                        }
                    )
                    self.connection.execute(
                        """
                        INSERT INTO entities(
                            kind, entity_id, generation, payload, archived, created_at, updated_at
                        ) VALUES('job', ?, ?, ?, 0, ?, ?)
                        ON CONFLICT(kind, entity_id) DO UPDATE SET
                            generation=excluded.generation,
                            payload=excluded.payload,
                            archived=0,
                            updated_at=excluded.updated_at
                        """,
                        (
                            job_id,
                            generation,
                            canonical_json(stored_job),
                            created,
                            now,
                        ),
                    )

                stored_campaign = dict(campaign)
                stored_campaign.update(
                    {
                        "generation": current_generation + 1,
                        "createdAt": str(campaign_row["created_at"]),
                        "updatedAt": now,
                    }
                )
                self.connection.execute(
                    """
                    UPDATE entities SET generation=?, payload=?, updated_at=?
                    WHERE kind='campaign' AND entity_id=?
                    """,
                    (
                        stored_campaign["generation"],
                        canonical_json(stored_campaign),
                        now,
                        campaign_id,
                    ),
                )
                self.connection.commit()
                return stored_campaign
            except BaseException:
                self.connection.rollback()
                raise

    def incomplete_campaign_starts(self) -> list[dict[str, str]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT campaign_id, operation_id, controller_id, phase
                FROM campaign_start_reservations
                WHERE phase IN ('reserved', 'launch-committed', 'running')
                ORDER BY created_at, campaign_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_campaign_start_reconciliation_required(
        self,
        campaign_id: str,
        operation_id: str,
        *,
        controller_id: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> bool:
        now = utc_now()
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                reservation = self.connection.execute(
                    """
                    SELECT phase FROM campaign_start_reservations
                    WHERE campaign_id=? AND operation_id=? AND controller_id=?
                    """,
                    (campaign_id, operation_id, controller_id),
                ).fetchone()
                cancel_reservation = self.connection.execute(
                    """
                    SELECT cancel_operation_id, start_operation_id, controller_id, phase
                    FROM campaign_cancel_reservations WHERE campaign_id=?
                    """,
                    (campaign_id,),
                ).fetchone()
                if reservation is None or reservation["phase"] not in {
                    "reserved",
                    "launch-committed",
                    "running",
                }:
                    self.connection.commit()
                    return False
                if cancel_reservation is not None and (
                    cancel_reservation["start_operation_id"] != operation_id
                    or cancel_reservation["controller_id"] != controller_id
                ):
                    raise ConflictError(
                        "Campaign cancellation identity differs from its active start",
                        code="campaign_cancel_identity_mismatch",
                    )
                row = self.connection.execute(
                    "SELECT payload FROM operations WHERE operation_id=?",
                    (operation_id,),
                ).fetchone()
                if row is None:
                    raise ConflictError(
                        "Campaign start reservation has no operation receipt",
                        code="campaign_start_receipt_missing",
                    )
                operation = json.loads(row["payload"])
                operation.update(
                    {
                        "state": "reconciliation-required",
                        "updatedAt": now,
                        "error": {
                            "code": "campaign_start_outcome_unknown",
                            "detail": reason,
                            **(details or {}),
                        },
                    }
                )
                self.connection.execute(
                    "UPDATE operations SET state=?, payload=?, updated_at=? WHERE operation_id=?",
                    (
                        operation["state"],
                        canonical_json(operation),
                        now,
                        operation_id,
                    ),
                )
                self.connection.execute(
                    """
                    UPDATE campaign_start_reservations
                    SET phase='reconciliation-required', updated_at=?
                    WHERE campaign_id=?
                    """,
                    (now, campaign_id),
                )
                if (
                    cancel_reservation is not None
                    and cancel_reservation["phase"] in {"requested", "signal-committed"}
                ):
                    cancel_operation_id = str(
                        cancel_reservation["cancel_operation_id"]
                    )
                    cancel_row = self.connection.execute(
                        "SELECT payload FROM operations WHERE operation_id=?",
                        (cancel_operation_id,),
                    ).fetchone()
                    if cancel_row is None:
                        raise ConflictError(
                            "Cancellation operation receipt is missing",
                            code="campaign_cancel_receipt_missing",
                        )
                    cancel_operation = json.loads(cancel_row["payload"])
                    cancel_operation.update(
                        {
                            "state": "reconciliation-required",
                            "updatedAt": now,
                            "error": {
                                "code": "campaign_cancel_outcome_unknown",
                                "detail": reason,
                                **(details or {}),
                            },
                        }
                    )
                    self.connection.execute(
                        "UPDATE operations SET state=?,payload=?,updated_at=? WHERE operation_id=?",
                        (
                            cancel_operation["state"],
                            canonical_json(cancel_operation),
                            now,
                            cancel_operation_id,
                        ),
                    )
                    self.connection.execute(
                        """
                        UPDATE campaign_cancel_reservations
                        SET phase='reconciliation-required',updated_at=?
                        WHERE campaign_id=? AND cancel_operation_id=?
                        """,
                        (now, campaign_id, cancel_operation_id),
                    )
                campaign_row = self.connection.execute(
                    "SELECT generation, payload FROM entities WHERE kind='campaign' AND entity_id=?",
                    (campaign_id,),
                ).fetchone()
                if campaign_row is not None:
                    campaign = json.loads(campaign_row["payload"])
                    if campaign.get("state") in {"running", "recovering"}:
                        campaign["state"] = "recovering"
                        campaign["generation"] = int(campaign_row["generation"]) + 1
                        campaign["updatedAt"] = now
                        self.connection.execute(
                            """
                            UPDATE entities SET generation=?, payload=?, updated_at=?
                            WHERE kind='campaign' AND entity_id=?
                            """,
                            (
                                campaign["generation"],
                                canonical_json(campaign),
                                now,
                                campaign_id,
                            ),
                        )
                self.connection.commit()
                return True
            except BaseException:
                self.connection.rollback()
                raise

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
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                if kind == "campaign":
                    reservation = self.connection.execute(
                        "SELECT phase FROM campaign_start_reservations WHERE campaign_id=?",
                        (entity_id,),
                    ).fetchone()
                    if (
                        reservation is not None
                        and reservation["phase"] == "reconciliation-required"
                        and payload.get("state") != "recovering"
                    ):
                        raise ConflictError(
                            "Campaign requires explicit start reconciliation",
                            code="campaign_start_reconciliation_required",
                        )
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
                self.connection.commit()
                return stored
            except BaseException:
                self.connection.rollback()
                raise

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

    def put_operation(
        self, payload: dict[str, Any], *, controller_id: str | None = None
    ) -> dict[str, Any]:
        now = utc_now()
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                previous = self.connection.execute(
                    "SELECT state FROM operations WHERE operation_id=?",
                    (payload["operationId"],),
                ).fetchone()
                if (
                    previous is not None
                    and previous["state"] == "reconciliation-required"
                    and payload.get("state") != "reconciliation-required"
                ):
                    raise ConflictError(
                        "Campaign start operation requires explicit reconciliation",
                        code="campaign_start_reconciliation_required",
                    )
                if payload.get("action") == "campaign.start" and controller_id is not None:
                    reservation = self.connection.execute(
                        """
                        SELECT controller_id, phase FROM campaign_start_reservations
                        WHERE operation_id=?
                        """,
                        (payload["operationId"],),
                    ).fetchone()
                    if reservation is None or reservation["controller_id"] != controller_id:
                        raise ConflictError(
                            "Campaign start operation is no longer owned by this controller",
                            code="campaign_start_reservation_lost",
                        )
                    if reservation["phase"] == "reconciliation-required":
                        raise ConflictError(
                            "Campaign start operation requires explicit reconciliation",
                            code="campaign_start_reconciliation_required",
                        )
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
                if payload.get("action") == "campaign.start":
                    phase = {
                        "running": "running",
                        "succeeded": "terminal",
                        "failed": "terminal",
                        "cancelled": "terminal",
                        "reconciliation-required": "reconciliation-required",
                    }.get(str(payload.get("state")))
                    if phase is not None:
                        self.connection.execute(
                            """
                            UPDATE campaign_start_reservations
                            SET phase=?, updated_at=? WHERE operation_id=?
                            """,
                            (phase, now, payload["operationId"]),
                        )
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise
        return payload

    def get_operation(self, operation_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.connection.execute(
                "SELECT payload FROM operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError("operation", operation_id)
        return json.loads(row["payload"])

    def list_operations(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT payload FROM operations ORDER BY created_at,operation_id"
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

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
    ) -> dict[str, Any]:
        with self._lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO idempotency(scope, idem_key, request_hash, response, created_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(scope, idem_key) DO NOTHING
                """,
                (scope, key, request_hash, canonical_json(response), utc_now()),
            )
        stored = self.get_idempotent(scope, key, request_hash)
        if stored is None:
            raise ConflictError(
                "Idempotency receipt could not be persisted",
                code="idempotency_receipt_missing",
            )
        return stored
