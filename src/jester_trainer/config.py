from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import TrainerError
from .util import atomic_write_json


@dataclass(frozen=True)
class Settings:
    home: Path
    database: Path
    token_path: Path
    topology: dict[str, Any]
    lease: dict[str, Any]
    api_host: str = "127.0.0.1"
    api_port: int = 8140

    @classmethod
    def from_rm(cls) -> "Settings":
        try:
            from jester_rm_client import require_lease

            effective = require_lease()
        except ImportError as exc:
            raise TrainerError(
                "RM_LEASE_REQUIRED: jester-rm-client is not installed",
                code="RM_LEASE_REQUIRED",
                status=409,
            ) from exc
        except Exception as exc:
            raise TrainerError(
                f"RM_LEASE_REQUIRED: {exc}",
                code=getattr(exc, "code", "RM_LEASE_REQUIRED"),
                status=409,
            ) from exc
        topology = effective.lease.get("context")
        if not isinstance(topology, dict) or topology.get("topologyType") != "training":
            raise TrainerError(
                "Trainer requires an active RM training topology context",
                code="rm_training_topology_required",
                status=409,
            )
        if topology.get("role") != "trainer":
            raise TrainerError(
                "RM lease role must be trainer",
                code="rm_role_mismatch",
                status=409,
            )
        home = (effective.run_directory / "trainer").resolve()
        return cls(
            home=home,
            database=home / "trainer.sqlite3",
            token_path=home / "trainer.token",
            topology=topology,
            lease=effective.lease,
            api_port=int(effective.lease.get("ports", {}).get("http", 8140)),
        )

    @classmethod
    def for_test(
        cls, root: str | Path, topology: dict[str, Any] | None = None
    ) -> "Settings":
        home = Path(root).resolve()
        topology = topology or {
            "topologyId": "test-topology",
            "topologyType": "training",
            "role": "trainer",
            "nodeId": "trainer",
            "channels": [],
            "nodes": [],
        }
        return cls(
            home=home,
            database=home / "trainer.sqlite3",
            token_path=home / "trainer.token",
            topology=topology,
            lease={
                "leaseId": "test-lease",
                "instanceId": "test-trainer",
                "hostId": "test-host",
                "requestedBackend": "cpu",
                "effectiveBackend": "cpu",
                "paths": {"run": str(home), "artifacts": str(home / "artifacts")},
            },
        )

    def initialize(self) -> str:
        self.home.mkdir(parents=True, exist_ok=True)
        if not self.token_path.exists():
            self.token_path.write_text(secrets.token_urlsafe(48), encoding="utf-8")
            try:
                os.chmod(self.token_path, 0o600)
            except OSError:
                pass
        atomic_write_json(
            self.home / "effective-topology.json",
            {
                "schemaVersion": "jester-trainer-topology-context-v1",
                "topology": self.topology,
                "lease": self.lease,
            },
        )
        return self.token_path.read_text(encoding="utf-8").strip()
