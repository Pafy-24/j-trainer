from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import BinaryIO


class ControllerLease:
    """Process-held marker used to distinguish a live peer from a crashed owner."""

    def __init__(self, root: Path):
        self.instance_id = uuid.uuid4().hex
        self._root = root / "controller-leases"
        self._root.mkdir(parents=True, exist_ok=True)
        self._handle = self._open(self.instance_id)
        self._lock(self._handle)

    def close(self) -> None:
        if self._handle.closed:
            return
        self._unlock(self._handle)
        self._handle.close()

    def owner_is_active(self, instance_id: str) -> bool:
        if instance_id == self.instance_id:
            return True
        if re.fullmatch(r"[0-9a-f]{32}", instance_id) is None:
            return False
        handle = self._open(instance_id)
        try:
            try:
                self._lock(handle)
            except OSError:
                return True
            self._unlock(handle)
            return False
        finally:
            handle.close()

    def _open(self, instance_id: str) -> BinaryIO:
        path = self._root / f"{instance_id}.lock"
        handle = path.open("a+b")
        if path.stat().st_size == 0:
            handle.write(b"1")
            handle.flush()
        return handle

    @staticmethod
    def _lock(handle: BinaryIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(handle: BinaryIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
