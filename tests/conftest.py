from __future__ import annotations

import faulthandler
import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for component in (PROJECT_ROOT / "Trainer", PROJECT_ROOT / "MI", PROJECT_ROOT / "RM"):
    value = str(component)
    if value not in sys.path:
        sys.path.insert(0, value)


def _preload_optional_pyarrow() -> None:
    """Avoid PyArrow's recoverable Windows SEH probe racing worker threads."""
    enabled = faulthandler.is_enabled()
    if enabled:
        faulthandler.disable()
    try:
        try:
            importlib.import_module("pyarrow")
        except ImportError:
            pass
    finally:
        if enabled:
            faulthandler.enable()


_preload_optional_pyarrow()
