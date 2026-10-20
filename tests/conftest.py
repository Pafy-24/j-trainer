from __future__ import annotations

import faulthandler
import importlib
import sys
from pathlib import Path


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = COMPONENT_ROOT.parents[2]
SOURCE_ROOTS = (
    COMPONENT_ROOT / "src",
    COMPONENT_ROOT.parent / "j-mi/src",
    ENGINE_ROOT / "jester-core/packages/python/core/src",
    ENGINE_ROOT / "jester-core/packages/python/rm-client/src",
    ENGINE_ROOT / "jester-rm/src",
    ENGINE_ROOT / "jester-rma/src",
)
for source_root in SOURCE_ROOTS:
    value = str(source_root)
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
