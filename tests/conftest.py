from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for component in (PROJECT_ROOT / "Trainer", PROJECT_ROOT / "MI", PROJECT_ROOT / "RM"):
    value = str(component)
    if value not in sys.path:
        sys.path.insert(0, value)
