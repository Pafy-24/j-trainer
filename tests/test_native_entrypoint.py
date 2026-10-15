import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


ENTRYPOINT = Path(__file__).parents[1] / "src" / "trainer.py"


def test_native_entrypoint_delegates_once_and_preserves_exit_code(monkeypatch) -> None:
    calls = []
    package = ModuleType("jester_trainer")
    package.__path__ = []
    cli = ModuleType("jester_trainer.cli")

    def main() -> int:
        calls.append("main")
        return 17

    cli.main = main
    monkeypatch.setitem(sys.modules, "jester_trainer", package)
    monkeypatch.setitem(sys.modules, "jester_trainer.cli", cli)

    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(ENTRYPOINT), run_name="__main__")

    assert raised.value.code == 17
    assert calls == ["main"]
