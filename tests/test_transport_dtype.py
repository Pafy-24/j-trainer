from __future__ import annotations

from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from jester_trainer.parameter_server import ParameterServer


def test_weight_transport_dtype_does_not_downcast_coordinator_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "round.safetensors"
    original = torch.tensor([1.125, -2.25], dtype=torch.float32)
    save_file({"weight": original}, str(checkpoint))

    server = object.__new__(ParameterServer)
    server.campaign = SimpleNamespace(
        spec=SimpleNamespace(
            training=SimpleNamespace(transport_dtype="bfloat16")
        )
    )
    transport = server._transport_weight_path(checkpoint)

    assert transport != checkpoint
    assert load_file(str(transport))["weight"].dtype == torch.bfloat16
    assert load_file(str(checkpoint))["weight"].dtype == torch.float32


def test_committed_gradient_downloads_are_removed(tmp_path) -> None:
    first = tmp_path / "gradients" / "round-1" / "a.safetensors"
    second = tmp_path / "gradients" / "round-1" / "b.safetensors"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    first.with_suffix(first.suffix + ".zst").write_bytes(b"compressed")

    ParameterServer._cleanup_downloaded_gradients({"a": first, "b": second})

    assert not first.exists()
    assert not second.exists()
    assert not first.parent.exists()
