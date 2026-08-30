from __future__ import annotations

from unittest.mock import MagicMock, patch

from jester_trainer.rm_client import RMScopedClient


def test_artifact_publish_allows_large_remote_cas_sync() -> None:
    client = RMScopedClient(
        {
            "topologyId": "training-one",
            "rmController": {"url": "http://127.0.0.1:8790"},
            "serviceGrant": {"credential": "opaque-grant"},
        }
    )
    response = MagicMock()
    response.read.return_value = b'{"artifact":{"digest":"sha256:test"}}'
    response.__enter__.return_value = response

    with patch("urllib.request.urlopen", return_value=response) as open_request:
        result = client.publish_artifact("trainer-results/facet")

    assert result["artifact"]["digest"] == "sha256:test"
    assert open_request.call_args.kwargs["timeout"] == 3600


def test_regular_rm_reads_keep_short_control_plane_timeout() -> None:
    client = RMScopedClient(
        {
            "topologyId": "training-one",
            "rmController": {"url": "http://127.0.0.1:8790"},
            "serviceGrant": {"credential": "opaque-grant"},
        }
    )
    response = MagicMock()
    response.read.return_value = b'{"items":[]}'
    response.__enter__.return_value = response

    with patch("urllib.request.urlopen", return_value=response) as open_request:
        client.metrics()

    assert open_request.call_args.kwargs["timeout"] == 30
