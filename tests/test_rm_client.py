from __future__ import annotations

from unittest.mock import MagicMock, patch

import json

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


def test_trainer_publication_registration_uses_exact_route_key_and_body() -> None:
    client = RMScopedClient(
        {
            "topologyId": "training-one",
            "rmController": {"url": "http://127.0.0.1:8790"},
            "serviceGrant": {"credential": "opaque-grant"},
        }
    )
    response = MagicMock()
    response.read.return_value = b'{"schemaVersion":"jester-rm-trainer-publication-authority-v1"}'
    response.__enter__.return_value = response
    payload = {"publicationId": "publication-one", "resultId": "result-one"}

    with patch("urllib.request.urlopen", return_value=response) as open_request:
        client.register_trainer_publication(
            payload, idempotency_key="a" * 32
        )

    request = open_request.call_args.args[0]
    assert request.full_url.endswith("/v1/topologies/training-one/trainer-publications")
    assert request.method == "POST"
    assert request.headers["Idempotency-key"] == "a" * 32
    assert json.loads(request.data) == payload
    assert open_request.call_args.kwargs["timeout"] == 3600

    with patch("urllib.request.urlopen", return_value=response) as recovery_request:
        client.register_trainer_publication(
            payload, idempotency_key="a" * 32, timeout_seconds=5
        )
    assert recovery_request.call_args.kwargs["timeout"] == 5
