from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from jester_trainer.config import Settings
from jester_trainer.parameter_server import ParameterServer
from jester_trainer.service import TrainerService

from .helpers import (
    artifact_weights,
    campaign,
    dataset_manifest,
    run_campaign,
    tiny_model,
    training_dataset,
    worker_topology,
)


def test_two_worker_update_matches_combined_single_worker_batch() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model = root / "base"
        data = root / "train.jsonl"
        tiny_model(model)
        training_dataset(data)
        with worker_topology(root / "workers", count=2) as topology:
            service = TrainerService(Settings.for_test(root / "trainer", topology))
            try:
                service.put_dataset(dataset_manifest(data))
                distributed = run_campaign(
                    service,
                    campaign(
                        root,
                        model,
                        campaign_id="distributed",
                        version="distributed",
                        workers=["worker-1", "worker-2"],
                    ),
                )
                combined = run_campaign(
                    service,
                    campaign(
                        root,
                        model,
                        campaign_id="combined",
                        version="combined",
                        workers=["worker-1"],
                    ),
                )
                distributed_state = artifact_weights(distributed)
                combined_state = artifact_weights(combined)
                assert set(distributed_state) == set(combined_state)
                for name in distributed_state:
                    assert torch.allclose(
                        distributed_state[name],
                        combined_state[name],
                        rtol=2e-5,
                        atol=2e-6,
                    ), name
                rounds = service.store.list_entities("round")
                assert len(rounds) == 2
                assert all(item["state"] == "committed" for item in rounds)
                assert all(len(item["contributions"]) >= 1 for item in rounds)
                assert service.get_campaign("distributed").state == "succeeded"
            finally:
                service.close()


def test_remote_cas_binding_uses_digest_as_portable_base_identity() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        campaign_model = campaign(
            root,
            root / "runtime-model-path",
            campaign_id="portable-identity",
            version="portable",
            workers=["worker-1"],
        )
        digest = "sha256:" + "c" * 64
        server = object.__new__(ParameterServer)
        server.campaign = campaign_model
        server.mi_config = campaign_model.spec.facets[0].mi_config
        server.mi_config["baseModel"]["artifactDigest"] = digest
        server.shard_plan = {"sources": []}
        server.output = root / "output"
        server.artifact_bindings = []

        config = server._worker_training_config()

        assert config["baseModel"]["id"] == "rm-artifact-pending"
        assert config["baseModel"]["canonicalId"] == digest
        assert server.artifact_bindings[0] == {
            "target": "/baseModel/id",
            "digest": digest,
            "logicalPath": None,
        }


def test_checkpoint_metrics_survive_a_resumed_completed_attempt() -> None:
    source = object.__new__(ParameterServer)
    source.losses = [4.5, 4.0]
    source.round_durations = [2.0, 3.0]
    source.worker_wait_seconds = {"worker-1": 0.25}
    source.total_supervised_tokens = 500

    resumed = object.__new__(ParameterServer)
    resumed.start_round = 2
    resumed.losses = []
    resumed.round_durations = []
    resumed.worker_wait_seconds = {"worker-1": 0.0}
    resumed.total_supervised_tokens = 0
    resumed._restore_checkpoint_metrics({"metrics": source._checkpoint_metrics()})

    assert resumed.losses == [4.5, 4.0]
    assert resumed.round_durations == [2.0, 3.0]
    assert resumed.worker_wait_seconds == {"worker-1": 0.25}
    assert resumed.total_supervised_tokens == 500
