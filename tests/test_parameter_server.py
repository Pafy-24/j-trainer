from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from jester_trainer.config import Settings
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
