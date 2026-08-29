from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from jester_trainer.datasets import DatasetManager
from jester_trainer.errors import TrainerError
from jester_trainer.models import Campaign, DatasetManifest, DatasetUse
from jester_trainer.service import _facet_dataset_uses
from jester_trainer.util import sha256_file


def _manifest(path: Path, *, split: str = "train") -> DatasetManifest:
    return DatasetManifest.model_validate(
        {
            "metadata": {"id": "dataset"},
            "files": [
                {
                    "split": split,
                    "path": str(path),
                    "digest": f"sha256:{sha256_file(path)}",
                }
            ],
        }
    )


def test_semantic_variants_never_cross_deterministic_shards() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "train.jsonl"
        records = [
            {"id": "a-ro", "semanticCaseId": "a", "text": "unu"},
            {"id": "a-en", "semanticCaseId": "a", "text": "one"},
            {"id": "b-ro", "semanticCaseId": "b", "text": "doi"},
            {"id": "c-ro", "semanticCaseId": "c", "text": "trei"},
        ]
        path.write_text(
            "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
        )
        manager = DatasetManager()
        manifest = _manifest(path)
        first = manager.shard_plan(
            [(manifest, DatasetUse(datasetId="dataset", split="train"))],
            ["w1", "w2"],
            seed=17,
        )
        second = manager.shard_plan(
            [(manifest, DatasetUse(datasetId="dataset", split="train"))],
            ["w1", "w2"],
            seed=17,
        )
        assert first == second
        owners = {
            index: worker
            for worker, indices in first["workers"].items()
            for index in indices
        }
        assert owners[0] == owners[1]


def test_test_jsonl_is_never_accepted_as_gradient_source() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "test.jsonl"
        path.write_text('{"text":"held out"}\n', encoding="utf-8")
        manager = DatasetManager()
        manifest = _manifest(path, split="train")
        with pytest.raises(TrainerError) as raised:
            manager.shard_plan(
                [(manifest, DatasetUse(datasetId="dataset", split="train"))],
                ["w1"],
                seed=1,
            )
        assert raised.value.code == "test_split_gradient_forbidden"


def test_dataset_weight_is_rejected_instead_of_being_silently_ignored() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "train.jsonl"
        path.write_text('{"semanticCaseId":"case-1","text":"sample"}\n', encoding="utf-8")
        manager = DatasetManager()
        manifest = _manifest(path)

        with pytest.raises(TrainerError) as raised:
            manager.shard_plan(
                [
                    (
                        manifest,
                        DatasetUse(datasetId="dataset", split="train", weight=2.0),
                    )
                ],
                ["w1"],
                seed=1,
            )

        assert raised.value.code == "dataset_weight_unsupported"
        assert raised.value.status == 422
        assert raised.value.fields["datasetUses"][0]["weight"] == 2.0


def test_campaign_rejects_cycles_and_test_split_before_runtime() -> None:
    base = {
        "metadata": {"id": "invalid"},
        "spec": {
            "topologyId": "topology",
            "datasets": [{"datasetId": "data", "split": "test"}],
            "facets": [
                {
                    "id": "a",
                    "dependsOn": ["b"],
                    "miConfig": {"facet": {"id": "a", "version": "1"}},
                },
                {
                    "id": "b",
                    "dependsOn": ["a"],
                    "miConfig": {"facet": {"id": "b", "version": "1"}},
                },
            ],
            "training": {"maxSteps": 1},
            "checkpoint": {"outputDirectory": "out"},
        },
    }
    with pytest.raises(ValidationError):
        Campaign.model_validate(base)


def test_facet_dataset_selection_prevents_parent_child_corpus_leakage() -> None:
    campaign = Campaign.model_validate(
        {
            "metadata": {"id": "isolated-datasets"},
            "spec": {
                "topologyId": "topology",
                "datasets": [],
                "facets": [
                    {
                        "id": "coder",
                        "datasets": [
                            {"datasetId": "coder-data", "split": "train"}
                        ],
                        "miConfig": {"facet": {"id": "coder", "version": "1"}},
                    },
                    {
                        "id": "coder.python",
                        "dependsOn": ["coder"],
                        "datasets": [
                            {
                                "datasetId": "coder-python-data",
                                "split": "train",
                            }
                        ],
                        "miConfig": {
                            "facet": {"id": "coder.python", "version": "1"}
                        },
                    },
                ],
                "training": {"maxSteps": 1},
                "checkpoint": {"outputDirectory": "out"},
            },
        }
    )
    assert [
        item.dataset_id for item in _facet_dataset_uses(campaign, "coder")
    ] == ["coder-data"]
    assert [
        item.dataset_id
        for item in _facet_dataset_uses(campaign, "coder.python")
    ] == ["coder-python-data"]


def test_campaign_requires_dataset_for_every_facet_without_global_fallback() -> None:
    with pytest.raises(ValidationError):
        Campaign.model_validate(
            {
                "metadata": {"id": "missing-child-data"},
                "spec": {
                    "topologyId": "topology",
                    "datasets": [],
                    "facets": [
                        {
                            "id": "coder",
                            "datasets": [
                                {"datasetId": "coder-data", "split": "train"}
                            ],
                            "miConfig": {
                                "facet": {"id": "coder", "version": "1"}
                            },
                        },
                        {
                            "id": "coder.python",
                            "dependsOn": ["coder"],
                            "miConfig": {
                                "facet": {
                                    "id": "coder.python",
                                    "version": "1",
                                }
                            },
                        },
                    ],
                    "training": {"maxSteps": 1},
                    "checkpoint": {"outputDirectory": "out"},
                },
            }
        )
