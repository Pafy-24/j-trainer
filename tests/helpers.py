from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

# Resolve Transformers' lazy Auto* imports on the main test thread. The real
# deployment uses one process per worker; the in-process two-worker harness
# would otherwise race native optional-dependency imports on Windows.
_ = AutoModelForCausalLM.from_pretrained, AutoTokenizer.from_pretrained

from jester_trainer.config import Settings
from jester_trainer.models import Campaign, DatasetManifest
from jester_trainer.service import TrainerService
from jester_trainer.util import sha256_file
from mixed_inference.worker.server import MIWorkerHTTPServer
from mixed_inference.worker.service import MIWorkerService


def tiny_model(path: Path) -> None:
    path.mkdir(parents=True)
    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "hello": 4,
        "world": 5,
        "write": 6,
        "code": 7,
        "python": 8,
        "answer": 9,
        "verify": 10,
        "plan": 11,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(path)
    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocabulary),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
        )
    )
    model.save_pretrained(path)


def training_dataset(path: Path) -> None:
    records = [
        {"id": "one", "semanticCaseId": "case-one", "text": "hello world write code answer"},
        {"id": "two", "semanticCaseId": "case-two", "text": "write python code answer"},
        {"id": "three", "semanticCaseId": "case-three", "text": "verify python code answer"},
        {"id": "four", "semanticCaseId": "case-four", "text": "plan write code answer"},
    ]
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )


@contextmanager
def worker_topology(root: Path, count: int = 2) -> Iterator[dict]:
    servers: list[MIWorkerHTTPServer] = []
    threads: list[threading.Thread] = []
    nodes = [
        {
            "nodeId": "trainer",
            "instanceId": "trainer-instance",
            "role": "trainer",
            "requestedBackend": "cpu",
            "effectiveBackend": "cpu",
        }
    ]
    channels = []
    try:
        for index in range(count):
            worker_id = f"worker-{index + 1}"
            service = MIWorkerService(root / worker_id, require_rm=False)
            server = MIWorkerHTTPServer(("127.0.0.1", 0), service)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            servers.append(server)
            threads.append(thread)
            nodes.append(
                {
                    "nodeId": worker_id,
                    "instanceId": f"{worker_id}-instance",
                    "role": "mi.worker",
                    "requestedBackend": "cpu",
                    "effectiveBackend": "cpu",
                }
            )
            channels.append(
                {
                    "channelId": f"trainer-to-{worker_id}",
                    "sourceNodeId": "trainer",
                    "destinationNodeId": worker_id,
                    "protocol": "http-json+binary",
                    "transport": "local-loopback",
                    "endpoint": {"host": "127.0.0.1", "port": server.server_port},
                    "state": "active",
                    "resumable": True,
                }
            )
        yield {
            "apiVersion": "jester.rm/topology-context-v1",
            "topologyId": "test-training-topology",
            "topologyType": "training",
            "role": "trainer",
            "nodeId": "trainer",
            "nodes": nodes,
            "channels": channels,
        }
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)


def dataset_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest.model_validate(
        {
            "metadata": {"id": "tiny-dataset", "displayName": "Tiny deterministic data"},
            "files": [
                {
                    "split": "train",
                    "path": str(path),
                    "digest": f"sha256:{sha256_file(path)}",
                }
            ],
        }
    )


def campaign(
    root: Path,
    model: Path,
    *,
    campaign_id: str,
    version: str,
    workers: list[str],
) -> Campaign:
    return Campaign.model_validate(
        {
            "metadata": {"id": campaign_id, "displayName": campaign_id},
            "spec": {
                "topologyId": "test-training-topology",
                "datasets": [{"datasetId": "tiny-dataset", "split": "train"}],
                "facets": [
                    {
                        "id": "tiny-coder",
                        "miConfig": {
                            "schemaVersion": "mi-training-v1",
                            "baseModel": {
                                "id": str(model),
                                "device": "cpu",
                                "dtype": "float32",
                            },
                            "registry": str(root / "registry"),
                            "facet": {
                                "id": "tiny-coder",
                                "version": version,
                                "description": "Distributed parity test",
                                "dependencies": {},
                                "module": {
                                    "type": "full-decoder-block",
                                    "slots": [
                                        {
                                            "id": "coder-slot",
                                            "afterLayer": 0,
                                            "blocks": [
                                                {
                                                    "sourceLayer": 0,
                                                    "initializer": "identity-copy",
                                                }
                                            ],
                                        }
                                    ],
                                },
                            },
                            "data": {"train": [], "maxSequenceLength": 32},
                        },
                    }
                ],
                "mode": "parameter-server",
                "cohort": {"workerIds": workers},
                "optimizer": {
                    "type": "sgd",
                    "learningRate": 0.001,
                    "weightDecay": 0.0,
                    "momentum": 0.0,
                },
                "scheduler": {"type": "constant"},
                "training": {
                    "maxSteps": 1,
                    "examplesPerWorker": 4,
                    "batchSize": 2,
                    "precision": "float32",
                    "maxGradNorm": 0.0,
                    "seed": 42,
                    "transportDtype": "float32",
                    "compression": "none",
                },
                "checkpoint": {
                    "outputDirectory": str(root / "results"),
                    "saveEveryRounds": 1,
                    "maxRetries": 0,
                },
            },
        }
    )


def run_campaign(service: TrainerService, campaign_model: Campaign, timeout: float = 90) -> dict:
    service.put_campaign(campaign_model)
    service.plan_campaign(campaign_model.metadata.id)
    operation = service.start_campaign(campaign_model.metadata.id)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = service.get_operation(operation["operationId"])
        if current["state"] in {"succeeded", "failed", "cancelled"}:
            if current["state"] != "succeeded":
                raise AssertionError(current)
            return current
        time.sleep(0.05)
    raise AssertionError("Trainer campaign did not finish before timeout")


def artifact_weights(operation: dict) -> dict[str, torch.Tensor]:
    result = operation["result"]["results"][0]
    return load_file(str(Path(result["artifactPath"]) / "facet.safetensors"), device="cpu")
