from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from importlib import metadata
from pathlib import Path

import pytest
from jester_core import ComponentManifest, component_identity
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


PACKAGE = Path(__file__).parents[1]
RESOURCE_ROOT = PACKAGE / "src" / "jester_trainer" / "resources"


def _json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def test_manifest_binds_exact_resources_and_dedicated_capability() -> None:
    manifest = ComponentManifest.model_validate(_json(RESOURCE_ROOT / "component-manifest.json"))
    assert (manifest.id, manifest.version) == ("trainer", "0.1.0")
    assert component_identity(manifest).manifest_sha256 == manifest.semantic_sha256
    binding = manifest.capabilities[0]
    assert binding.capability.id == "trainer.campaign"
    assert binding.capability.concurrency.sharing == "dedicated-only"
    assert binding.capability.concurrency.max_concurrent_requests == 1
    assert binding.configuration_defaults_resource_id is None
    for resource in manifest.resources:
        path = RESOURCE_ROOT.parent / resource.installed_path
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == resource.sha256


def test_serve_configuration_is_empty_and_does_not_absorb_campaign_or_rm_state() -> None:
    schema = _json(RESOURCE_ROOT / "trainer-serve-configuration-v1.schema.json")
    validator = Draft202012Validator(schema)
    validator.validate({})
    for payload in (
        {"host": "127.0.0.1"},
        {"port": 8140},
        {"topologyId": "training.one"},
        {"role": "trainer"},
        {"leaseId": "lease.one"},
        {"datasetId": "dataset.one"},
        {"seed": 42},
        {"campaign": {}},
    ):
        with pytest.raises(ValidationError):
            validator.validate(payload)


def test_mapping_is_closed_and_leaves_rm_owned_arguments_out() -> None:
    mapping = _json(RESOURCE_ROOT / "trainer-serve-adapter-mapping-v1.json")
    assert mapping == {
        "schemaVersion": "jester-adapter-mapping-v1",
        "id": "trainer.serve",
        "version": "1",
        "adapter": "trainer.serve",
        "assetRoles": {},
        "arguments": [],
        "environment": {},
        "additionalArguments": False,
    }


def test_capability_operations_match_actual_managed_api_routes() -> None:
    contract = _json(RESOURCE_ROOT / "trainer-campaign-capability-v1.json")
    expected = {
        ("trainer.dataset:register", "POST", "/v1/datasets"),
        ("trainer.dataset:read", "GET", "/v1/datasets/{dataset_id}"),
        ("trainer.campaign:create", "POST", "/v1/campaigns"),
        ("trainer.campaign:status", "GET", "/v1/campaigns/{campaign_id}"),
        ("trainer.campaign:start", "POST", "/v1/campaigns/{campaign_id}/actions/start"),
        ("trainer.campaign:cancel", "POST", "/v1/campaigns/{campaign_id}/actions/cancel"),
        ("trainer.campaign:execution", "GET", "/v1/campaigns/{campaign_id}/execution"),
        (
            "trainer.campaign:publication-readback",
            "GET",
            "/v1/campaigns/{campaign_id}/results/{result_id}/publications/{publication_id}/readback",
        ),
    }
    declared = {
        (item["channelOperation"], item["method"], item["path"])
        for item in contract["operations"]
    }
    assert declared == {
        (operation, method, path.replace("_id", "Id"))
        for operation, method, path in expected
    }

    tree = ast.parse((PACKAGE / "src" / "jester_trainer" / "api.py").read_text(encoding="utf-8"))
    actual = set()
    for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
        operation = None
        for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
            name = call.func.id if isinstance(call.func, ast.Name) else None
            if name == "authorize_managed_channel":
                value = next((item.value for item in call.keywords if item.arg == "operation"), None)
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    operation = value.value
                    break
        if operation is None:
            continue
        for decorator in function.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                continue
            method = decorator.func.attr.upper()
            if method not in {"GET", "POST"} or not decorator.args:
                continue
            path = decorator.args[0]
            if isinstance(path, ast.Constant) and isinstance(path.value, str):
                actual.add((operation, method, path.value))
    assert expected <= actual


def test_daemon_requires_rm_training_topology_and_trainer_role() -> None:
    source = (PACKAGE / "src" / "jester_trainer" / "config.py").read_text(encoding="utf-8")
    assert 'topology.get("topologyType") != "training"' in source
    assert 'topology.get("role") != "trainer"' in source
    assert "effective.run_directory / \"trainer\"" in source
    assert "effective.lease.get(\"ports\", {}).get(\"http\", 8140)" in source


@pytest.mark.skipif(
    os.environ.get("JESTER_TRAINER_INSTALLED_RESOURCE_PROOF") != "1",
    reason="runs only in the isolated installed-wheel acceptance environment",
)
def test_installed_resources_without_importing_trainer_or_torch() -> None:
    assert "jester_trainer" not in sys.modules and "torch" not in sys.modules
    distribution = metadata.distribution("jester-trainer")
    package_root = Path(distribution.locate_file("jester_trainer")).resolve()
    manifest = ComponentManifest.model_validate(
        _json(package_root / "resources" / "component-manifest.json")
    )
    assert "jester_trainer" not in sys.modules and "torch" not in sys.modules
    installed = {str(item).replace("\\", "/") for item in distribution.files or ()}
    for resource in manifest.resources:
        relative = f"jester_trainer/{resource.installed_path}"
        assert relative in installed
        path = Path(distribution.locate_file(relative)).resolve()
        assert path.is_relative_to(package_root)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == resource.sha256
    assert "jester_trainer" not in sys.modules and "torch" not in sys.modules
