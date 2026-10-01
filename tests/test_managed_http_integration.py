from __future__ import annotations

import os
import pytest

if os.environ.get("JESTER_CROSS_REPO_TESTS") != "1":
    pytest.skip("cross-repo RM/component integration; set JESTER_CROSS_REPO_TESTS=1", allow_module_level=True)

import json
import os
import threading
import sysconfig
from importlib.metadata import distribution
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import jester_rm
from jester_rm.config import Settings
from jester_rm.controller.service import ControllerService
from jester_rm.models import Deployment, EntityMetadata, HostSpec, Lease

from fastapi.testclient import TestClient
from jester_trainer.channel_sessions import ManagedChannelContext, ManagedChannelGuard
from jester_trainer.api import create_app
from tests.helpers import campaign


def _assert_installed_rm_origin() -> None:
    module_file = Path(jester_rm.__file__).resolve()
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    old_rm_checkout = Path(__file__).resolve().parents[3] / "packages" / "rm"
    assert module_file.is_relative_to(site_packages), module_file
    assert not module_file.is_relative_to(old_rm_checkout), module_file
    installed = distribution("jester-resource-manager")
    files = tuple(installed.files or ())
    module_record = next(entry for entry in files if str(entry).replace("\\", "/") == "jester_rm/__init__.py")
    record = next(entry for entry in files if entry.name == "RECORD")
    assert Path(installed.locate_file(module_record)).resolve() == module_file
    assert Path(installed.locate_file(record)).resolve().is_relative_to(site_packages)


_assert_installed_rm_origin()


class _RouteService: pass


def _post(client, path, credential, payload=None):
    return client.post(path, headers={"X-Jester-Component-Session":credential}, json=payload)


def test_rm_issued_session_controls_actual_trainer_campaign_handlers(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JESTER_RM_HOME", str(tmp_path / "rm-home"))
    monkeypatch.setenv("JESTER_RM_AGENT_HOME", str(tmp_path / "rm-agent"))
    controller = ControllerService(Settings.load(create=True))
    server = None
    try:
        host = HostSpec(metadata=EntityMetadata(id="host-one"), transport="local", status="online")
        controller.put_host(host)
        from jester_rm.models import ComponentSpec, InstanceBody, InstanceSpec, ResourceRequirements
        def instance(identifier: str, adapter: str, port: int) -> InstanceSpec:
            return InstanceSpec(metadata=EntityMetadata(id=identifier), spec=InstanceBody(
                hostId="host-one", component=ComponentSpec(adapter=adapter), resources=ResourceRequirements.model_validate({"cpu":{},"memory":{"ramBytes":1},"storage":{"poolId":"pool-one","quotaBytes":1}}), network={"port": port},
            ))
        for item in (instance("engine-one", "engine.serve", 8780), instance("trainer-one", "trainer.serve", 9123)):
            controller.store.put_entity("instance", item.metadata.id, item.api_dict(), expected_generation=0)
        deployment = Deployment.model_validate({"metadata":{"id":"deployment-one"}, "spec":{
            "projectId":"project-one", "engineInstanceId":"engine-one", "members":[{"memberId":"trainer","instanceId":"trainer-one"}],
            "controllerChannels":[{"channelId":"engine-trainer","destinationMemberId":"trainer","destinationPort":"http","purpose":"training","allowedOperations":["trainer.dataset:register", "trainer.dataset:read", "trainer.campaign:create", "trainer.campaign:status"]}],
        }})
        stored = controller.put_deployment(deployment)
        running = Deployment.model_validate(stored).model_copy(update={"state":"running"})
        running = Deployment.model_validate(controller.store.put_entity("deployment", "deployment-one", running.api_dict(), expected_generation=1))
        for lease_id, instance_id, port in (("engine-lease", "engine-one", 8780), ("trainer-lease", "trainer-one", 9123)):
            controller.store.put_lease(Lease(leaseId=lease_id, instanceId=instance_id, instanceGeneration=1, hostId="host-one", hostEpoch="epoch", state="active", configSha256="hash", telemetryToken="token", paths={"root":str(tmp_path / instance_id)}, ports={"http":port}).api_dict())
        grant = controller._issue_engine_grant(running)
        first = controller.controller_channel_sessions(grant["credential"], project_id="project-one", deployment_id="deployment-one")["items"][0]
        def local(operation: str, method: str):
            return {"destinationInstanceId":"trainer-one","destinationInstanceGeneration":1,"destinationLeaseId":"trainer-lease","operation":operation,"method":method}
        guard = ManagedChannelGuard(ManagedChannelContext(rm_url="http://unused"), introspect=lambda credential, expected: controller.introspect_controller_channel_session(credential, expectation=expected) or {}, expectation_provider=local)
        from jester_trainer.config import Settings as TrainerSettings
        from jester_trainer.service import TrainerService
        data = tmp_path / "train.jsonl"; data.write_text('{"id":"one","text":"x"}\n', encoding="utf-8")
        import hashlib
        digest = hashlib.sha256(data.read_bytes()).hexdigest()
        artifact_manifest = tmp_path / "trainer-artifacts.json"
        artifact_manifest.write_text(json.dumps({"entries": [{
            "digest": f"sha256:{digest}", "kind": "file", "checkoutPath": str(data),
        }]}), encoding="utf-8")
        monkeypatch.setenv("JESTER_RM_ARTIFACT_MANIFEST", str(artifact_manifest))
        service = TrainerService(TrainerSettings.for_test(tmp_path / "trainer", topology={"topologyId":"test-training-topology","topologyType":"training","role":"trainer","nodeId":"trainer","channels":[],"nodes":[]}))
        body = campaign(tmp_path, tmp_path / "model", campaign_id="campaign-one", version="v1", workers=[]).api_dict()
        with TestClient(create_app(service, managed_channel_guard=guard, require_managed_session=True)) as client:
            registered = _post(client, "/v1/datasets", first["credential"], {
                "metadata": {"id": "tiny-dataset"},
                "files": [{
                    "split": "train", "digest": f"sha256:{digest}",
                    "artifactDigest": f"sha256:{digest}", "logicalPath": "train.jsonl",
                }],
            })
            assert registered.status_code == 201
            created = _post(client, "/v1/campaigns", first["credential"], body)
            assert created.status_code == 201
            assert client.get("/v1/campaigns/campaign-one", headers={"X-Jester-Component-Session":first["credential"]}).status_code == 200
            renewed = controller.controller_channel_sessions(grant["credential"], project_id="project-one", deployment_id="deployment-one")["items"][0]
            assert client.get("/v1/campaigns/campaign-one", headers={"X-Jester-Component-Session":first["credential"]}).status_code == 401
            assert client.get("/v1/campaigns/campaign-one", headers={"X-Jester-Component-Session":renewed["credential"]}).status_code == 200
            guard._expectation_provider = lambda operation, method: {**local(operation, method), "destinationLeaseId":"wrong-lease"}
            assert client.get("/v1/campaigns/campaign-one", headers={"X-Jester-Component-Session":renewed["credential"]}).status_code == 401
    finally:
        
        controller.close()



def test_campaign_owner_is_immutable_across_store_reopen(tmp_path):
    from jester_trainer.db import StateStore
    from jester_trainer.errors import ConflictError
    path = tmp_path / "owners.sqlite3"
    first = StateStore(path)
    first.bind_campaign_owner("campaign-one", project_id="project-a", deployment_id="deployment-a")
    assert first.campaign_owner("campaign-one") == ("project-a", "deployment-a")
    first.close()
    reopened = StateStore(path)
    assert reopened.campaign_owner("campaign-one") == ("project-a", "deployment-a")
    with pytest.raises(ConflictError) as conflict:
        reopened.bind_campaign_owner("campaign-one", project_id="project-b", deployment_id="deployment-b")
    assert conflict.value.code == "campaign_owner_conflict"
    reopened.close()
