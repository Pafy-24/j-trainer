from __future__ import annotations

import os
import pytest

if os.environ.get("JESTER_CROSS_REPO_TESTS") != "1":
    pytest.skip("cross-repo RM/component integration; set JESTER_CROSS_REPO_TESTS=1", allow_module_level=True)

import json
import os
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve()
PACKAGES = HERE.parents[1].parent
WORKSPACE = PACKAGES.parents[1]
for source in (PACKAGES / "rm" / "src", WORKSPACE / "jester-core" / "packages" / "python" / "core" / "src"):
    sys.path.insert(0, str(source))

from jester_rm.config import Settings
from jester_rm.controller.service import ControllerService
from jester_rm.models import Deployment, EntityMetadata, HostSpec, Lease

from fastapi.testclient import TestClient
from jester_trainer.channel_sessions import ManagedChannelContext, ManagedChannelGuard
from jester_trainer.api import create_app
from tests.helpers import campaign


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
            "controllerChannels":[{"channelId":"engine-trainer","destinationMemberId":"trainer","destinationPort":"http","purpose":"inference","allowedOperations":["trainer.campaign:create", "trainer.campaign:status"]}],
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
        from tests.helpers import dataset_manifest
        data = tmp_path / "train.jsonl"; data.write_text('{"id":"one","text":"x"}\n', encoding="utf-8")
        service = TrainerService(TrainerSettings.for_test(tmp_path / "trainer", topology={"topologyId":"test-training-topology","topologyType":"training","role":"trainer","nodeId":"trainer","channels":[],"nodes":[]}))
        service.put_dataset(dataset_manifest(data))
        body = campaign(tmp_path, tmp_path / "model", campaign_id="campaign-one", version="v1", workers=[]).api_dict()
        with TestClient(create_app(service, managed_channel_guard=guard, require_managed_session=True)) as client:
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
