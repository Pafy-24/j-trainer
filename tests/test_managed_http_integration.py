from __future__ import annotations

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


class _RouteService:
    def __init__(self, root):
        from jester_trainer.db import StateStore
        self.token="legacy"; self.campaigns={}; self.store=StateStore(root / "trainer.sqlite3")
    def close(self): pass
    def health(self): return {"status":"ok"}
    def idempotent(self, _scope, _key, _payload, action): return action()
    def put_campaign(self, item): self.campaigns[item.metadata.id]=item; return item.api_dict()
    def get_campaign(self, identifier): return self.campaigns[identifier]


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
        service = _RouteService(tmp_path)
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

