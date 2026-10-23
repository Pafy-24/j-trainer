from __future__ import annotations
import pytest
from jester_trainer.channel_sessions import ChannelSessionDenied, ManagedChannelContext, ManagedChannelGuard

def _context(): return ManagedChannelContext(rm_url="http://rm.invalid")
def _local(operation, method): return {"destinationInstanceId":"destination-one","destinationInstanceGeneration":4,"destinationLeaseId":"destination-lease","operation":operation,"method":method}
def _claims(e): return {"schemaVersion":"jester-channel-session-claims-v1","active":True,"sessionId":"session-one","projectId":"project-one","deploymentId":"deployment-one","deploymentGeneration":2,"engineInstanceId":"engine-one","engineInstanceGeneration":3,"engineLeaseId":"engine-lease","channelId":"channel-one","destinationMemberId":"destination","destinationInstanceId":e["destinationInstanceId"],"destinationInstanceGeneration":e["destinationInstanceGeneration"],"destinationLeaseId":e["destinationLeaseId"],"operation":e["operation"],"method":e["method"],"expiresAt":"2030-01-01T00:00:00+00:00"}
def test_guard_validates_dynamic_claims():
 guard=ManagedChannelGuard(_context(),introspect=lambda _c,e:_claims(e),expectation_provider=_local)
 result = guard.authorize({"X-Jester-Component-Session":"opaque-session-token-at-least-24-chars"},operation="trainer.campaign:create",method="POST")
 assert result.destination_lease_id == "destination-lease"
 with pytest.raises(ChannelSessionDenied): guard.authorize({},operation="trainer.campaign:create",method="POST")
def test_guard_rejects_wrong_destination_lease():
 def wrong(e):
  c=_claims(e); c["destinationLeaseId"]="wrong"; return c
 guard=ManagedChannelGuard(_context(),introspect=lambda _c,e:wrong(e),expectation_provider=_local)
 with pytest.raises(ChannelSessionDenied): guard.authorize({"X-Jester-Component-Session":"opaque-session-token-at-least-24-chars"},operation="trainer.campaign:create",method="POST")
