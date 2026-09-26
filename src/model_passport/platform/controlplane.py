"""Client for the Go control plane (deployments, rollback, the kill switch) over gRPC.

The backend forwards the caller's own session token and organization, so the control plane
applies that person's role, exactly as if they had used the ``mp`` CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import grpc

from model_passport.platform.rpc import controlplane_pb2 as pb
from model_passport.platform.rpc import controlplane_pb2_grpc as rpc

ENVIRONMENTS = {"dev": pb.DEV, "consumer": pb.CONSUMER}
TIMEOUT = 15.0


class ControlPlaneError(RuntimeError):
    """The control plane refused or failed; ``code`` is the gRPC status name."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _deployment(d: Any) -> dict[str, Any]:
    return {
        "id": d.id, "version_id": d.version_id, "model_id": d.model_id, "model": d.model,
        "version": d.version, "environment": pb.Environment.Name(d.environment).lower(),
        "status": d.status, "endpoint": d.endpoint, "created_at": d.created_at,
    }  # fmt: skip


@dataclass
class ControlPlane:
    """One caller's view of the control plane at ``address`` (host:port)."""

    address: str
    token: str
    tenant: str

    def _call(self, method: str, request: Any) -> Any:
        metadata = (("authorization", f"Bearer {self.token}"), ("x-mp-tenant", self.tenant))
        with grpc.insecure_channel(self.address) as channel:
            stub = rpc.ControlPlaneStub(channel)
            try:
                return getattr(stub, method)(request, metadata=metadata, timeout=TIMEOUT)
            except grpc.RpcError as exc:
                code = exc.code().name if exc.code() else "UNKNOWN"
                raise ControlPlaneError(code, exc.details() or str(exc)) from exc

    def deploy(self, version_id: str, environment: str) -> dict[str, Any]:
        request = pb.DeployRequest(version_id=version_id, environment=ENVIRONMENTS[environment])
        return _deployment(self._call("Deploy", request))

    def rollback(self, model_id: str) -> dict[str, Any]:
        return _deployment(self._call("Rollback", pb.RollbackRequest(model_id=model_id)))

    def kill(self, version_id: str, reason: str) -> dict[str, Any]:
        out = self._call("Kill", pb.KillRequest(version_id=version_id, reason=reason))
        return {"deployments_stopped": out.deployments_stopped, "version_state": out.version_state}

    def deployments(self, model_id: str = "") -> list[dict[str, Any]]:
        out = self._call("ListDeployments", pb.ListDeploymentsRequest(model_id=model_id))
        return [_deployment(d) for d in out.deployments]

    def resolve(self, model: str, environment: str) -> dict[str, Any]:
        out = self._call(
            "Resolve", pb.ResolveRequest(model=model, environment=ENVIRONMENTS[environment])
        )
        return {"found": out.found, "killed": out.killed, "version_id": out.version_id,
                "version": out.version, "endpoint": out.endpoint}  # fmt: skip
