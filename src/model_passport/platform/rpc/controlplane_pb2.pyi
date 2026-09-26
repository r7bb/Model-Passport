from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Environment(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ENVIRONMENT_UNSPECIFIED: _ClassVar[Environment]
    DEV: _ClassVar[Environment]
    CONSUMER: _ClassVar[Environment]
ENVIRONMENT_UNSPECIFIED: Environment
DEV: Environment
CONSUMER: Environment

class Deployment(_message.Message):
    __slots__ = ("id", "version_id", "model_id", "model", "version", "environment", "status", "endpoint", "created_at")
    ID_FIELD_NUMBER: _ClassVar[int]
    VERSION_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    id: str
    version_id: str
    model_id: str
    model: str
    version: str
    environment: Environment
    status: str
    endpoint: str
    created_at: str
    def __init__(self, id: _Optional[str] = ..., version_id: _Optional[str] = ..., model_id: _Optional[str] = ..., model: _Optional[str] = ..., version: _Optional[str] = ..., environment: _Optional[_Union[Environment, str]] = ..., status: _Optional[str] = ..., endpoint: _Optional[str] = ..., created_at: _Optional[str] = ...) -> None: ...

class DeployRequest(_message.Message):
    __slots__ = ("version_id", "environment")
    VERSION_ID_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    version_id: str
    environment: Environment
    def __init__(self, version_id: _Optional[str] = ..., environment: _Optional[_Union[Environment, str]] = ...) -> None: ...

class RollbackRequest(_message.Message):
    __slots__ = ("model_id",)
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class KillRequest(_message.Message):
    __slots__ = ("version_id", "reason")
    VERSION_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    version_id: str
    reason: str
    def __init__(self, version_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class KillResponse(_message.Message):
    __slots__ = ("deployments_stopped", "version_state")
    DEPLOYMENTS_STOPPED_FIELD_NUMBER: _ClassVar[int]
    VERSION_STATE_FIELD_NUMBER: _ClassVar[int]
    deployments_stopped: int
    version_state: str
    def __init__(self, deployments_stopped: _Optional[int] = ..., version_state: _Optional[str] = ...) -> None: ...

class ListDeploymentsRequest(_message.Message):
    __slots__ = ("model_id",)
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class ListDeploymentsResponse(_message.Message):
    __slots__ = ("deployments",)
    DEPLOYMENTS_FIELD_NUMBER: _ClassVar[int]
    deployments: _containers.RepeatedCompositeFieldContainer[Deployment]
    def __init__(self, deployments: _Optional[_Iterable[_Union[Deployment, _Mapping]]] = ...) -> None: ...

class ResolveRequest(_message.Message):
    __slots__ = ("model", "environment")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    model: str
    environment: Environment
    def __init__(self, model: _Optional[str] = ..., environment: _Optional[_Union[Environment, str]] = ...) -> None: ...

class ResolveResponse(_message.Message):
    __slots__ = ("found", "killed", "version_id", "version", "endpoint")
    FOUND_FIELD_NUMBER: _ClassVar[int]
    KILLED_FIELD_NUMBER: _ClassVar[int]
    VERSION_ID_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    found: bool
    killed: bool
    version_id: str
    version: str
    endpoint: str
    def __init__(self, found: _Optional[bool] = ..., killed: _Optional[bool] = ..., version_id: _Optional[str] = ..., version: _Optional[str] = ..., endpoint: _Optional[str] = ...) -> None: ...
