"""Encrypted object storage for datasets and model files.

Objects live under ``<tenant id>/...`` in a local directory or an S3 bucket (SeaweedFS, MinIO,
AWS S3). ``TenantStore`` encrypts every object with the tenant's data key and binds the
ciphertext to its tenant and key, so an object copied to another tenant or path does not
decrypt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from model_passport.platform.security import decrypt, encrypt


class StorageError(RuntimeError):
    """The object store is unreachable or the object does not exist."""


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...


def _safe(key: str) -> str:
    parts = [p for p in key.split("/") if p]
    if not parts or any(p in {".", ".."} for p in parts):
        raise StorageError(f"invalid object key {key!r}")
    return "/".join(parts)


class LocalStore:
    """Objects as files under a directory (development and single-machine installs)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / _safe(key)

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise StorageError(f"object not found: {key}")
        return path.read_bytes()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class S3Store:
    """Objects in an S3 bucket; ``endpoint`` points at SeaweedFS or another S3 service."""

    def __init__(self, bucket: str, endpoint: str | None = None) -> None:
        try:
            import boto3  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise StorageError(
                "S3 storage needs boto3: pip install 'model-passport[platform]'"
            ) from exc
        self.bucket = bucket
        self.client: Any = boto3.client("s3", endpoint_url=endpoint)

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=_safe(key), Body=data)

    def get(self, key: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=_safe(key))
        except Exception as exc:  # botocore raises ClientError subclasses
            raise StorageError(f"object not found: {key}") from exc
        body: bytes = response["Body"].read()
        return body

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=_safe(key))


def open_store(url: str, endpoint: str | None = None) -> ObjectStore:
    """``file:///path`` or ``file://./relative``, or ``s3://bucket``."""
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return LocalStore(Path((parsed.netloc or "") + parsed.path).expanduser())
    if parsed.scheme == "s3":
        return S3Store(parsed.netloc, endpoint)
    raise StorageError(f"unsupported storage URL {url!r}; use file:// or s3://")


class TenantStore:
    """One tenant's view of the store: keys are prefixed and objects encrypted."""

    def __init__(self, store: ObjectStore, tenant_id: str, data_key: bytes) -> None:
        self.store = store
        self.tenant_id = tenant_id
        self.key = data_key

    def _key(self, key: str) -> str:
        return f"{self.tenant_id}/{_safe(key)}"

    def _context(self, key: str) -> bytes:
        return f"object:{self._key(key)}".encode()

    def put(self, key: str, data: bytes) -> str:
        self.store.put(self._key(key), encrypt(self.key, data, self._context(key)))
        return key

    def get(self, key: str) -> bytes:
        return decrypt(self.key, self.store.get(self._key(key)), self._context(key))

    def delete(self, key: str) -> None:
        self.store.delete(self._key(key))
