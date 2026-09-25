"""Platform configuration, read from ``MP_*`` environment variables."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path


class SettingsError(ValueError):
    """A required setting is missing or malformed."""


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """``MP_DATABASE_URL``: SQLAlchemy URL (PostgreSQL in production; SQLite for local use).

    ``MP_JWT_SECRET``: key for signing session tokens.

    ``MP_MASTER_KEY``: 32-byte base64 key that wraps each tenant's data key; in the cloud this
    is a KMS key instead.

    ``MP_STORAGE``: ``file:///path`` or ``s3://bucket``; ``MP_S3_ENDPOINT`` points at
    SeaweedFS or another S3 service.

    ``MP_BASE_DOMAIN``: tenants are ``<slug>.<base domain>`` (e.g. ``usps.mp.com``).
    """

    database_url: str
    jwt_secret: str
    master_key: bytes
    storage: str = "file://./mp-data"
    s3_endpoint: str | None = None
    base_domain: str = "mp.localhost"
    token_minutes: int = 60
    work_dir: Path = field(default_factory=lambda: Path("./mp-work"))

    @classmethod
    def from_env(cls) -> Settings:
        url = _env("MP_DATABASE_URL", "sqlite:///./mp.db")
        secret = _env("MP_JWT_SECRET")
        master = _env("MP_MASTER_KEY")
        if not secret or len(secret) < 32:
            raise SettingsError("set MP_JWT_SECRET to at least 32 random characters")
        if not master:
            raise SettingsError("set MP_MASTER_KEY to 32 random bytes, base64 encoded")
        return cls(
            database_url=str(url),
            jwt_secret=secret,
            master_key=decode_key(master),
            storage=str(_env("MP_STORAGE", "file://./mp-data")),
            s3_endpoint=_env("MP_S3_ENDPOINT"),
            base_domain=str(_env("MP_BASE_DOMAIN", "mp.localhost")),
            token_minutes=int(str(_env("MP_TOKEN_MINUTES", "60"))),
            work_dir=Path(str(_env("MP_WORK_DIR", "./mp-work"))),
        )


def decode_key(value: str) -> bytes:
    try:
        key = base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise SettingsError("MP_MASTER_KEY must be base64") from exc
    if len(key) != 32:
        raise SettingsError("MP_MASTER_KEY must decode to 32 bytes (AES-256)")
    return key


def new_key() -> str:
    """A fresh base64 master key, for ``passport platform keygen``."""
    return base64.b64encode(os.urandom(32)).decode()
