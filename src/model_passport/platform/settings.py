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


def _int(name: str, default: int, minimum: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a whole number") from exc
    if value < minimum:
        raise SettingsError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    """``MP_DATABASE_URL``: SQLAlchemy URL (PostgreSQL in production; SQLite for local use).

    ``MP_JWT_SECRET``: key for signing session tokens.

    ``MP_MASTER_KEY``: 32-byte base64 key that wraps each tenant's data key; in the cloud this
    is a KMS key instead.

    ``MP_STORAGE``: ``file:///path`` or ``s3://bucket``; ``MP_S3_ENDPOINT`` points at
    SeaweedFS or another S3 service.

    ``MP_BASE_DOMAIN``: tenants are ``<slug>.<base domain>`` (e.g. ``usps.mp.com``).

    ``MP_CONTROLPLANE_ADDR``: the Go control plane (``host:port``). When set, canary and release
    deploy through it, and rollback and the kill switch go through it.

    ``MP_GUARD_ALLOW_PRIVATE_UPSTREAMS``: let the guard forward to private or local addresses
    (a self-hosted Ollama or vLLM). Off by default, so an organization cannot point the
    platform's server at internal services.

    Limits on the Guard endpoint (``/guard/v1``): ``MP_GUARD_MAX_BODY_BYTES`` (largest request
    body, default 1 MiB), ``MP_GUARD_MAX_TOKENS`` (largest ``max_tokens``; also applied when a
    request names none), ``MP_GUARD_MAX_N`` (largest ``n``, completions per request), and
    ``MP_GUARD_RATE_PER_KEY`` / ``MP_GUARD_RATE_PER_IP`` (requests per minute for one client key
    or one address; 0 turns that limit off).
    """

    database_url: str
    jwt_secret: str
    master_key: bytes
    storage: str = "file://./mp-data"
    s3_endpoint: str | None = None
    base_domain: str = "mp.localhost"
    token_minutes: int = 60
    work_dir: Path = field(default_factory=lambda: Path("./mp-work"))
    controlplane: str | None = None  # host:port of the Go control plane, if deployed
    guard_private_upstreams: bool = False
    guard_max_body_bytes: int = 1024 * 1024
    guard_max_tokens: int = 4096
    guard_max_n: int = 1
    guard_rate_per_key: int = 60
    guard_rate_per_ip: int = 120

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
            controlplane=_env("MP_CONTROLPLANE_ADDR"),
            guard_private_upstreams=_env("MP_GUARD_ALLOW_PRIVATE_UPSTREAMS", "") in ("1", "true"),
            guard_max_body_bytes=_int("MP_GUARD_MAX_BODY_BYTES", 1024 * 1024, 1),
            guard_max_tokens=_int("MP_GUARD_MAX_TOKENS", 4096, 1),
            guard_max_n=_int("MP_GUARD_MAX_N", 1, 1),
            guard_rate_per_key=_int("MP_GUARD_RATE_PER_KEY", 60, 0),
            guard_rate_per_ip=_int("MP_GUARD_RATE_PER_IP", 120, 0),
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
