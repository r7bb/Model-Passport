"""Incoming request bodies reject oversized strings (and accept ones exactly at the limit)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from model_passport.platform.api.schemas import GuardSettingsIn, Login, MemberIn, TenantIn

_VALID: dict[type, dict[str, Any]] = {
    Login: {"email": "a@b.co", "password": "pw"},
    TenantIn: {"slug": "acme", "name": "Acme"},
    MemberIn: {"email": "a@b.co", "role": "org_admin", "name": "A", "password": "pw"},
    GuardSettingsIn: {"upstream_url": "https://x.test", "upstream_key": "k", "default_model": "m"},
}

_LIMITS = [
    (Login, "email", 254),
    (Login, "password", 1024),
    (TenantIn, "slug", 63),
    (TenantIn, "name", 200),
    (MemberIn, "email", 254),
    (MemberIn, "name", 200),
    (MemberIn, "password", 1024),
    (GuardSettingsIn, "upstream_url", 2048),
    (GuardSettingsIn, "upstream_key", 4096),
    (GuardSettingsIn, "default_model", 200),
]


@pytest.mark.parametrize(("model", "field", "limit"), _LIMITS)
def test_value_at_the_limit_is_accepted(model: type, field: str, limit: int) -> None:
    body = {**_VALID[model], field: "a" * limit}
    assert getattr(model(**body), field) == "a" * limit


@pytest.mark.parametrize(("model", "field", "limit"), _LIMITS)
def test_value_over_the_limit_is_rejected(model: type, field: str, limit: int) -> None:
    body = {**_VALID[model], field: "a" * (limit + 1)}
    with pytest.raises(ValidationError) as caught:
        model(**body)
    assert caught.value.errors()[0]["loc"] == (field,)
    assert caught.value.errors()[0]["type"] == "string_too_long"
