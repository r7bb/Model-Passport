"""Minimal registry client using only the standard library."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, cast

TOKEN_ENV = "PASSPORT_REGISTRY_TOKEN"  # noqa: S105 - env var name
URL_ENV = "PASSPORT_REGISTRY_URL"
DEFAULT_URL = "http://localhost:8000"


class RegistryError(Exception):
    """Raised for HTTP or connection failures, with the server's explanation when available."""


class RegistryClient:
    def __init__(self, url: str | None = None, token: str | None = None, timeout: float = 30):
        self.url = (url or os.environ.get(URL_ENV) or DEFAULT_URL).rstrip("/")
        if urllib.parse.urlparse(self.url).scheme not in {"http", "https"}:
            raise RegistryError(f"registry URL must be http(s): {self.url}")
        self.token = token or os.environ.get(TOKEN_ENV)
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(  # noqa: S310 - http(s) enforced in __init__
            f"{self.url}{path}", data=data, headers=headers, method=method
        )
        try:
            # The URL scheme is restricted to http(s) in __init__.
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:  # noqa: S310
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RegistryError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RegistryError(f"cannot reach registry at {self.url}: {exc}") from exc

    def upload(self, passport: dict[str, Any], public_key_pem: str) -> dict[str, Any]:
        body = {"passport": passport, "public_key_pem": public_key_pem}
        return cast(dict[str, Any], self._request("POST", "/passports", body))

    def append_event(self, passport_id: str, event: dict[str, Any]) -> dict[str, Any]:
        path = f"/passports/{passport_id}/events"
        return cast(dict[str, Any], self._request("POST", path, {"event": event}))

    def search(self, **filters: str) -> list[dict[str, Any]]:
        query = "&".join(f"{k}={v}" for k, v in filters.items() if v)
        path = f"/passports{'?' + query if query else ''}"
        return cast(list[dict[str, Any]], self._request("GET", path))

    def get(self, passport_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], self._request("GET", f"/passports/{passport_id}"))

    def verify(self, passport_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], self._request("GET", f"/passports/{passport_id}/verify"))

    def lineage(self, passport_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], self._request("GET", f"/passports/{passport_id}/lineage"))

    def dag(self, passport_id: str) -> dict[str, str]:
        return cast(dict[str, str], self._request("GET", f"/passports/{passport_id}/dag"))
