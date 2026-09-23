"""Helpers for pipeline scripts executed by ``passport run``."""

from __future__ import annotations

import json
import os
from typing import Any

from model_passport.core.capture import PARAMS_ENV


def params(defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """Stage parameters from ``passport run``, merged over ``defaults``.

    Scripts can still be run directly (without ``passport run``); they then get ``defaults``.
    """
    merged = dict(defaults or {})
    raw = os.environ.get(PARAMS_ENV)
    if raw:
        merged.update(json.loads(raw))
    return merged
