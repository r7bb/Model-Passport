"""OpenAI-style chat completions through the guard (the format most providers accept).

OpenAI, Anthropic, Google Gemini, Mistral, Groq, Together, vLLM, and Ollama all offer an
OpenAI-compatible chat endpoint, so one format covers them.
"""

from __future__ import annotations

import json
from typing import Any

from model_passport.guard.engine import Guard, Report, Vault

PLACEHOLDER_NOTE = (
    "Some personal details in this conversation were replaced with placeholders such as "
    "[EMAIL_1]. When you refer to one, write the placeholder exactly as it appears."
)


def _json_escape(value: str) -> str:
    return json.dumps(value)[1:-1]


def _protect_part(guard: Guard, part: Any, vault: Vault, report: Report) -> Any:
    if isinstance(part, dict) and part.get("type") == "text":
        return {**part, "text": guard.protect(str(part.get("text", "")), vault, report)}
    return part


def _protect_content(guard: Guard, content: Any, vault: Vault, report: Report) -> Any:
    if isinstance(content, str):
        return guard.protect(content, vault, report)
    if isinstance(content, list):
        return [_protect_part(guard, part, vault, report) for part in content]
    return content


def protect_messages(
    guard: Guard, messages: list[dict[str, Any]], vault: Vault, report: Report
) -> list[dict[str, Any]]:
    """Messages as the model may see them, plus a note when anything was masked."""
    protected = []
    for message in messages:
        out = dict(message)
        if "content" in out:
            out["content"] = _protect_content(guard, out["content"], vault, report)
        if isinstance(out.get("tool_calls"), list):
            out["tool_calls"] = [
                _protect_call(guard, call, vault, report) for call in out["tool_calls"]
            ]
        protected.append(out)
    if len(vault):
        protected.insert(0, {"role": "system", "content": PLACEHOLDER_NOTE})
    return protected


def _protect_call(guard: Guard, call: Any, vault: Vault, report: Report) -> Any:
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
        return call
    arguments = guard.protect(function["arguments"], vault, report)
    return {**call, "function": {**function, "arguments": arguments}}


def review_completion(
    guard: Guard, completion: dict[str, Any], vault: Vault, report: Report
) -> dict[str, Any]:
    """The reply as the user may see it: leaks handled and placeholders restored."""
    for choice in completion.get("choices") or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            continue
        if isinstance(message.get("content"), str):
            message["content"] = guard.review(message["content"], vault, report)
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                function["arguments"] = guard.review(
                    function["arguments"], vault, report, escape=_json_escape
                )
    return completion
