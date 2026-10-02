"""MP Guard: what reaches the model, what reaches the user, and what is recorded.

All values are made up.
"""

from __future__ import annotations

import json

import pytest

from model_passport.guard.chat import PLACEHOLDER_NOTE, protect_messages, review_completion
from model_passport.guard.engine import (
    Action,
    BlockedError,
    Guard,
    Policy,
    Report,
    Vault,
    fingerprint,
)

EMAIL, SSN, CARD = "jane.doe@example.com", "123-45-6789", "4111 1111 1111 1111"


def test_masked_values_never_reach_the_model_and_come_back_in_the_reply() -> None:
    guard, vault, report = Guard(), Vault(), Report()
    sent = guard.protect(f"Email {EMAIL} about it, then email {EMAIL} again.", vault, report)
    assert EMAIL not in sent
    assert sent.count("[EMAIL_1]") == 2  # one value, one placeholder
    reply = guard.review("I emailed [EMAIL_1] twice.", vault, report)
    assert reply == f"I emailed {EMAIL} twice."
    assert report.masked == {"EMAIL": 2}


def test_policy_actions_for_requests() -> None:
    policy = Policy(inbound={"SSN": Action.BLOCK, "EMAIL": Action.REDACT})
    with pytest.raises(BlockedError) as blocked:
        Guard(policy).protect(f"My SSN is {SSN}", Vault(), Report())
    assert blocked.value.types == ["SSN"]
    vault, report = Vault(), Report()
    sent = Guard(policy).protect(f"Write to {EMAIL}", vault, report)
    assert sent == "Write to [EMAIL]"  # irreversible
    assert len(vault) == 0
    assert report.redacted == {"EMAIL": 1}
    allowed = Guard(Policy(inbound_default=Action.ALLOW)).protect(EMAIL, Vault(), Report())
    assert allowed == EMAIL


def test_values_the_model_produces_are_handled_by_impact() -> None:
    guard, report = Guard(), Report()
    reply = guard.review(f"Their card is {CARD}; write to {EMAIL}.", Vault(), report)
    assert CARD not in reply  # high impact: redacted by default
    assert "[CREDITCARDNUMBER]" in reply
    assert EMAIL in reply  # lower impact: allowed unless the policy says otherwise
    assert report.leaked == {"CREDITCARDNUMBER": 1, "EMAIL": 1}
    strict = Guard(Policy(outbound={"EMAIL": Action.BLOCK}))
    with pytest.raises(BlockedError):
        strict.review(f"Write to {EMAIL}.", Vault(), Report())


def test_values_the_audit_found_memorized_are_caught_by_fingerprint() -> None:
    key = b"k" * 32
    memorized = {fingerprint(key, "EMAIL", EMAIL)}
    guard = Guard(is_memorized=lambda kind, value: fingerprint(key, kind, value) in memorized)
    report = Report()
    reply = guard.review(f"Contact {EMAIL} or ops@example.com.", Vault(), report)
    assert EMAIL not in reply
    assert "ops@example.com" in reply
    assert report.memorized == {"EMAIL": 1}
    blocking = Guard(Policy(memorized=Action.BLOCK), is_memorized=guard.is_memorized)
    with pytest.raises(BlockedError):
        blocking.review(f"Contact {EMAIL}.", Vault(), Report())


def test_values_the_user_sent_are_not_counted_as_leaks() -> None:
    guard, vault, report = Guard(Policy(inbound_default=Action.ALLOW)), Vault(), Report()
    guard.protect(f"I am {EMAIL}", vault, report)
    guard.review("Noted, [EMAIL_1].", vault, report)
    assert report.leaked == {}


def test_chat_messages_and_tool_calls() -> None:
    guard, vault, report = Guard(), Vault(), Report()
    messages = [
        {"role": "system", "content": "You are a support agent."},
        {"role": "user", "content": [{"type": "text", "text": f'Refund "{EMAIL}"'}]},
    ]
    sent = protect_messages(guard, messages, vault, report)
    assert sent[0] == {"role": "system", "content": PLACEHOLDER_NOTE}
    assert EMAIL not in json.dumps(sent)
    # The caller's copy is untouched.
    assert messages[1]["content"][0]["text"] == f'Refund "{EMAIL}"'
    arguments = json.dumps({"to": "[EMAIL_1]", "note": 'say "hi"'})
    completion = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Refunding [EMAIL_1] now.",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "refund", "arguments": arguments}}
                    ],
                }
            }
        ]
    }
    out = review_completion(guard, completion, vault, report)
    message = out["choices"][0]["message"]
    assert message["content"] == f"Refunding {EMAIL} now."
    assert json.loads(message["tool_calls"][0]["function"]["arguments"])["to"] == EMAIL


def test_restored_values_stay_valid_json_inside_tool_arguments() -> None:
    vault = Vault()
    token = vault.placeholder("FULLNAME", 'Dana "DJ" Lee')
    arguments = json.dumps({"name": token})
    guard = Guard()
    restored = guard.review(arguments, vault, Report(), escape=lambda v: json.dumps(v)[1:-1])
    assert json.loads(restored) == {"name": 'Dana "DJ" Lee'}


def test_nothing_is_masked_without_personal_data() -> None:
    vault, report = Vault(), Report()
    sent = protect_messages(
        Guard(), [{"role": "user", "content": "Summarize this memo."}], vault, report
    )
    assert sent == [{"role": "user", "content": "Summarize this memo."}]  # no extra note


def test_reports_and_policies_hold_no_values() -> None:
    guard, vault, report = Guard(), Vault(), Report()
    guard.protect(f"{EMAIL} {SSN} {CARD}", vault, report)
    guard.review(f"Also {CARD}", vault, report)
    recorded = json.dumps(report.to_json())
    for value in (EMAIL, SSN, CARD):
        assert value not in recorded
    policy = Policy.from_json({"inbound": {"ssn": "block"}, "memorized": "block"})
    assert policy.for_request("SSN") is Action.BLOCK
    assert Policy.from_json(policy.to_json()) == policy
    with pytest.raises(ValueError, match="'nope' is not a valid Action"):
        Policy.from_json({"inbound_default": "nope"})
