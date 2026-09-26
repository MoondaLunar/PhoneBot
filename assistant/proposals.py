"""Untrusted JSON in, a small immutable permitted proposal out.

No semantic safety claim is made about arbitrary generated natural language.
Only exact catalog utterances can become simulated speech.
"""
import json
from dataclasses import dataclass
from types import MappingProxyType


CATALOG = MappingProxyType({
    "SPEAK": ("Thank you for that information.", "Thank you. I have noted that answer."),
    "CLARIFY": ("Could you repeat that, please?", "Could you spell that, please?",
                "Could you clarify the answer to my question?",
                "Is there a reference number for that information?"),
    "ASK_USER": ("I need to confirm that with the person I'm assisting. One moment.",),
    "END": ("Thank you for your time. I am ending the call now. Goodbye.",),
})
MAX_PROPOSAL_CHARS = 8192


class ProposalRejected(ValueError):
    pass


@dataclass(frozen=True)
class Proposal:
    intent: str
    proposed_text: str
    requested_fact_ids: tuple[str, ...]
    reason: str  # Untrusted diagnostic text; never instructions or emitted speech.


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProposalRejected("duplicate_key")
        result[key] = value
    return result


def _constant(value):
    raise ProposalRejected("non_json_number")


def validate_proposal(raw, fact_ids):
    if type(raw) is not str or not 0 < len(raw) <= MAX_PROPOSAL_CHARS:
        raise ProposalRejected("invalid_size_or_type")
    try:
        data = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
    except (ValueError, RecursionError):
        raise ProposalRejected("invalid_json") from None
    if type(data) is not dict or set(data) != {"intent", "proposed_text", "requested_fact_ids", "reason"}:
        raise ProposalRejected("invalid_fields")
    if any(type(data[key]) is not str for key in ("intent", "proposed_text", "reason")):
        raise ProposalRejected("invalid_string_field")
    intent = data["intent"]
    if intent not in CATALOG:
        raise ProposalRejected("unknown_intent")
    if data["proposed_text"] not in CATALOG[intent]:
        raise ProposalRejected("speech_not_in_catalog")
    reason = data["reason"]
    if not reason.strip() or len(reason) > 1000:
        raise ProposalRejected("invalid_reason")
    requested = data["requested_fact_ids"]
    if (type(requested) is not list or len(requested) > 1
            or any(type(value) is not str or value not in fact_ids for value in requested)):
        raise ProposalRejected("invalid_fact_ids")
    if requested and intent != "ASK_USER":
        raise ProposalRejected("fact_disclosure_requires_user")
    return Proposal(intent, data["proposed_text"], tuple(requested), reason)
