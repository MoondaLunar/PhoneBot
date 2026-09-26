"""Bounded local diagnostics. Never serialize SDK errors, clients, or environment."""
import os
import re
from .proposals import MAX_PROPOSAL_CHARS


def redact_credentials(raw):
    # Exact environment credential removal also covers nonstandard key formats.
    for name in ("OPENAI_API_KEY", "TWILIO_AUTH_TOKEN", "TWILIO_ACCOUNT_SID"):
        value = os.environ.get(name)
        if value:
            raw = raw.replace(value, "[REDACTED CREDENTIAL]")
    raw = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED API KEY]", raw)
    return raw


def safe_diagnostic(raw):
    if type(raw) is not str:
        return "[Non-text model output withheld]"
    raw = redact_credentials(raw)
    return raw[:MAX_PROPOSAL_CHARS] + ("[TRUNCATED]" if len(raw) > MAX_PROPOSAL_CHARS else "")
