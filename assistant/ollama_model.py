"""Local text-only Ollama proposal adapter using its HTTP API."""
import json
import os
from urllib.parse import urlsplit
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from .adapters import ModelUnavailable, MODEL_INSTRUCTIONS
from .diagnostics import safe_metadata
from .proposals import MAX_PROPOSAL_CHARS


DEFAULT_MODEL = "qwen3:8b"
DEFAULT_HOST = "http://localhost:11434"
MAX_RESPONSE_BYTES = 1_048_576


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def _local_opener():
    return build_opener(ProxyHandler({}), _NoRedirect).open


class OllamaModelAdapter:
    def __init__(self, *, model=None, host=None, timeout=20.0, opener=None):
        model = model if model is not None else os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
        host = host if host is not None else os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
        if type(model) is not str or not model.strip() or len(model) > 128:
            raise ValueError("Choose a valid Ollama model identifier.")
        parsed_host = urlsplit(host) if type(host) is str else None
        if (parsed_host is None or parsed_host.scheme not in {"http", "https"}
            or not parsed_host.netloc or parsed_host.username or parsed_host.password
            or parsed_host.query or parsed_host.fragment or parsed_host.path not in {"", "/"}):
            raise ValueError("Ollama host must be an HTTP or HTTPS URL.")
        if not 0 < timeout <= 60:
            raise ValueError("Timeout must be between zero and 60 seconds.")
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._opener = opener or _local_opener()
        self.last_diagnostic = {}

    def propose(self, context):
        self.last_diagnostic = {"stage": "context", "parser_failure_reason": "not_attempted"}
        try:
            payload = json.dumps(context, ensure_ascii=True)
            if len(payload) > 32768:
                raise ModelUnavailable("context_too_large")
            body = json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": MODEL_INSTRUCTIONS},
                    {"role": "user", "content": payload},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            }).encode("utf-8")
            request = Request(self.host + "/api/chat", data=body,
                              headers={"Content-Type": "application/json"}, method="POST")
            self.last_diagnostic["stage"] = "generate"
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ModelUnavailable("invalid_response")
            result = json.loads(raw)
            message = result.get("message") if type(result) is dict else None
            if (type(result) is not dict or result.get("done") is not True
                    or type(message) is not dict or message.get("role") != "assistant"):
                raise ModelUnavailable("invalid_response")
            text = message.get("content")
            if type(text) is not str or not 0 < len(text) <= MAX_PROPOSAL_CHARS:
                raise ModelUnavailable("invalid_text_length")
            self.last_diagnostic = {"stage": "complete", "parser_failure_reason": "none"}
            return text
        except ModelUnavailable as exc:
            self.last_diagnostic = safe_metadata({**self.last_diagnostic, **exc.diagnostic})
            raise ModelUnavailable(exc.diagnostic.get("failure_reason", "adapter_exception"),
                                   self.last_diagnostic) from None
        except Exception as exc:
            reason = "ollama_http_error" if isinstance(exc, HTTPError) else "adapter_exception"
            self.last_diagnostic = safe_metadata({**self.last_diagnostic,
                                                  "error_class": type(exc).__name__,
                                                  "parser_failure_reason": "not_attempted",
                                                  "failure_reason": reason})
            raise ModelUnavailable(reason, self.last_diagnostic) from None