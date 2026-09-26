"""Opt-in text-only Realtime proposal adapter. No audio or telephone code."""
import asyncio
import json
import logging
import os
from .diagnostics import event_metadata, exception_metadata, safe_metadata
from .proposals import MAX_PROPOSAL_CHARS
from .adapters import ModelUnavailable, MODEL_INSTRUCTIONS


DEFAULT_MODEL = "gpt-realtime-2.1"
def _client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ModelUnavailable("missing_key")
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ModelUnavailable("missing_sdk") from None
    # Do not inherit alternate endpoints or enable SDK debug/header/frame logging.
    for name in ("openai", "openai._base_client", "openai.resources.realtime.realtime",
                 "openai.lib._websocket", "httpx", "httpx2", "websockets.client"):
        logging.getLogger(name).disabled = True
    return AsyncOpenAI(api_key=key, base_url="https://api.openai.com/v1",
                       websocket_base_url="wss://api.openai.com/v1", max_retries=0,
                       timeout=20.0)


def response_text(event):
    """Never accept partial, tool, audio, or multi-item results as proposals."""
    metadata = event_metadata(event)
    def reject(reason):
        raise ModelUnavailable(reason, {**metadata, "stage": "parse", "parser_failure_reason": reason})
    if type(event) is not dict or event.get("type") != "response.done":
        reject("invalid_event")
    response = event.get("response")
    if type(response) is not dict:
        reject("invalid_response")
    if response.get("status") != "completed":
        reject("incomplete_response")
    output = response.get("output")
    if type(output) is not list or len(output) != 1:
        reject("invalid_output_count")
    item = output[0]
    if type(item) is not dict or item.get("type") != "message" or item.get("role") != "assistant":
        reject("non_message_output")
    if item.get("status") not in (None, "completed"):
        reject("incomplete_item")
    content = item.get("content")
    if type(content) is not list or len(content) != 1:
        reject("invalid_content_count")
    if type(content[0]) is not dict or content[0].get("type") != "output_text":
        reject("non_text_output")
    text = content[0].get("text")
    if type(text) is not str or not 0 < len(text) <= MAX_PROPOSAL_CHARS:
        reject("invalid_text_length")
    return text


class OpenAIRealtimeAdapter:
    def __init__(self, *, allow_api=False, model=DEFAULT_MODEL, timeout=20.0, client_factory=None):
        if allow_api is not True:
            raise ValueError("API access requires explicit --allow-api opt-in.")
        if type(model) is not str or not model.startswith("gpt-realtime"):
            raise ValueError("Choose a Realtime model identifier.")
        if not 0 < timeout <= 60:
            raise ValueError("Timeout must be between zero and 60 seconds.")
        self.model = model
        self.timeout = timeout
        self._client_factory = client_factory or _client
        self.last_diagnostic = {}

    def propose(self, context):
        self.last_diagnostic = {"stage": "context", "parser_failure_reason": "not_attempted"}
        try:
            result = asyncio.run(self._bounded(context))
            self.last_diagnostic.update(stage="complete", parser_failure_reason="none")
            return result
        except Exception as exc:
            diagnostic = {**self.last_diagnostic, **exception_metadata(exc)}
            if type(exc) is ModelUnavailable:
                diagnostic.update(exc.diagnostic)
            else:
                diagnostic["failure_reason"] = "adapter_exception"
            self.last_diagnostic = safe_metadata(diagnostic)
            raise ModelUnavailable(diagnostic["failure_reason"], self.last_diagnostic) from None

    async def _receive(self, connection):
        event = (await connection.recv()).model_dump()
        self.last_diagnostic.update(event_metadata(event))
        if type(event) is not dict or type(event.get("type")) is not str:
            raise ModelUnavailable("invalid_event")
        return event

    async def _bounded(self, context):
        return await asyncio.wait_for(self._request(context), timeout=self.timeout)

    async def _request(self, context):
        payload = json.dumps(context, ensure_ascii=True)
        if len(payload) > 32768:
            raise ModelUnavailable("context_too_large")
        # One isolated request per event avoids retaining rejected proposals in model history.
        self.last_diagnostic["stage"] = "connect"
        async with self._client_factory() as client:
            logger = logging.getLogger("phone_assistant.transport")
            logger.disabled = True
            async with client.realtime.connect(
                model=self.model, max_retries=0,
                websocket_connection_options={"open_timeout": 10, "close_timeout": 1,
                                              "max_size": 65536, "logger": logger},
            ) as connection:
                self.last_diagnostic["stage"] = "configure"
                await connection.session.update(session={
                    "type": "realtime", "output_modalities": ["text"],
                    "instructions": MODEL_INSTRUCTIONS, "tools": [], "tool_choice": "none",
                    "max_output_tokens": 1024, "tracing": None,
                })
                # Confirm our text-only/tool-free configuration before submitting user data.
                while True:
                    event = await self._receive(connection)
                    if event["type"] == "error":
                        raise ModelUnavailable("session_configuration_error")
                    if event["type"] == "session.updated":
                        config = event.get("session", {})
                        if (config.get("output_modalities") != ["text"] or config.get("tools") != []
                                or config.get("tool_choice") != "none"):
                            raise ModelUnavailable("unsafe_session_configuration")
                        break
                self.last_diagnostic["stage"] = "generate"
                await connection.conversation.item.create(item={
                    "type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": payload}],
                })
                await connection.response.create(response={
                    "output_modalities": ["text"], "tools": [], "tool_choice": "none",
                    "max_output_tokens": 1024,
                })
                while True:
                    event = await self._receive(connection)
                    kind = event["type"]
                    if kind == "error":
                        raise ModelUnavailable("model_request_error")
                    if "audio" in kind or "function_call" in kind or "mcp" in kind:
                        raise ModelUnavailable("unexpected_modality_or_tool")
                    if kind == "response.done":
                        return response_text(event)
                    # Deltas are NEVER displayed or emitted, even if they look safe.
