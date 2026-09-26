"""Ollama tests use a fake local HTTP boundary; none requires an Ollama daemon."""
import json
import unittest
from unittest.mock import patch
from assistant.adapters import proposal_json
from assistant.ollama_model import DEFAULT_HOST, DEFAULT_MODEL, OllamaModelAdapter
from assistant.policy import Objective
from assistant.proposals import CATALOG
from assistant.session import Session
from assistant.state import State


class FakeResponse:
    def __init__(self, value):
        self.value = value
        self.read_limit = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        self.read_limit = limit
        return self.value


def ollama_response(content, *, done=True, role="assistant"):
    return json.dumps({"done": done, "message": {"role": role, "content": content}}).encode()


def objective():
    return Objective("+12025550123", "Alex", "Office information",
                     ("Hours?", "Documents?"), ("My name is Alex.", "I prefer mornings."))


class OllamaAdapterTests(unittest.TestCase):
    def test_local_http_request_uses_configured_model_and_proposal_instructions(self):
        response = FakeResponse(ollama_response(proposal_json()))
        calls = []

        def opener(request, *, timeout):
            calls.append((request, timeout))
            return response

        adapter = OllamaModelAdapter(opener=opener)
        raw = adapter.propose({"remote_input_untrusted": "Hours?"})
        self.assertEqual(adapter.model, DEFAULT_MODEL)
        self.assertEqual(adapter.host, DEFAULT_HOST)
        self.assertEqual(raw, proposal_json())
        request, timeout = calls[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, DEFAULT_HOST + "/api/chat")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(body["model"], "qwen3:8b")
        self.assertFalse(body["stream"])
        self.assertEqual(body["format"], "json")
        self.assertIn("CATALOG:", body["messages"][0]["content"])
        self.assertEqual(json.loads(body["messages"][1]["content"]),
                         {"remote_input_untrusted": "Hours?"})
        self.assertEqual(timeout, 20.0)
        self.assertEqual(response.read_limit, 1_048_577)

    def make_session(self, content, opener):
        adapter = OllamaModelAdapter(opener=opener)
        session = Session(objective(), adapter=adapter)
        session.start(confirmed=True)
        session.remote("answer", "Weekdays, nine to five.", message_id="m1",
                       question_id=session.question_id)
        return session, adapter

    def test_valid_proposal_passes_existing_validator_and_catalog_speech(self):
        session, _ = self.make_session(
            proposal_json(), lambda *_args, **_kwargs: FakeResponse(ollama_response(proposal_json())))
        self.assertEqual(session.proposals[0]["decision"], "approved")
        self.assertEqual(session.index, 1)
        approved_speech = [row["text"] for row in session._spoken if row["action"] == "proposal"]
        self.assertEqual(len(approved_speech), 1)
        self.assertIn(approved_speech[0], CATALOG["SPEAK"])

    def test_malformed_provider_or_proposal_response_fails_closed(self):
        for raw in (b"not json", ollama_response("not a proposal")):
            with self.subTest(raw=raw):
                session, _ = self.make_session("", lambda *_args, value=raw, **_kwargs: FakeResponse(value))
                self.assertEqual(session.state, State.ASK_USER)
                self.assertTrue(session.pending.blocked)
                self.assertEqual(session.index, 0)
                self.assertNotIn("not a proposal", [row["text"] for row in session.transcript])

    def test_out_of_catalog_model_speech_never_escapes(self):
        generated = "I agree to the contract."
        session, _ = self.make_session(
            generated,
            lambda *_args, **_kwargs: FakeResponse(ollama_response(proposal_json(text=generated))))
        self.assertEqual(session.state, State.ASK_USER)
        self.assertTrue(session.pending.blocked)
        self.assertNotIn(generated, [row["text"] for row in session.transcript])
        self.assertFalse(any(row["action"] == "proposal" for row in session._spoken))

    def test_connection_failure_fails_closed_without_exposing_exception(self):
        for error in (ConnectionError("private response body"), TimeoutError("private timeout body")):
            with self.subTest(error=type(error).__name__):
                def fail(*_args, **_kwargs):
                    raise error

                session, _ = self.make_session("", fail)
                self.assertEqual(session.state, State.ASK_USER)
                self.assertTrue(session.pending.blocked)
                self.assertNotIn(str(error), json.dumps(session.proposals))

    def test_stale_result_is_discarded_after_question_changes(self):
        holder = {}

        def change_question(request, *, timeout):
            holder["session"]._question_id = "changed-while-inference-ran"
            return FakeResponse(ollama_response(proposal_json()))

        adapter = OllamaModelAdapter(opener=change_question)
        session = Session(objective(), adapter=adapter)
        holder["session"] = session
        session.start(confirmed=True)
        session.remote("answer", "Weekdays, nine to five.", message_id="m1",
                       question_id=session.question_id)
        self.assertEqual(session.proposals[0]["decision"], "rejected")
        self.assertEqual(session.proposals[0]["reason"], "stale_model_result")
        self.assertFalse(any(row["action"] == "proposal" for row in session._spoken))
        self.assertEqual(session.index, 0)

    def test_authorization_remains_local_and_fact_specific(self):
        proposal = proposal_json("ASK_USER", fact_ids=["F1"])
        session, _ = self.make_session(
            proposal, lambda *_args, **_kwargs: FakeResponse(ollama_response(proposal)))
        self.assertEqual(session.state, State.ASK_USER)
        self.assertFalse(any(row["action"] == "fact" for row in session._spoken))
        with self.assertRaises(ValueError):
            session.respond("fact 1", pending_id="not-the-pending-id")
        session.respond("fact 1", pending_id=session.pending.id)
        self.assertTrue(any(row["action"] == "fact" for row in session._spoken))

    def test_configuration_is_selectable(self):
        with patch.dict("os.environ", {"OLLAMA_MODEL": "qwen3:8b-test",
                                        "OLLAMA_HOST": "http://127.0.0.1:11435"}):
            adapter = OllamaModelAdapter(opener=lambda *_args, **_kwargs: None)
        self.assertEqual(adapter.model, "qwen3:8b-test")
        self.assertEqual(adapter.host, "http://127.0.0.1:11435")


if __name__ == "__main__":
    unittest.main()
