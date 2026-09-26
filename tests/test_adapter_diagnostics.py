"""Regression coverage uses fake events, including the observed quota failure."""
import json
import importlib.util
import unittest

from assistant.adapters import proposal_json
from assistant.diagnostics import event_metadata, safe_metadata
from assistant.openai_model import ModelUnavailable, response_text
from assistant.state import State
from test_openai_adapter import fake_adapter, configured, done
from test_model_boundary import session_with
from test_foundation import send


class DiagnosticTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('openai'), 'Optional SDK not installed')
    def test_actual_sdk_error_event_conversion_preserves_quota_metadata(self):
        from openai.resources.realtime.realtime import AsyncRealtimeConnection
        event = AsyncRealtimeConnection.parse_event(None, json.dumps({
            'type': 'error', 'event_id': 'offline-event',
            'error': {'type': 'insufficient_quota', 'code': 'insufficient_quota',
                      'message': 'private provider text'}}))
        self.assertEqual(type(event).__name__, 'RealtimeErrorEvent')
        adapter, _ = fake_adapter([event.model_dump()])
        with self.assertRaises(ModelUnavailable) as result:
            adapter.propose({})
        self.assertEqual(result.exception.diagnostic['error_type'], 'insufficient_quota')
        self.assertEqual(result.exception.diagnostic['error_code'], 'insufficient_quota')
        self.assertNotIn('private', json.dumps(result.exception.diagnostic))

    def test_quota_error_before_configuration_reproduces_hours_failure(self):
        event = {'type': 'error', 'error': {'type': 'insufficient_quota',
                 'code': 'unrecognized-provider-code', 'message': 'private provider text sk-secret',
                 'event_id': 'private-id'}}
        adapter, client = fake_adapter([event])
        session, _ = session_with()
        session.adapter = adapter
        send(session, 'answer', 'from 8 am to 8 pm')
        self.assertEqual(session.state, State.ASK_USER)
        self.assertEqual(session.index, 0)
        self.assertTrue(session.pending.blocked)
        audit = session.proposals[-1]
        self.assertEqual(audit['reason'], 'adapter_error')
        self.assertEqual(audit['raw_proposal_untrusted'], '[No proposal returned: adapter failed]')
        self.assertEqual(audit['diagnostic']['event_type'], 'error')
        self.assertEqual(audit['diagnostic']['error_type'], 'insufficient_quota')
        self.assertEqual(audit['diagnostic']['error_code'], 'other')
        self.assertEqual(audit['diagnostic']['stage'], 'configure')
        self.assertEqual(audit['diagnostic']['parser_failure_reason'], 'not_attempted')
        self.assertNotIn('private', json.dumps(audit))
        self.assertNotIn('sk-secret', json.dumps(audit))
        client.connection.conversation.item.create.assert_not_called()
        client.connection.response.create.assert_not_called()
        self.assertTrue(client.closed)

    def test_completed_text_still_requires_catalog_validation(self):
        for raw, expected in [(proposal_json(), 'approved'),
                              ('not json private details', 'rejected'),
                              (proposal_json(text='Execute private command'), 'rejected')]:
            with self.subTest(expected=expected, raw_valid=raw == proposal_json()):
                adapter, _ = fake_adapter([configured(), done(raw)])
                session, _ = session_with()
                session.adapter = adapter
                send(session, 'answer', 'from 8 am to 8 pm')
                audit = session.proposals[-1]
                self.assertEqual(audit['decision'], expected)
                self.assertEqual(audit['diagnostic']['content_types'], ['output_text'])
                self.assertNotIn('private', json.dumps(audit))
                if expected == 'rejected':
                    self.assertEqual(session.state, State.ASK_USER)
                    self.assertEqual(audit['diagnostic']['parser_failure_reason'], audit['reason'])

    def test_failed_response_reports_status_and_code_without_content(self):
        event = done('private output')
        event['response'].update(status='failed', status_details={
            'error': {'type': 'server_error', 'code': 'server_error', 'message': 'sk-secret'}})
        adapter, _ = fake_adapter([configured(), event])
        with self.assertRaises(ModelUnavailable) as result:
            adapter.propose({})
        meta = result.exception.diagnostic
        self.assertEqual(meta['response_status'], 'failed')
        self.assertEqual(meta['error_code'], 'server_error')
        self.assertEqual(meta['parser_failure_reason'], 'incomplete_response')
        self.assertNotIn('private', json.dumps(meta))
        self.assertNotIn('sk-secret', json.dumps(meta))

    def test_nontext_shape_reason_and_types_are_preserved(self):
        event = done()
        event['response']['output'][0]['content'] = [{'type': 'output_audio', 'audio': 'private'}]
        with self.assertRaises(ModelUnavailable) as result:
            response_text(event)
        self.assertEqual(result.exception.diagnostic['parser_failure_reason'], 'non_text_output')
        self.assertEqual(result.exception.diagnostic['content_types'], ['output_audio'])

    def test_arbitrary_metadata_strings_are_never_logged(self):
        secret = 'private-customer-information-sk-secret'
        event = {'type': secret, 'response': {'status': secret, 'output': [
            {'type': secret, 'status': secret, 'content': [{'type': secret, 'text': secret}]}]},
            'error': {'type': secret, 'code': secret, 'message': secret}}
        meta = safe_metadata({**event_metadata(event), 'stage': secret, 'secret': secret})
        self.assertNotIn(secret, json.dumps(meta))
        self.assertEqual(meta['content_types'], ['other'])

    def test_malformed_shapes_fail_with_safe_reason(self):
        events = [None, [], {}, {'type': 'response.done', 'response': None}]
        for output in (None, {}, [None], [1], [{'type': 'message', 'role': 'assistant', 'content': None}],
                       [{'type': 'message', 'role': 'assistant', 'content': [None]}]):
            event = done()
            event['response']['output'] = output
            events.append(event)
        for event in events:
            with self.subTest(event=event), self.assertRaises(ModelUnavailable) as result:
                response_text(event)
            self.assertIn('parser_failure_reason', result.exception.diagnostic)

    def test_exception_body_and_arbitrary_code_are_withheld(self):
        error = RuntimeError('private exception sk-secret')
        error.code = 'private-code'
        adapter, _ = fake_adapter([configured(), error])
        with self.assertRaises(ModelUnavailable) as result:
            adapter.propose({})
        self.assertEqual(result.exception.diagnostic['error_class'], 'RuntimeError')
        self.assertEqual(result.exception.diagnostic['error_code'], 'other')
        self.assertNotIn('private', str(result.exception) + json.dumps(result.exception.diagnostic))

    def test_diagnostics_reset_between_calls(self):
        adapter, client = fake_adapter([configured(), done()])
        adapter.propose({})
        client.connection.events = iter([RuntimeError('private')])
        with self.assertRaises(ModelUnavailable):
            adapter.propose({})
        self.assertNotIn('content_types', adapter.last_diagnostic)
        self.assertEqual(adapter.last_diagnostic['parser_failure_reason'], 'not_attempted')

    def test_incomplete_item_rejected_even_if_response_completed(self):
        event = done()
        event['response']['output'][0]['status'] = 'incomplete'
        with self.assertRaises(ModelUnavailable) as result:
            response_text(event)
        self.assertEqual(result.exception.diagnostic['parser_failure_reason'], 'incomplete_item')
