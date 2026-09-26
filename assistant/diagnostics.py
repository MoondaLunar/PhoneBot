"""Allowlisted structural diagnostics; never copy arbitrary provider strings."""

TOKENS = frozenset('''absent other error session.created session.updated response.created
response.done response.output_item.added response.output_item.done
response.content_part.added response.content_part.done response.output_text.delta
response.output_text.done response.output_audio.delta response.function_call_arguments.delta
response.mcp_call.completed conversation.item.added conversation.item.done rate_limits.updated
completed failed incomplete cancelled in_progress message assistant user system
output_text output_audio input_text text audio function_call function_call_output
invalid_request_error invalid_value unknown_parameter insufficient_quota rate_limit_exceeded
server_error invalid_api_key model_not_found billing_not_active usage_limit_reached
quota_exceeded insufficient_credits billing_hard_limit_reached insufficient_quota_error
authentication_error permission_denied invalid_api_key_error
turn_detected client_cancelled max_output_tokens content_filter
ModelUnavailable TimeoutError PermissionError ConnectionError OSError RuntimeError
ValueError TypeError KeyError AttributeError StopIteration StopAsyncIteration
APIConnectionError APITimeoutError AuthenticationError RateLimitError APIStatusError
BadRequestError PermissionDeniedError NotFoundError InternalServerError
InvalidStatus ConnectionClosedError ConnectionClosedOK ValidationError
connect configure generate parse complete context
session_configuration_error unsafe_session_configuration model_request_error
unexpected_modality_or_tool invalid_event invalid_response incomplete_response
invalid_output_count non_message_output invalid_content_count non_text_output
invalid_text_length incomplete_item context_too_large missing_key missing_sdk
ollama_http_error adapter_exception
adapter_exception not_attempted none invalid_size_or_type invalid_json invalid_fields
invalid_string_field unknown_intent speech_not_in_catalog invalid_reason invalid_fact_ids
fact_disclosure_requires_user duplicate_key non_json_number'''.split())


def token(value):
    if value is None:
        return 'absent'
    return value if type(value) is str and value in TOKENS else 'other'


def mapping(value):
    return value if type(value) is dict else {}


def event_metadata(event):
    event = mapping(event)
    response = mapping(event.get('response'))
    details = mapping(response.get('status_details'))
    error = mapping(event.get('error')) or mapping(details.get('error'))
    result = {'event_type': token(event.get('type')),
              'response_status': token(response.get('status')),
              'status_reason': token(details.get('reason')),
              'error_type': token(error.get('type')), 'error_code': token(error.get('code'))}
    output = response.get('output')
    if type(output) is list:
        result['output_count'] = min(len(output), 100)
        result['output_types'] = [token(mapping(item).get('type')) for item in output[:8]]
        result['item_statuses'] = [token(mapping(item).get('status')) for item in output[:8]]
        result['content_types'] = []
        for item in output[:8]:
            content = mapping(item).get('content', [])
            if type(content) is list:
                result['content_types'].extend(token(mapping(part).get('type')) for part in content[:8])
            else:
                result['content_types'].append('other')
    return result


def safe_metadata(value):
    """Recheck even adapter-provided diagnostics at the audit boundary."""
    value = mapping(value)
    result = {}
    for name in ('event_type', 'response_status', 'status_reason', 'error_type', 'error_code',
                 'error_class', 'stage', 'failure_reason', 'parser_failure_reason'):
        if name in value:
            result[name] = token(value[name])
    for name in ('output_types', 'content_types', 'item_statuses'):
        if type(value.get(name)) is list:
            result[name] = [token(item) for item in value[name][:64]]
    if type(value.get('output_count')) is int:
        result['output_count'] = max(0, min(value['output_count'], 100))
    if type(value.get('http_status')) is int and 100 <= value['http_status'] <= 599:
        result['http_status'] = value['http_status']
    return result


def exception_metadata(exc):
    result = {'error_class': token(type(exc).__name__)}
    # Never inspect repr, message, body, headers, request, or response objects.
    if getattr(exc, 'code', None) is not None:
        result.update(safe_metadata({'error_code': exc.code}))
    result.update(safe_metadata({'http_status': getattr(exc, 'status_code', None)}))
    return result
