"""Versioned wire structure plus cross-event identity/cursor invariants."""
from functools import lru_cache
from pathlib import Path
from jsonschema import Draft202012Validator,FormatChecker
from .json_io import load_json
from .protocol import ProtocolError,encode,summary_bytes


@lru_cache(maxsize=2)
def validator(name):
    root=Path(__file__).parent/'schemas'
    schema=load_json(root/(name+'.schema.json'))
    if name=='response':
        schema['$defs']={'event':load_json(root/'event.schema.json')}
    return Draft202012Validator(schema,format_checker=FormatChecker())


def validate_event(event):
    if next(validator('event').iter_errors(event),None) is not None:
        raise ProtocolError('INVALID_INPUT: event schema')
    data=event['data']
    if event['type']=='command.accepted':
        if (data['operation_id']!=event['operation_id'] or data['after_seq']!=event['seq']-1 or
                (data['kind']=='cancel' and 'scope' not in data)):
            raise ProtocolError('INVALID_INPUT: command event identity')
    if event['type']=='task.initialized' and data['task_id']!=event['task_id']:
        raise ProtocolError('INVALID_INPUT: initialized event identity')
    for key in ('evidence_ref','result_ref','resource_results_ref'):
        if key in data and (not isinstance(data[key],str) or any(part in ('','.','..') for part in data[key].split('/'))):
            raise ProtocolError('INVALID_INPUT: event evidence path')


def validate_response(response,*,limit=8192):
    if (next(validator('response').iter_errors(response),None) is not None or
            len(encode(response))>limit or summary_bytes(response)>1024):
        raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: response schema or budget')
    last=0
    for event in response['events']:
        try:
            validate_event(event)
        except ProtocolError as error:
            raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: response event payload') from error
        if event['task_id']!=response['task_id'] or event['seq']<=last or event['seq']>response['next_cursor']:
            raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: response event identity or cursor')
        last=event['seq']
