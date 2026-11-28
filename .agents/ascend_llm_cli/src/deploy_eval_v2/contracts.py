"""Versioned input validation before command admission."""
import hashlib
import json
import re
from pathlib import Path
from jsonschema import Draft202012Validator
from .protocol import encode, ProtocolError


def validate_command(command):
    schema = json.loads((Path(__file__).parent/'schemas/command.schema.json').read_text(encoding='utf-8'))
    if list(Draft202012Validator(schema).iter_errors(command)):
        raise ProtocolError('INVALID_INPUT')
    allowed = {
        'deploy': {'plan_ref'}, 'eval': {'service_id', 'case_id'},
        'release': {'service_id'}, 'cancel': {'scope', 'operation_id'}}
    common = {'protocol_version', 'task_id', 'kind', 'request_id', 'expected_version', 'wait_ms'}
    if set(command) - common - allowed[command['kind']]:
        raise ProtocolError('INVALID_INPUT')


def fingerprint(command):
    # Observation budget is not part of business-command identity.
    value = {k: v for k, v in command.items() if k != 'wait_ms'}
    return hashlib.sha256(encode(dict(sorted(value.items())))).hexdigest()


def validate_query(request):
    if not isinstance(request, dict) or request.get('protocol_version') != '1.0':
        raise ProtocolError('INVALID_INPUT')
    if not isinstance(request.get('task_id'), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', request['task_id']):
        raise ProtocolError('INVALID_INPUT')
    fields = {'get-status': set(), 'wait-event': {'after_seq', 'wait_ms', 'limit'},
              'read-result': {'artifact_ref', 'offset', 'max_bytes','tail_bytes','line_start','line_end','keyword','scan_offset','scan_line','expected_sha256'}}
    kind = request.get('kind')
    if kind not in fields or set(request) - fields[kind] - {'kind', 'task_id', 'protocol_version'}:
        raise ProtocolError('INVALID_INPUT')
    bounds = {'after_seq': (0, 2**63-1), 'wait_ms': (0, 30000), 'limit': (1, 10),
              'offset': (0, 2**63-1), 'max_bytes': (1024, 32768),'tail_bytes':(1,32768),
              'line_start':(1,2**63-1),'line_end':(1,2**63-1),'scan_offset':(0,2**63-1),'scan_line':(1,2**63-1)}
    for key in fields[kind] & bounds.keys():
        if key in request and (type(request[key]) is not int or not bounds[key][0] <= request[key] <= bounds[key][1]):
            raise ProtocolError('INVALID_INPUT')
    if kind == 'wait-event' and 'after_seq' not in request:
        raise ProtocolError('INVALID_INPUT')
    if kind == 'read-result':
        if 'expected_sha256' in request and (not isinstance(request['expected_sha256'],str) or
                not re.fullmatch('[a-f0-9]{64}',request['expected_sha256'])):
            raise ProtocolError('INVALID_INPUT: evidence digest')
        modes=sum(key in request for key in ('tail_bytes','line_start','keyword'))
        if (modes>1 or (modes and request.get('offset',0)) or
                ('line_start' in request)!=('line_end' in request) or
                ('line_start' in request and request['line_start']>request['line_end'])):
            raise ProtocolError('INVALID_INPUT: artifact locator conflict')
        if 'keyword' in request and (not isinstance(request['keyword'],str) or not 1<=len(request['keyword'].encode('utf-8'))<=256):
            raise ProtocolError('INVALID_INPUT: artifact keyword')
        ref = request.get('artifact_ref')
        if not isinstance(ref, str) or not 1 <= len(ref) <= 512 or '\\' in ref or '\x00' in ref:
            raise ProtocolError('INVALID_INPUT')
        parts = ref.split('/')
        if ref not in ('task.jsonl','service.json') and (parts[0] not in ('logs', 'results') or len(parts) < 2 or any(p in ('', '.', '..') for p in parts)):
            raise ProtocolError('INVALID_INPUT')


TRANSPORT_ERRORS = {'UNKNOWN_OUTCOME', 'SSH_UNAVAILABLE', 'TRANSPORT_PROTOCOL_ERROR'}


def exit_code(response):
    if response['status'] == 'ERROR':
        return 3 if response['error']['code'] in TRANSPORT_ERRORS else 2
    if any(e.get('type') in ('operation.failed', 'release.unverified') for e in response.get('events', [])):
        return 4
    if any(e.get('type')=='evaluation.completed' and e.get('data',{}).get('execution_status')=='FAILED'
           for e in response.get('events',[])):
        return 4
    return 0
