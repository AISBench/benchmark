"""Bounded JSON protocol primitives. No stable-version imports."""
import json
from datetime import datetime, timezone

LIMIT = 8192
NON_BOUNDARY = {'phase.changed', 'command.accepted'}


def summary_bytes(value):
    if isinstance(value, dict):
        return sum(len(encode(v)) if k in ('summary', 'message', 'reason', 'error') else summary_bytes(v)
                   for k, v in value.items())
    if isinstance(value, list):
        return sum(summary_bytes(v) for v in value)
    return 0


class ProtocolError(ValueError):
    pass


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def envelope(task, cursor, status='EVENTS'):
    return dict(protocol_version='1.0', task_id=task, status=status,
                time=now(), next_cursor=cursor, has_more=False, events=[])


def page(task, events, after_seq, limit=10, *, project=None, metadata=None):
    if type(after_seq) is not int or not 0 <= after_seq <= len(events):
        raise ProtocolError('CURSOR_INVALID')
    if type(limit) is not int or not 1 <= limit <= 10:
        raise ProtocolError('INVALID_INPUT')
    result = envelope(task, after_seq)
    if metadata:
        result.update(metadata)
    for event in events[after_seq:]:
        if event['type'] in NON_BOUNDARY:
            result['next_cursor'] = event['seq']
            continue
        visible = project(event) if project else event
        candidate = dict(result, events=result['events'] + [visible], next_cursor=event['seq'])
        if len(encode(candidate)) > LIMIT or len(candidate['events']) > limit or summary_bytes(candidate) > 1024:
            if not result['events']:
                raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: event requires externalization')
            break
        result = candidate
    result['has_more'] = result['next_cursor'] < len(events)
    return result
