"""Bounded, strict UTF-8 JSON input. Duplicate keys must not change reviewed intent."""
import json
from .protocol import ProtocolError


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError('INVALID_INPUT: duplicate JSON key')
        result[key] = value
    return result


def reject_constant(value):
    raise ProtocolError('INVALID_INPUT: non-finite JSON number')


def load_json(path, limit=4*1024*1024):
    with open(path, 'rb') as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise ProtocolError('INVALID_INPUT: file too large')
    return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=unique_object,
                      parse_constant=reject_constant)
