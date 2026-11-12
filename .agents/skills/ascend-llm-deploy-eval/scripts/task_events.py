"""Append-only progress events. Evidence files remain the source of completion truth."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid
import sys

STATES = {'doing', 'done', 'fail', 'skipped', 'blocked', 'retry'}


def emit(root, stage, status, **fields):
    if status not in STATES:
        raise ValueError('Invalid event status')
    event = {**fields, 'event_id': uuid.uuid4().hex,
             'time': datetime.now(timezone.utc).isoformat(), 'stage': stage, 'status': status}
    path = Path(root)/'logs/events.jsonl'
    data = (json.dumps(event, ensure_ascii=False)+'\n').encode('utf-8')
    fd = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        # Concurrent writers on Linux share an advisory lock, including container writers.
        if os.name == 'posix':
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        if os.write(fd, data) != len(data):
            raise OSError('Incomplete event write')
    except OSError as exc:
        # Progress recording must never prevent client/service cleanup.
        try: print('Event recording failed: '+repr(exc), file=sys.stderr, flush=True)
        except OSError: pass
    finally:
        if fd is not None:
            os.close(fd)
    return event


def read_events(root, offset=0, limit=262144):
    path = Path(root)/'logs/events.jsonl'
    if not path.exists():
        return {'offset': offset, 'events': []}
    with path.open('rb') as stream:
        if offset < 0 or offset > path.stat().st_size:
            raise ValueError('Invalid event cursor; do not silently replay')
        stream.seek(offset)
        data = stream.read(limit)
    end = data.rfind(b'\n')+1
    return {'offset': offset+end, 'events': [json.loads(line) for line in data[:end].splitlines()]}
