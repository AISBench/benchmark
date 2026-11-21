"""Single-owner journal core; caller must hold an exclusive runtime process lock.

This module deliberately does not launch runtimes or perform resource operations.
"""
import json
import os
import threading
from pathlib import Path
from .protocol import encode, now, page, envelope, ProtocolError, NON_BOUNDARY, summary_bytes
from .durable import atomic_json
from .json_io import unique_object, reject_constant
from .wire_contracts import validate_event


class Journal:
    def __init__(self, directory, task_id, *, repair_tail=False):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/'.runtime').mkdir(exist_ok=True)
        self.path = self.root / 'task.jsonl'
        self.task_id = task_id
        self.condition = threading.Condition()
        self.events = []
        self.projections = {}
        repaired_tail = None
        if self.path.exists():
            from .task_log import locked_stream
            with locked_stream(self.path) as stream:
                stream.seek(0)
                while True:
                    line = stream.readline(16386)
                    if not line:
                        break
                    if not line.endswith(b'\n') and len(line) <= 16384 and stream.read(1) == b'' and repair_tail:
                        # Caller must already hold the exclusive runtime lock.
                        offset = stream.tell()-len(line)
                        stream.truncate(offset)
                        stream.flush()
                        os.fsync(stream.fileno())
                        repaired_tail = line.hex()
                        break
                    try:
                        event = json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
                        if not isinstance(event, dict) or not line.endswith(b'\n') or len(line) > 16385:
                            raise ValueError('Invalid journal record')
                        if event.get('record_type') == 'diagnostic':
                            if any(not isinstance(event.get(k), str) for k in ('time', 'source', 'message')) or len(event['message']) > 2048:
                                raise ValueError('Invalid diagnostic')
                            continue
                        validate_event(event)
                        if (not line.endswith(b'\n') or len(line) > 16385 or
                                event['task_id'] != task_id or event['protocol_version'] != '1.0' or
                                type(event['seq']) is not int or event['seq'] != len(self.events)+1 or
                                type(event['task_version']) is not int or event['task_version'] < 0 or
                                (self.events and event['task_version']<self.events[-1]['task_version']) or
                                not isinstance(event['data'], dict) or not isinstance(event['type'], str)):
                            raise ValueError('Invalid event')
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ProtocolError('RECOVERY_REQUIRED') from exc
                    self.events.append(event)

        if repaired_tail is not None:
            from .task_log import diagnostic
            diagnostic(self.path, repaired_tail, event='journal.partial_tail', encoding='hex')

    def visible(self, event):
        if len(encode(event)) <= 6000 and summary_bytes(event) <= 256:
            return event
        if event['seq'] in self.projections:
            return self.projections[event['seq']]
        reference = 'results/events/'+str(event['seq'])+'.json'
        atomic_json(self.root/reference, event)
        keys = {'service_id', 'case_id', 'endpoint', 'model', 'evidence_ref', 'result_ref',
                'execution_status', 'evidence_status', 'performance_validity', 'cleanup_status',
                'error_code', 'resource_results_ref', 'request_id', 'request_hash', 'kind',
                'operation_id', 'after_seq', 'scope', 'bundle_sha256', 'ownership', 'task_id','phase','resource_refs'}
        data = {k: v for k, v in event['data'].items() if k in keys}
        if 'reason' in event['data']:
            data['reason'] = str(event['data']['reason']).encode('utf-8')[:128].decode('utf-8', errors='ignore')
        data.update(artifact_ref=reference, details_externalized=True)
        projected = dict(event, data=data)
        if len(encode(projected)) > 6000 or summary_bytes(projected) > 256:
            raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: essential event fields exceed budget')
        self.projections[event['seq']] = projected
        return projected

    def append(self, kind, operation_id, version, status, data, *, before_notify=None):
        with self.condition:
            if self.events and version<self.events[-1]['task_version']:
                raise ProtocolError('INVALID_INPUT: event task version regression')
            event = dict(protocol_version='1.0', task_id=self.task_id,
                         seq=len(self.events)+1, task_version=version,
                         operation_id=operation_id, type=kind, time=now(),
                         status=status, data=data)
            validate_event(event)
            if len(encode(event)) > 16384:
                raise ProtocolError('INVALID_INPUT: internal event exceeds 16 KiB; use artifact reference')
            from .task_log import append
            append(self.path,event)
            self.events.append(event)
            if before_notify is not None:
                before_notify(event)
            self.condition.notify_all()
            return event

    def watch(self, after_seq, wait_ms=30000, limit=10, *, metadata=None):
        if type(wait_ms) is not int or not 0 <= wait_ms <= 30000:
            raise ProtocolError('INVALID_INPUT')
        with self.condition:
            page(self.task_id, self.events, after_seq, limit, project=self.visible, metadata=metadata)
            self.condition.wait_for(lambda: any(e['type'] not in NON_BOUNDARY for e in self.events[after_seq:]), wait_ms/1000)
            response = page(self.task_id, self.events, after_seq, limit, project=self.visible, metadata=metadata)
            if not response['events']:
                response['status'] = 'WAIT_TIMEOUT'
            return response
