"""Stage tools share submission and bounded observation semantics."""
import time
from .protocol import envelope, encode, ProtocolError


class API:
    def __init__(self, runtime):
        self.runtime = runtime

    def submit(self, command):
        accepted = self.runtime.submit(command)
        cursor = accepted['after_seq']
        deadline = time.monotonic()+command.get('wait_ms', 30000)/1000
        # Return acceptance together with a boundary if it arrives within budget.
        journal = self.runtime.journal
        operation = accepted['operation_id']
        with journal.condition:
            journal.condition.wait_for(
                lambda: any(e['operation_id'] == operation and e['type'] not in
                            ('command.accepted', 'phase.changed') for e in journal.events[cursor:]),
                max(0, deadline-time.monotonic()))
            response = journal.watch(cursor, 0, metadata=dict(operation_id=operation, task_version=self.runtime.version))
            if not any(e['operation_id'] == operation and e['type'] not in
                       ('command.accepted', 'phase.changed') for e in response['events']):
                response['status'] = 'RUNNING'
        if len(encode(response)) > 8192:
            raise ProtocolError('TRANSPORT_PROTOCOL_ERROR')
        return response

    def status(self):
        with self.runtime.lock:
            result = envelope(self.runtime.journal.task_id, len(self.runtime.journal.events))
            result.update(task_version=self.runtime.version, operation_id=self.runtime.active,
                          task_state=self.runtime.state, service_id=self.runtime.service_id,
                          result_index_ref='results/operation-index.json')
            return result
