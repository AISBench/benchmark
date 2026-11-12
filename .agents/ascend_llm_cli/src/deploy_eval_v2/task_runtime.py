"""Journal-driven command admission. Executor injection is explicit, never inferred."""
import threading
import uuid
import traceback
from .task_log import DiagnosticStream, task_root, record_exception
from datetime import datetime,timezone
from .contracts import validate_command, fingerprint
from .event_journal import Journal
from .protocol import ProtocolError
from .durable import atomic_json


class Runtime:
    def __init__(self, directory, task_id, executor, *, repair_tail=False):
        self.journal = Journal(directory, task_id, repair_tail=repair_tail)
        self.executor = executor
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.version = 0
        self.active = None
        self.commands = {}
        self.operations = {}
        self.state = 'NEW'
        self.service_id = None
        self.task_cancelled = False
        self.initial_identity = None
        self.idle_timer = None
        for event in self.journal.events:
            self._apply(event)
        identity = getattr(executor, 'initial_identity', None)
        if identity is not None:
            if identity['task_id'] != task_id:
                raise ProtocolError('INVALID_INPUT: frozen task identity')
            if self.journal.events:
                if self.initial_identity != identity:
                    raise ProtocolError('RECOVERY_REQUIRED: frozen identity changed')
            else:
                self._emit('task.initialized', None, 'EXTERNAL_READY', identity)
        if self.active:
            # A durable acceptance is NOT permission to replay an uncertain effect.
            self._emit('runtime.recovery_required',self.active,'RECOVERY_REQUIRED',dict(
                reason='Runtime restarted with an accepted operation lacking terminal evidence',
                resource_refs=['task.jsonl','service.json']))
        self._snapshot()
        self._schedule_idle()

    def _schedule_idle(self):
        if self.idle_timer is not None:
            self.idle_timer.cancel()
            self.idle_timer = None
        policy = getattr(self.executor,'idle_policy',None)
        if (not policy or policy['on_idle']!='release_owned' or self.active or
                self.state not in ('READY','EVALUATED') or self.task_cancelled):
            return
        # Use the persisted boundary time, so runtime restart does not renew a lease.
        boundary = next(e for e in reversed(self.journal.events)
                        if e['type'] in ('deployment.ready','evaluation.completed'))
        stamp = datetime.fromisoformat(boundary['time'].replace('Z','+00:00'))
        elapsed = (datetime.now(timezone.utc)-stamp).total_seconds()
        delay = max(0,policy['idle_timeout_seconds']-elapsed)
        self.idle_timer = threading.Timer(delay,self._expire,args=(self.version,))
        self.idle_timer.daemon = True
        self.idle_timer.start()

    def _expire(self,version):
        with self.lock:
            if self.version!=version or self.active or self.state not in ('READY','EVALUATED') or self.task_cancelled:
                return
            command = dict(protocol_version='1.0',task_id=self.journal.task_id,kind='release',
                           service_id=self.service_id,request_id='idle-'+str(version),expected_version=version)
            validator = getattr(self.executor,'validate',None)
            try:
                if validator:
                    validator(command)
            except Exception:
                self._emit('runtime.recovery_required',None,'RECOVERY_REQUIRED',
                           dict(reason='Idle cleanup inputs could not be verified',resource_refs=['service.json']))
                return
            operation = uuid.uuid4().hex
            self._emit('command.accepted',operation,'RUNNING',dict(request_id=command['request_id'],
                request_hash=fingerprint(command),operation_id=operation,kind='release',after_seq=len(self.journal.events)))
            self.cancelled.clear()
        self._execute(command,operation,expired=True)

    def _snapshot(self):
        root = self.journal.root
        atomic_json(root/'results'/'operation-index.json',dict(schema_version='1.0',task_id=self.journal.task_id,
                    operations=self.operations))
        atomic_json(root/'state.json', dict(protocol_version='1.0', task_id=self.journal.task_id,
                    task_version=self.version, state=self.state, operation_id=self.active,
                    service_id=self.service_id, task_cancelled=self.task_cancelled,
                    next_cursor=len(self.journal.events)))

    def _apply_and_snapshot(self, event):
        self._apply(event)
        self._snapshot()

    def _apply(self, event):
        self.version = event['task_version']
        data = event['data']
        if event['type'] == 'task.initialized':
            self.initial_identity = dict(data)
            self.service_id = data['service_id']
            self.state = 'EXTERNAL_READY'
        elif event['type'] == 'command.accepted':
            self.commands[data['request_id']] = data
            if data['kind'] != 'cancel':
                self.active = event['operation_id']
                self.operations[self.active]=dict(operation_id=self.active,kind=data['kind'],
                    request_id=data['request_id'],status='RUNNING',accepted_seq=event['seq'],time=event['time'])
            elif data['scope'] == 'task':
                self.task_cancelled = True
                if data.get('idle_cancel'):
                    self.active = event['operation_id']
                    self.operations[self.active]=dict(operation_id=self.active,kind='cancel',request_id=data['request_id'],
                        status='RUNNING',accepted_seq=event['seq'],time=event['time'])
        elif event['type'] in ('deployment.ready','evaluation.completed','operation.failed','release.completed',
                               'release.unverified','operation.cancelled','task.expired','runtime.recovery_required'):
            operation=event['operation_id']
            if operation is not None:
                previous=self.operations.get(operation,{})
                self.operations[operation]=dict(previous,operation_id=operation,status=event['status'],
                    terminal_seq=event['seq'],event_type=event['type'],time=event['time'],result=dict(data))
            self.active = None
            self.state = event['status']
            self.service_id = data.get('service_id', self.service_id)

    def _emit(self, kind, operation, status, data):
        data=dict(data)
        if kind in ('operation.failed','operation.cancelled'):
            if kind=='operation.failed':
                data.setdefault('phase',self.operations.get(operation,{}).get('kind','unknown'))
            if 'evidence_ref' not in data:
                reference='results/terminal/'+str(operation)+'.json'
                atomic_json(self.journal.root/reference,dict(schema_version='1.0',task_id=self.journal.task_id,
                    operation_id=operation,event_type=kind,status=status,data=data))
                data['evidence_ref']=reference
        if kind=='release.unverified':
            data.setdefault('reason','Resource absence or ownership could not be verified')
        # Waiters cannot observe the event before its in-memory state is applied.
        with self.journal.condition:
            try:
                return self.journal.append(kind, operation, self.version+1, status, data,
                                           before_notify=self._apply_and_snapshot)
            except Exception:
                self.state = 'RECOVERY_REQUIRED'
                raise

    def submit(self, command):
        validate_command(command)
        if command['task_id'] != self.journal.task_id:
            raise ProtocolError('INVALID_INPUT')
        digest = fingerprint(command)
        with self.lock:
            old = self.commands.get(command['request_id'])
            if old:
                if old['request_hash'] != digest:
                    raise ProtocolError('REQUEST_CONFLICT')
                return dict(old)
            if self.state == 'RECOVERY_REQUIRED':
                raise ProtocolError('RECOVERY_REQUIRED')
            if self.version != command['expected_version']:
                raise ProtocolError('STALE_VERSION')
            kind = command['kind']
            if self.initial_identity and kind in ('deploy', 'release'):
                raise ProtocolError('INVALID_INPUT: external service not owned')
            if self.task_cancelled:
                raise ProtocolError('INVALID_INPUT: task cancelled')
            if kind == 'cancel':
                if (command['scope']=='operation' and self.active != command['operation_id']) or (
                        'operation_id' in command and self.active != command['operation_id']):
                    raise ProtocolError('INVALID_INPUT: operation not active')
            elif self.active:
                raise ProtocolError('RESOURCE_BUSY')
            elif kind == 'deploy' and self.state != 'NEW':
                raise ProtocolError('INVALID_INPUT: deployment already attempted')
            elif kind in ('eval', 'release') and (self.state not in ('READY', 'EXTERNAL_READY', 'EVALUATED') or command['service_id'] != self.service_id):
                raise ProtocolError('INVALID_INPUT: service unavailable')
            validator = getattr(self.executor, 'validate', None)
            if validator:
                validator(command)
            idle_cancel = kind=='cancel' and self.active is None
            operation = (self.active or uuid.uuid4().hex) if kind == 'cancel' else uuid.uuid4().hex
            data = dict(request_id=command['request_id'], request_hash=digest,
                        operation_id=operation, kind=kind, after_seq=len(self.journal.events))
            if kind == 'cancel':
                data['scope'] = command['scope']
                if idle_cancel:
                    data['idle_cancel'] = True
            self._emit('command.accepted', operation, 'RUNNING', data)
            if self.idle_timer is not None:
                self.idle_timer.cancel()
                self.idle_timer = None
            if kind == 'cancel':
                self.cancelled.set()
                if idle_cancel:
                    threading.Thread(target=self._cancel_idle,args=(operation,),daemon=True).start()
            else:
                self.cancelled.clear()
                threading.Thread(target=self._execute, args=(dict(command), operation), daemon=True).start()
            return dict(data)

    def _cancel_idle(self,operation):
        handler=getattr(self.executor,'cancel_idle',None)
        try:
            if handler:
                kind,status,data=handler(operation)
            else:
                verified=self.service_id is None or bool(self.initial_identity and self.initial_identity.get('ownership')=='external')
                kind,status,data='operation.cancelled','CANCELLED',dict(cleanup_status='VERIFIED' if verified else 'UNVERIFIED')
        except Exception:
            kind,status,data='operation.cancelled','RECOVERY_REQUIRED',dict(cleanup_status='UNVERIFIED')
        with self.lock:
            self._emit(kind,operation,status,data)

    def _execute(self, command, operation,expired=False):
        try:
            kind, status, data = self.executor(command, operation, self.cancelled)
        except Exception:
            # Preserve diagnostics even when the stage adapter itself fails.
            reference='task.jsonl'
            path=self.journal.root/reference
            path.parent.mkdir(parents=True,exist_ok=True)
            with DiagnosticStream(path) as stream:
                traceback.print_exc(file=stream)
            kind, status, data = 'operation.failed', 'RECOVERY_REQUIRED', {
                'error_code': 'EXECUTOR_FAILED', 'cleanup_status': 'UNVERIFIED','evidence_ref':reference}
        with self.lock:
            late_cancel=self.cancelled.is_set() and kind in ('deployment.ready','evaluation.completed')
            if not late_cancel:
                self._finish_execution(kind,status,data,operation,expired)
                return
        # Admission is still blocked by self.active. Never hold the runtime lock
        # during cleanup: other clients must be able to observe/wait/cancel.
        handler=getattr(self.executor,'cancel_idle',None)
        try:
            if handler:
                kind,status,data=handler(operation+'-cancel')
            else:
                external=bool(self.initial_identity and self.initial_identity.get('ownership')=='external')
                kind,status,data='operation.cancelled','CANCELLED' if external else 'RECOVERY_REQUIRED',{
                    'cleanup_status':'VERIFIED' if external else 'UNVERIFIED'}
        except Exception:
            kind,status,data='operation.cancelled','RECOVERY_REQUIRED',{'cleanup_status':'UNVERIFIED'}
        with self.lock:
            self._finish_execution(kind,status,data,operation,False)

    def _finish_execution(self,kind,status,data,operation,expired):
        """Caller holds runtime lock: publish exactly one terminal boundary."""
        if kind in ('operation.failed','operation.cancelled') and data.get('cleanup_status')=='UNVERIFIED':
            status='RECOVERY_REQUIRED'
        if expired and kind=='release.completed':
            kind='task.expired'
            data.update(reason='idle_timeout',cleanup_status='VERIFIED')
        self._emit(kind, operation, status, data)
        self._schedule_idle()
