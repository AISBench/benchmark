"""Host-global persistent claims and journaled command identities.

All v2 tasks on a physical host must use the same guard root. Claims survive
process crashes; absence of a live process is never sufficient to remove them.
"""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re
from .compiler import digest
from .durable import atomic_json
from .json_io import load_json
from .protocol import ProtocolError


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if os.name == 'posix':
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ProtocolError('RESOURCE_BUSY') from exc
        else:
            import msvcrt
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ProtocolError('RESOURCE_BUSY') from exc
        try:
            yield
        finally:
            if os.name == 'posix':
                fcntl.flock(stream, fcntl.LOCK_UN)
            else:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


class HostGuard:
    def __init__(self, root, task_id, token):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', task_id) or not re.fullmatch('[a-f0-9]{64}', token):
            raise ProtocolError('INVALID_INPUT: guard identity')
        self.root = Path(root)
        self.owner = dict(task_id=task_id, token=token)
        self.root.mkdir(parents=True, exist_ok=True)

    def claim(self, resources):
        if not resources or any(not isinstance(r, str) or not r or len(r)>256 for r in resources):
            raise ProtocolError('INVALID_INPUT: resources')
        with exclusive(self.root/'host.lock'):
            paths = [(resource, self.root/'claims'/(hashlib.sha256(resource.encode()).hexdigest()+'.json'))
                     for resource in sorted(set(resources))]
            for resource, path in paths:
                if path.exists() and load_json(path) != dict(self.owner, resource=resource):
                    raise ProtocolError('RESOURCE_BUSY: resource claimed by another task')
            for resource, path in paths:
                atomic_json(path, dict(self.owner, resource=resource))

    def release_claims(self, resources, *, resources_verified_absent):
        if resources_verified_absent is not True:
            raise ProtocolError('RECOVERY_REQUIRED: resource absence not verified')
        with exclusive(self.root/'host.lock'):
            paths = []
            for resource in resources:
                path = self.root/'claims'/(hashlib.sha256(resource.encode()).hexdigest()+'.json')
                if path.exists():
                    if load_json(path) != dict(self.owner, resource=resource):
                        raise ProtocolError('RECOVERY_REQUIRED: foreign resource claim')
                    paths.append(path)
            # Validate the whole set before removing any claims.
            for path in paths:
                path.unlink()

    def command(self, command_id, payload, action):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', command_id):
            raise ProtocolError('INVALID_INPUT: remote command identity')
        key = digest(dict(owner=self.owner, command_id=command_id))
        path = self.root/'commands'/(key+'.json')
        signature = digest(payload)
        with exclusive(self.root/'command-locks'/(key+'.lock')):
            if path.exists():
                record = load_json(path)
                if record['payload_sha256'] != signature:
                    raise ProtocolError('REQUEST_CONFLICT')
                if record['status'] != 'COMPLETED':
                    raise ProtocolError('RECOVERY_REQUIRED: remote command outcome unknown')
                return record['result']
            record = dict(owner=self.owner, command_id=command_id, payload_sha256=signature,
                          effect_kind=payload.get('kind','unknown') if isinstance(payload,dict) else 'unknown',status='ACCEPTED')
            atomic_json(path, record)
            result = action()
            atomic_json(path, dict(record, status='COMPLETED', result=result))
            return result

    def release_unstarted(self,resources):
        """Only unlock a pre-create failure if durable history proves no create.

        Absence of an in-memory container ID is not enough. Old records without
        effect classification are conservatively treated as unknown effects.
        """
        with exclusive(self.root/'host.lock'):
            for path in (self.root/'commands').glob('*.json'):
                record=load_json(path)
                if record.get('owner')==self.owner and record.get('effect_kind','unknown') in ('create','unknown'):
                    raise ProtocolError('RECOVERY_REQUIRED: prior resource effect requires inventory verification')
            paths=[]
            for resource in resources:
                path=self.root/'claims'/(hashlib.sha256(resource.encode()).hexdigest()+'.json')
                if path.exists():
                    if load_json(path)!=dict(self.owner,resource=resource):
                        raise ProtocolError('RECOVERY_REQUIRED: foreign resource claim')
                    paths.append(path)
            for path in paths:
                path.unlink()
        return dict(cleanup_status='VERIFIED',claims_released=True,basis='no durable create acceptance')


def require_owned_container(inspect, task_id, token, container_id):
    if not isinstance(inspect, dict) or not isinstance(inspect.get('Config'), dict):
        raise ProtocolError('RECOVERY_REQUIRED: invalid container ownership evidence')
    labels = inspect['Config'].get('Labels')
    if not isinstance(labels, dict):
        raise ProtocolError('RECOVERY_REQUIRED: missing container ownership labels')
    if (inspect.get('Id') != container_id or labels.get('bench.task') != task_id or
            labels.get('bench.execution') != token):
        raise ProtocolError('RECOVERY_REQUIRED: container ownership mismatch')
