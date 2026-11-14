"""Detached Python launch with persistent process and live socket identity."""
import hashlib
import re
from pathlib import PurePosixPath
from .compiler import canonical
from .protocol import ProtocolError


def validate_identity(identity):
    required = {'protocol_version', 'task_id', 'task_root', 'runtime_sha256', 'bundle_sha256', 'nonce'}
    if not isinstance(identity, dict) or set(identity) != required or identity['protocol_version'] != '1.0':
        raise ProtocolError('INVALID_INPUT: runtime identity')
    if not isinstance(identity['task_id'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', identity['task_id']):
        raise ProtocolError('INVALID_INPUT: task identity')
    for key in ('runtime_sha256', 'bundle_sha256', 'nonce'):
        if not isinstance(identity[key], str) or not re.fullmatch('[a-f0-9]{64}', identity[key]):
            raise ProtocolError('INVALID_INPUT: identity digest or nonce')
    path = identity['task_root']
    if not isinstance(path, str) or not path.startswith('/') or path == '/' or '..' in PurePosixPath(path).parts or any(c in path for c in '\x00\r\n\\'):
        raise ProtocolError('INVALID_INPUT: task root')


def launch_argv(identity, python, identity_file, *, fake=False):
    validate_identity(identity)
    for value in (python, identity_file):
        if not isinstance(value,str) or not value.startswith('/') or '..' in PurePosixPath(value).parts:
            raise ProtocolError('INVALID_INPUT: launcher path')
    root=identity['task_root']
    bootstrap='import sys; sys.path.insert(0, '+repr(root+'/runtime')+'); from deploy_eval_v2.task_log import redirect_output; redirect_output('+repr(root)+'); from deploy_eval_v2.server import main; main()'
    return [python,'-B','-c',bootstrap,'--root',root,'--task-id',identity['task_id'],'--identity-file',identity_file]+(['--fake'] if fake else [])


def process_identity(pid):
    from pathlib import Path
    import os
    stat=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
    if stat[0]=='Z':
        raise ProcessLookupError(pid)
    return dict(pid=pid,start_ticks=stat[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),session=os.getsid(pid))


def spawn_owned(argv, log, record):
    import os,subprocess
    from pathlib import Path
    from .durable import atomic_json
    if os.name!='posix':
        raise ProtocolError('INVALID_INPUT: Linux required')
    with open(os.devnull,'ab',buffering=0) as output:
        process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    evidence=process_identity(process.pid)
    atomic_json(record,evidence)
    return evidence


def ensure_started(identity, python, identity_file, *, fake=False,allow_start=True):
    import time
    from pathlib import Path
    from .process_io import probe_identity
    from .json_io import load_json
    argv=launch_argv(identity,python,identity_file,fake=fake)
    root=Path(identity['task_root']);record=root/'.runtime/process.json'
    if not fake:
        from .executor_factory import build_executor
        build_executor(root,identity)
    if not record.exists():
        if not allow_start:
            raise ProtocolError('UNKNOWN_OUTCOME: prior launch requires reconciliation')
        evidence=spawn_owned(argv,root/'task.jsonl',record)
    else:
        evidence=load_json(record)
    deadline=time.monotonic()+10
    while True:
        try:
            if process_identity(evidence['pid'])!=evidence or evidence['session']!=evidence['pid']:
                raise ProtocolError('RECOVERY_REQUIRED: process identity mismatch')
            handshake=probe_identity(root,identity['task_id'],timeout=1)
            if handshake.get('identity')!=identity or handshake.get('pid')!=evidence['pid']:
                raise ProtocolError('RECOVERY_REQUIRED: socket identity mismatch')
            return dict(evidence,identity_verified=True,log=str(root/'task.jsonl'))
        except (OSError,ValueError):
            if time.monotonic()>=deadline:
                raise ProtocolError('UNKNOWN_OUTCOME: inspect runtime log and process receipt')
            time.sleep(.05)
