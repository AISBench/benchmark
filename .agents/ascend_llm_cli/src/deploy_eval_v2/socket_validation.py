"""Explicit simulated-stage host acceptance; never submits stages for the Agent."""
from pathlib import Path
from functools import partial
import tempfile
from .protocol import ProtocolError, envelope
from .publication import build_archive
from .publication_cli import task_binding
from .ssh_publication import upload_archive
from .ssh_transport import exchange, ssh_argv


def start_socket(request, config):
    if set(request) != {'task_ref', 'nonce'}:
        raise ProtocolError('INVALID_INPUT: socket-start request')
    import re
    if not isinstance(request['nonce'], str) or not re.fullmatch('[a-f0-9]{64}', request['nonce']):
        raise ProtocolError('INVALID_INPUT: nonce')
    binding = task_binding(config, request['task_ref'])
    python = binding['python']
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary)/'runtime.zip'
        package = build_archive(Path(__file__).parent, archive)
        uploaded = upload_archive(binding['host'], python, binding['root'], archive, request['task_ref'])
    source = Path(__file__).with_name('socket_receiver.py').read_text(encoding='utf-8')
    packet = dict(task_id=request['task_ref'], nonce=request['nonce'], python=python,
                  runtime_sha256=package['runtime_sha256'], archive_sha256=uploaded['sha256'])
    return exchange(ssh_argv(binding['host'], [python, '-B', '-c', source, binding['root']]), packet, timeout=60)


def start_installed(root, request):
    from .startup import start
    from .launcher import ensure_started
    from .hashes import digest
    from .durable import atomic_json
    from .json_io import load_json
    root = Path(root)
    marker = dict(simulated=True, task_id=request['task_id'], runtime_sha256=request['runtime_sha256'])
    marker_path = root/'socket-validation.json'
    if (root/'inputs').exists():
        raise ProtocolError('REQUEST_CONFLICT: production input exists')
    if marker_path.exists() and load_json(marker_path) != marker:
        raise ProtocolError('REQUEST_CONFLICT: socket marker differs')
    atomic_json(marker_path, marker)
    packet = dict(protocol_version='1.0', task_id=request['task_id'], nonce=request['nonce'],
                  python=request['python'], runtime_sha256=request['runtime_sha256'], bundle_sha256=digest(marker))
    result = start(root, packet, starter=partial(ensure_started, fake=True))
    result.update(simulated=True, runtime_sha256=request['runtime_sha256'], bundle_sha256=digest(marker))
    return result


def stop_runtime(request, config):
    if set(request) != {'task_ref','nonce'}:
        raise ProtocolError('INVALID_INPUT: stop-runtime request')
    binding=task_binding(config,request['task_ref'])
    script='import sys; sys.path.insert(0, '+repr(binding['root']+'/runtime')+'); from deploy_eval_v2.socket_validation import stop_cli; stop_cli('+repr(binding['root'])+')'
    return exchange(ssh_argv(binding['host'],[binding['python'],'-B','-c',script]),
                    dict(task_id=request['task_ref'],nonce=request['nonce']),timeout=15)


def stop_cli(root):
    import sys,json
    request=json.load(sys.stdin)
    try:
        response=stop_installed(root,request)
    except ProtocolError as error:
        response=envelope(request.get('task_id'),0,'ERROR')
        response['error']=dict(code=str(error).split(':',1)[0],message=str(error)[:1000],retryable=False)
    print(json.dumps(response))


def stop_installed(root,request):
    import os,signal,time
    from .launcher import process_identity
    from .json_io import load_json
    from .remote_client import relay
    root=Path(root)
    identity=load_json(root/'.runtime/launch-identity.json')
    if identity['task_id']!=request['task_id'] or identity['nonce']!=request['nonce']:
        raise ProtocolError('REQUEST_CONFLICT: stop identity')
    status=relay(root,dict(protocol_version='1.0',task_id=request['task_id'],kind='get-status'))
    if status.get('task_state') not in ('NEW','RELEASED','CANCELLED','FAILED'):
        raise ProtocolError('RESOURCE_BUSY: release or cancel and verify resources before stopping runtime')
    expected=load_json(root/'.runtime/process.json')
    if process_identity(expected['pid'])!=expected:
        raise ProtocolError('RECOVERY_REQUIRED: process identity changed')
    import socket,json
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
        client.settimeout(5)
        client.connect(str(root/'.runtime/control.sock'))
        packet=dict(protocol_version='1.0',task_id=request['task_id'],kind='stop-runtime',nonce=request['nonce'])
        client.sendall(json.dumps(packet).encode()+b'\n')
        with client.makefile('rb') as stream:
            reply=json.loads(stream.readline(8193))
    if reply.get('pid')!=expected['pid'] or reply.get('stop_accepted') is not True:
        raise ProtocolError('RECOVERY_REQUIRED: stop not accepted')
    deadline=time.monotonic()+5
    while True:
        try:
            if process_identity(expected['pid'])!=expected:
                break
        except (FileNotFoundError,ProcessLookupError):
            break
        if time.monotonic()>=deadline:
            raise ProtocolError('UNKNOWN_OUTCOME: runtime exit not observed')
        time.sleep(.05)
    response=envelope(request['task_id'],status.get('next_cursor',0))
    response.update(runtime_stopped=True,process=expected)
    return response
