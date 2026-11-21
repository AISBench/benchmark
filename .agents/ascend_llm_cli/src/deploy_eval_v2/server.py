"""Linux Unix-socket runtime host. Fake mode must be explicitly requested."""
import argparse
import json
import os
from pathlib import Path
import socketserver
from .api import API
from .contracts import validate_query
from .artifacts import read_result
from .fake_runtime import FakeExecutor
from .task_runtime import Runtime
from .protocol import encode, envelope, ProtocolError
from .json_io import load_json,unique_object,reject_constant
from .launcher import validate_identity


def serve(root, task_id, executor, identity=None):
    import fcntl
    root = Path(root)
    control = root/'.runtime'
    control.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (control/'task.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if identity is not None:
            validate_identity(identity)
            if identity['task_id'] != task_id or identity['task_root'] != str(root):
                raise ProtocolError('INVALID_INPUT: launcher identity mismatch')
            identity_path = control/'identity.json'
            if identity_path.exists():
                if load_json(identity_path) != identity:
                    raise ProtocolError('RECOVERY_REQUIRED: existing runtime identity differs')
            else:
                with identity_path.open('xb') as stream:
                    stream.write(encode(identity))
                    stream.flush()
                    os.fsync(stream.fileno())
        api = API(Runtime(root, task_id, executor, repair_tail=True))
        endpoint = control/'control.sock'
        if endpoint.exists():
            endpoint.unlink()  # Exclusive owner lock acquired before stale socket removal.

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.request.settimeout(35)
                try:
                    raw = self.rfile.readline(65537)
                    if len(raw) > 65536 or not raw.endswith(b'\n'):
                        raise ProtocolError('INVALID_INPUT')
                    request = json.loads(raw,object_pairs_hook=unique_object,parse_constant=reject_constant)
                    if request.get('task_id') != task_id or request.get('protocol_version') != '1.0':
                        raise ProtocolError('INVALID_INPUT')
                    kind = request['kind']
                    if kind == 'stop-runtime':
                        if identity is None or request.get('nonce') != identity['nonce']:
                            raise ProtocolError('REQUEST_CONFLICT: runtime stop identity')
                        with api.runtime.lock:
                            if not api.runtime.can_stop():
                                raise ProtocolError('RESOURCE_BUSY: runtime holds active resources')
                            # Prevent new phases from racing an admitted shutdown.
                            api.runtime.task_cancelled = True
                        self.wfile.write(encode(dict(protocol_version='1.0',task_id=task_id,status='EVENTS',stop_accepted=True,pid=os.getpid()))+b'\n')
                        self.wfile.flush()
                        import threading
                        threading.Thread(target=self.server.shutdown,daemon=True).start()
                        return
                    if kind == 'identity':
                        self.wfile.write(encode({'protocol_version': '1.0', 'task_id': task_id,
                            'status': 'EVENTS', 'identity': identity, 'pid': os.getpid()})+b'\n')
                        return
                    if kind not in ('deploy', 'eval', 'release', 'cancel'):
                        validate_query(request)
                    if kind in ('deploy', 'eval', 'release', 'cancel'):
                        response = api.submit(request)
                    elif kind == 'get-status':
                        response = api.status()
                    elif kind == 'wait-event':
                        response = api.runtime.journal.watch(request['after_seq'], request.get('wait_ms', 30000), request.get('limit', 10))
                    elif kind == 'read-result':
                        response = read_result(root, task_id, request['artifact_ref'], request.get('offset', 0), request.get('max_bytes', 8192),
                            **{key:request[key] for key in ('tail_bytes','line_start','line_end','keyword','scan_offset','scan_line','expected_sha256') if key in request})
                    else:
                        raise ProtocolError('INVALID_INPUT: unsupported query')
                    if len(encode(response)) > (request.get('max_bytes', 8192) if kind == 'read-result' else 8192):
                        raise ProtocolError('TRANSPORT_PROTOCOL_ERROR')
                except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
                    code = str(exc).split(':', 1)[0] if isinstance(exc, ProtocolError) else 'INVALID_INPUT'
                    response = envelope(task_id, 0, 'ERROR')
                    response['error'] = dict(code=code, message=code, retryable=False)
                # Socket output is outside all runtime/journal locks.
                try:
                    self.wfile.write(encode(response)+b'\n')
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True

        with Server(str(endpoint), Handler) as server:
            os.chmod(endpoint, 0o600)
            server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--fake', action='store_true')
    parser.add_argument('--identity-file')
    args = parser.parse_args()
    identity = load_json(args.identity_file) if args.identity_file else None
    if args.fake:
        executor=FakeExecutor()
    else:
        if identity is None:
            parser.error('Production runtime requires a verified publication identity')
        from .executor_factory import build_executor
        executor=build_executor(args.root,identity)
    serve(args.root, args.task_id, executor, identity)


if __name__ == '__main__':
    main()
