"""Bounded transport; disconnect never implies cancellation or resubmission."""
import json
import re
import shlex
import subprocess
import threading
import time
from .protocol import encode, ProtocolError
from .json_io import unique_object, reject_constant


def ssh_argv(host, remote_argv):
    if not re.fullmatch(r'(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9_.-]*', host):
        raise ProtocolError('INVALID_INPUT')
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
            host, shlex.join(remote_argv)]


def exchange(argv, request, timeout=45, response_limit=8192, cancelled=None, input_file=None):
    """One final JSON response. Threads drain pipes with fixed memory bounds."""
    payload = encode(request) + b'\n'
    if len(payload) > 65536 or not 1 <= response_limit <= 32768 or not 0 < timeout <= 86400:
        raise ProtocolError('INVALID_INPUT')
    deadline = time.monotonic()+timeout
    if cancelled is not None and cancelled.is_set():
        raise ProtocolError('UNKNOWN_OUTCOME: observer cancelled')
    output, diagnostic = bytearray(), bytearray()
    overflow = threading.Event()
    try:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
    except OSError as exc:
        raise ProtocolError('SSH_UNAVAILABLE') from exc

    def drain(pipe, target, bound, tail=False):
        try:
            while True:
                block = pipe.read1(1024)
                if not block:
                    break
                if tail:
                    target.extend(block)
                    del target[:-bound]
                elif len(target) + len(block) > bound:
                    overflow.set()
                    if process.poll() is None:
                        process.kill()
                    break
                else:
                    target.extend(block)
        finally:
            pipe.close()

    def send():
        try:
            process.stdin.write(payload)
            if input_file is not None:
                with open(input_file,'rb') as source:
                    sent=0
                    while block:=source.read(65536):
                        sent+=len(block)
                        if sent>64*1024*1024:
                            process.kill()
                            return
                        process.stdin.write(block)
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            process.stdin.close()

    workers = [threading.Thread(target=drain, args=(process.stdout, output, response_limit+1),daemon=True),
               threading.Thread(target=drain, args=(process.stderr, diagnostic, 2048, True),daemon=True),
               threading.Thread(target=send,daemon=True)]
    for worker in workers:
        worker.start()
    try:
        while process.poll() is None:
            if (cancelled is not None and cancelled.is_set()) or time.monotonic()>=deadline:
                raise subprocess.TimeoutExpired(argv, timeout)
            try:
                process.wait(timeout=min(.05,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for worker in workers:
            worker.join(timeout=max(0,deadline-time.monotonic()))
        if any(worker.is_alive() for worker in workers):
            raise subprocess.TimeoutExpired(argv,timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        raise ProtocolError('UNKNOWN_OUTCOME') from exc
    if overflow.is_set():
        raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: oversized response')
    if process.returncode not in (0, 2, 3, 4):
        raise ProtocolError('UNKNOWN_OUTCOME')
    try:
        value = json.loads(output,object_pairs_hook=unique_object,parse_constant=reject_constant)
        if len(encode(value)) > response_limit:
            raise ValueError('budget exceeded')
        if value['protocol_version'] != '1.0' or value['task_id'] != request['task_id']:
            raise ValueError('identity mismatch')
        if value['status'] not in ('EVENTS', 'RUNNING', 'WAIT_TIMEOUT', 'ERROR'):
            raise ValueError('invalid status')
    except (ValueError, KeyError, TypeError) as exc:
        raise ProtocolError('TRANSPORT_PROTOCOL_ERROR') from exc
    return value
