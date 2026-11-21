"""Bounded subprocess commands and live Unix socket identity probe."""
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
from .json_io import unique_object, reject_constant
from .protocol import ProtocolError, encode


def bounded_command(argv, timeout=10, limit=8192):
    if not 0 < timeout <= 30 or not 1 <= limit <= 32768:
        raise ProtocolError('INVALID_INPUT')
    stdout, stderr = bytearray(), bytearray()
    overflow = threading.Event()
    options = {'start_new_session': True} if os.name == 'posix' else {}
    try:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, **options)
    except OSError as exc:
        raise ProtocolError('SSH_UNAVAILABLE: system manager executable unavailable') from exc
    def stop():
        try:
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
        except ProcessLookupError:
            pass
    def drain(pipe, target, tail=False):
        try:
            while True:
                block = pipe.read1(1024)
                if not block:
                    return
                if tail:
                    target.extend(block)
                    del target[:-2048]
                elif len(target)+len(block) > limit:
                    overflow.set()
                    stop()
                    return
                else:
                    target.extend(block)
        finally:
            pipe.close()
    workers = [threading.Thread(target=drain, args=(process.stdout, stdout), daemon=True),
               threading.Thread(target=drain, args=(process.stderr, stderr, True), daemon=True)]
    deadline = time.monotonic()+timeout
    for worker in workers:
        worker.start()
    try:
        process.wait(timeout=timeout)
        for worker in workers:
            worker.join(max(0, deadline-time.monotonic()))
        if any(worker.is_alive() for worker in workers):
            raise subprocess.TimeoutExpired(argv, timeout)
    except subprocess.TimeoutExpired as exc:
        stop()
        process.wait()
        raise ProtocolError('UNKNOWN_OUTCOME: manager command deadline') from exc
    finally:
        if process.poll() is None:
            stop()
            process.wait()
    if overflow.is_set():
        raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: manager output exceeds budget')
    return process.returncode, bytes(stdout), bytes(stderr)


def probe_identity(root, task_id, timeout=5):
    if not 0 < timeout <= 30:
        raise ProtocolError('INVALID_INPUT')
    request = dict(protocol_version='1.0', task_id=task_id, kind='identity')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(Path(root)/'.runtime/control.sock'))
        client.sendall(encode(request)+b'\n')
        with client.makefile('rb') as reader:
            raw = reader.readline(8194)
    if len(raw) > 8193 or not raw.endswith(b'\n'):
        raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: identity response size')
    try:
        value = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
        if value['protocol_version'] != '1.0' or value['task_id'] != task_id or value['status'] != 'EVENTS':
            raise ValueError('response identity')
        return value
    except (ValueError, KeyError, TypeError) as exc:
        raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: identity response') from exc
