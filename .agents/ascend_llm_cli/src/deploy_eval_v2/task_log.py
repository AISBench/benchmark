"""One task-local JSONL stream for events and framework diagnostics."""
import contextlib
import io
import json
import os
from pathlib import Path
import threading
import traceback
from .protocol import now

_lock = threading.RLock()


def task_root(path):
    path = Path(path).resolve()
    folder = path if path.is_dir() else path.parent
    for candidate in (folder, *folder.parents):
        if (candidate/'.runtime').is_dir() or (candidate/'inputs').is_dir() or (candidate/'task.jsonl').exists():
            return candidate
    return folder


@contextlib.contextmanager
def locked_stream(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open('a+b',buffering=0) as stream:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        else:
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            stream.seek(0, 2)
            yield stream
        finally:
            if os.name == 'posix':
                fcntl.flock(stream, fcntl.LOCK_UN)
            else:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def append(path, record):
    raw=json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')+b'\n'
    with locked_stream(path) as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def diagnostic(path, message, **fields):
    root=task_root(path)
    text=str(message)
    for offset in range(0, max(1,len(text)), 2048):
        append(root/'task.jsonl',dict(record_type='diagnostic',time=now(),source=Path(path).name,
                                     message=text[offset:offset+2048],**fields))
    return root/'task.jsonl'


class DiagnosticStream(io.TextIOBase):
    def __init__(self,path):
        self.path=path
    def write(self,value):
        if value:
            diagnostic(self.path,value)
        return len(value)
    def flush(self):
        pass


def record_exception(path):
    return diagnostic(path,traceback.format_exc(),level='ERROR')


def redirect_output(root):
    import sys
    root=Path(root)
    (root/'.runtime').mkdir(parents=True,exist_ok=True)
    sys.stdout=DiagnosticStream(root/'runtime')
    sys.stderr=DiagnosticStream(root/'runtime')
