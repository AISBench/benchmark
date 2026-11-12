"""Atomic JSON snapshots; journal is always the recovery source of truth."""
import os
from pathlib import Path
import tempfile
from .protocol import encode


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(encode(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == 'posix':
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
def rename_new_directory(source,destination):
    """Publish a new directory; retry only Windows transient access/share denial.

    A destination appearing at any time is a conflict, never an overwrite.
    No retry here repeats remote commands or lifecycle effects.
    """
    import time
    from .protocol import ProtocolError
    deadline=time.monotonic()+1
    while True:
        if destination.exists():
            raise ProtocolError('REQUEST_CONFLICT: publication destination exists')
        try:
            source.rename(destination)
            return
        except PermissionError as error:
            if os.name!='nt' or getattr(error,'winerror',None) not in (5,32,33) or time.monotonic()>=deadline:
                raise
            time.sleep(.05)
