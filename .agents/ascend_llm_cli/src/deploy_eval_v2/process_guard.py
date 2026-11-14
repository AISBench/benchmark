"""Owned local process groups, source-routed logs and a single stage deadline."""
import os
from pathlib import Path
import signal
import subprocess
import time


class Cancelled(RuntimeError):
    pass


def remaining(deadline):
    value = deadline-time.monotonic()
    if value <= 0:
        raise TimeoutError('Stage deadline expired')
    return value


def run_logged(argv, log, deadline, cancelled, cwd=None, env=None):
    """Stream bounded diagnostics; preserve native output. Own the process group."""
    remaining(deadline)
    if cancelled.is_set():
        raise Cancelled('Operation cancelled')
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    options = {'start_new_session': True} if os.name == 'posix' else {}
    import contextlib,threading
    from .task_log import diagnostic
    native = log.name == 'aisbench.log'
    context = log.open('ab',buffering=0) if native else contextlib.nullcontext(subprocess.PIPE)
    diagnostic(log,__import__('json').dumps({'argv':argv},ensure_ascii=False),event='process.start')
    with context as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, cwd=cwd, env=env, **options)
        drain=None
        reader_errors=[]
        if not native:
            def consume():
                import codecs
                decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
                try:
                    while True:
                        data=process.stdout.read1(2048)
                        text=decoder.decode(data,final=not data)
                        if text: diagnostic(log,text)
                        if not data: break
                except Exception as error:
                    reader_errors.append(error)
                finally:
                    process.stdout.close()
            drain=threading.Thread(target=consume,daemon=True)
            drain.start()
        try:
            while process.poll() is None:
                if reader_errors:
                    raise RuntimeError("Framework log write failed") from reader_errors[0]
                if cancelled.is_set():
                    raise Cancelled('Operation cancelled')
                try:
                    process.wait(timeout=min(.1, remaining(deadline)))
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if os.name == 'posix':
                # Remove descendants even if the direct parent already exited.
                _stop(process)
            elif process.poll() is None:
                _stop(process)
            if drain is not None:
                drain.join(timeout=5)
                if drain.is_alive():
                    raise RuntimeError('Framework output reader did not terminate')
            diagnostic(log,'process.exit',returncode=process.returncode)
        if reader_errors:
            raise RuntimeError('Framework log write failed') from reader_errors[0]
        return process.returncode


def _stop(process):
    if os.name == 'posix':
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait()
