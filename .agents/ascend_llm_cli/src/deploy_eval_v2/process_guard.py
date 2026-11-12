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
    """No shell, no captured output. POSIX session is owned from creation."""
    remaining(deadline)
    if cancelled.is_set():
        raise Cancelled('Operation cancelled')
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    options = {'start_new_session': True} if os.name == 'posix' else {}
    with log.open('ab', buffering=0) as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, cwd=cwd, env=env, **options)
        try:
            while process.poll() is None:
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
