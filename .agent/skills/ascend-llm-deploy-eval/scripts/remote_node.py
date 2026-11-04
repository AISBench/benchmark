"""Bounded SSH transport. Actual hosts stay in task-local bindings, not examples."""
import json
import re
import shlex
import subprocess


class PortConflict(RuntimeError):
    pass


class RemoteNode:
    def __init__(self, host, script, run_dir, job_id, fingerprint, scripts_sha256=None):
        if not re.fullmatch(r'(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9_.-]*', host):
            raise ValueError('Invalid SSH destination')
        self.host, self.script = host, script
        self.registration = dict(run_dir=run_dir, job_id=job_id, fingerprint=fingerprint, scripts_sha256=scripts_sha256 or {})

    def call(self, action, timeout=180, **values):
        payload = {**self.registration, **values, 'action': action}
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            self.host, shlex.join(['python3', '-B', self.script])],
            input=json.dumps(payload), capture_output=True, text=True, timeout=timeout)
        # Stage helpers can emit progress JSON before the final response.
        try:
            response = json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise RuntimeError('Node response unavailable: '+result.stderr[-2000:])
        if result.returncode or response.get('status') != 'PASSED':
            if response.get('error_type') == 'PortUnavailable':
                raise PortConflict('Node port unavailable: '+json.dumps(response))
            if str(response.get('error','')).startswith('ValueError('):
                raise ValueError('Node rejected operation: '+json.dumps(response))
            raise RuntimeError('Node operation failed: '+json.dumps(response))
        return response
