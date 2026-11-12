"""Bounded SSH transport. Actual hosts stay in task-local bindings, not examples."""
import json
import re
import shlex
import subprocess
import getpass
import sys


def is_local(host):
    """Use local transport only for the current user and an actual local IP."""
    user, separator, address = host.rpartition('@')
    address = address if separator else host
    if sys.platform.startswith('linux') and (not separator or user == getpass.getuser()):
        result=subprocess.run(['ip','-j','addr','show'],capture_output=True,text=True,timeout=10)
        if result.returncode==0:
            addresses={a['local'] for interface in json.loads(result.stdout)
                       for a in interface.get('addr_info',[]) if 'local' in a}
            if address in addresses:
                return True
    return False


def transport_argv(host, argv):
    if is_local(host):
        return list(argv)
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,shlex.join(argv)]


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
        result = subprocess.run(transport_argv(self.host, ['python3', '-B', self.script]),
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
