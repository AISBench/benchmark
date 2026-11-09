"""Bounded, read-only host snapshots; no environment or process argv collection."""
import datetime
import json
from pathlib import Path
import re
import subprocess
import sys


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=3)
        return {'returncode': result.returncode, 'stdout': result.stdout[:131072]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'error': type(error).__name__}


def snapshot(devices, proc=Path('/proc')):
    evidence = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'npu_smi': command(['npu-smi', 'info'])}
    owners = command(['fuser'] + ['/dev/davinci'+str(i) for i in devices])
    evidence['device_owners'] = owners
    containers = command(['docker', 'ps', '--no-trunc', '--format', '{{.ID}} {{.Names}}'])
    evidence['containers'] = containers
    names = dict(line.split(' ', 1) for line in containers.get('stdout', '').splitlines() if ' ' in line)
    evidence['processes'] = []
    for pid in sorted(set(re.findall(r'\b\d+\b', owners.get('stdout', ''))), key=int)[:256]:
        item = {'pid': int(pid)}
        try:
            group = (proc/pid/'cgroup').read_text()[:16384]
            item['cgroup'] = group
            item['containers'] = {cid: name for cid, name in names.items() if cid in group}
        except OSError as error:
            item['error'] = type(error).__name__
        evidence['processes'].append(item)
    return evidence


if __name__ == '__main__':
    target = Path(sys.argv[1])
    devices = [int(value) for value in sys.argv[2:]]
    if not devices:
        raise SystemExit('Device list required')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(snapshot(devices), ensure_ascii=False)+'\n')
