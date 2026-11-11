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
        return {'returncode': result.returncode, 'stdout': result.stdout[:131072],
                'stderr': result.stderr[:131072], 'stdout_truncated_chars': max(0,len(result.stdout)-131072),
                'stderr_truncated_chars': max(0,len(result.stderr)-131072)}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'error': type(error).__name__}


def owner_pids(text):
    pids = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('/dev/'):
            label, separator, line = line.partition(':')
            if not separator:
                continue
            line = line.strip()
        if not re.fullmatch(r'(?:[1-9][0-9]*[a-zA-Z]*\s*)*', line):
            continue
        pids.update(int(value) for value in re.findall(r'[1-9][0-9]*', line))
    return sorted(pids)


def snapshot(devices, proc=Path('/proc')):
    evidence = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'npu_smi': command(['npu-smi', 'info'])}
    owners = command(['fuser'] + ['/dev/davinci'+str(i) for i in devices])
    evidence['device_owners'] = owners
    containers = command(['docker', 'ps', '--no-trunc', '--format', '{{.ID}} {{.Names}}'])
    evidence['containers'] = containers
    names = dict(line.split(' ', 1) for line in containers.get('stdout', '').splitlines() if ' ' in line)
    evidence['processes'] = []
    pids = owner_pids(owners.get('stdout', ''))
    evidence['pid_selection'] = {'limit':256, 'parsed':len(pids), 'discarded':max(0,len(pids)-256)}
    for pid in pids[:256]:
        item = {'pid': int(pid)}
        try:
            group = (proc/str(pid)/'cgroup').read_text()[:16384]
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
