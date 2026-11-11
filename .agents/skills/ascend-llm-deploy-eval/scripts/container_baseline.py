"""Required Ascend container bindings for the user's execution environment."""
from pathlib import Path

MOUNT_PATHS = [
    ('/root/host', '/root/host'),
    ('/usr/local/sbin', '/usr/local/sbin'),
    ('/usr/local/sbin/npu-smi', '/usr/local/sbin/npu-smi'),
    ('/usr/local/bin/npu-smi', '/usr/local/bin/npu-smi'),
    ('/usr/local/dcmi', '/usr/local/dcmi'),
    ('/var/log/npu', '/usr/slog'),
    ('/mnt', '/mnt'),
    ('/usr/lib64', '/usr/lib64'),
    ('/home', '/home'),
    ('/etc/hixlep', '/etc/hixlep'),
    ('/etc/hccl_rootinfo.json', '/etc/hccl_rootinfo.json'),
    ('/usr/local/Ascend/driver', '/usr/local/Ascend/driver'),
]
DEVICES = ['/dev/davinci_manager', '/dev/hisi_hdc', '/dev/ummu', '/dev/uburma'] + [
    '/dev/davinci'+str(i) for i in range(8)]
DEFAULT_CONTAINER = {'shm_size': '500g', 'user': 'root', 'pid_mode': 'host'}


def baseline_mounts(mounts, exists=None):
    exists = exists or (lambda p: Path(p).exists())
    result = [dict(m) for m in mounts]
    if len({m['target'] for m in result}) != len(result):
        raise ValueError('Duplicate mount targets')
    for source, target in MOUNT_PATHS:
        matches = [m for m in result if m['target'] == target]
        if matches:
            continue  # Explicit binding overrides the default source and access mode.
        else:
            result.append(dict(source=source, target=target, read_only=False))
    missing = [m['source'] for m in result if not exists(m['source'])]
    if missing:
        raise ValueError('Missing required Ascend container mounts: '+', '.join(missing))
    return result


def docker_run_args(container):
    if type(container.get('privileged')) is not bool:
        raise ValueError('Explicit privileged=true/false is required; recommended Ascend value: true')
    values = {**DEFAULT_CONTAINER, **container}
    return ['-u', values['user'], '-it', '--net=host', '--pid='+values['pid_mode'],
            '--shm-size='+values['shm_size'], '--privileged='+str(values['privileged']).lower()]
