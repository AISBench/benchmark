"""Publish only the pinned execution runtime and explicit inputs; never start a job."""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

from bundle_distribution import PROGRAM
from remote_node import transport_argv

PACKAGE = Path(__file__).resolve().parents[1]
MODULES = (
    'run', 'bundle_distribution', 'remote_node', 'shared_storage', 'task_model',
    'topology', 'container_baseline', 'mount_profiles', 'review_job',
    'container_dependencies', 'aisbench_cli', 'test_evidence',
    'validate_case', 'cache_reset', 'metric_evidence', 'ownership_evidence',
    'watch_events', 'task_events', 'watch_logs',
)
RECEIVER = PROGRAM.replace(
    "root.parent.mkdir(parents=True, exist_ok=True)",
    """base = pathlib.Path(p['work_root'])
if root.parent != base or root.is_symlink() or root.resolve().parent != base.resolve():
    raise SystemExit('Runtime path escapes declared work root')
root.parent.mkdir(parents=True, exist_ok=True)""",
)


def build(host, work_root, task_id, inputs=(), package=PACKAGE):
    if not re.fullmatch(r'(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9_.-]*', host):
        raise ValueError('Invalid SSH destination')
    base = PurePosixPath(work_root)
    if not base.is_absolute() or '..' in base.parts or str(base) == '/':
        raise ValueError('An absolute non-root remote work directory is required')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', task_id):
        raise ValueError('Invalid task identity')
    root = str(base / (task_id + '.runtime'))
    data = {}
    for name in MODULES:
        data['scripts/' + name + '.py'] = (package / 'scripts' / (name + '.py')).read_bytes()
    for name in ('job_spec', 'bindings', 'shared_storage', 'status'):
        data['schemas/' + name + '.schema.json'] = (package / 'schemas' / (name + '.schema.json')).read_bytes()
    data['requirements.txt'] = (package / 'requirements.txt').read_bytes()
    input_paths = {}
    for source in inputs:
        source = Path(source).resolve()
        if not source.is_file() or not re.fullmatch(r'[A-Za-z0-9_.-]+', source.name):
            raise ValueError('Input must be a readable file with a portable basename')
        key = 'inputs/' + source.name
        if key in data:
            raise ValueError('Duplicate input basename: ' + source.name)
        data[key] = source.read_bytes()
        input_paths[str(source)] = root + '/' + key
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest = dict(task_id=task_id, work_root=str(base), runtime_root=root,
                    runtime_sha256=identity, files=hashes, input_paths=input_paths)
    data['runtime-manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    payload = dict(root=root, work_root=str(base),
                   files={name: base64.b64encode(value).decode() for name, value in data.items()},
                   hashes={name: hashlib.sha256(value).hexdigest() for name, value in data.items()})
    return manifest, payload


def publish(host, work_root, task_id, inputs=(), *, runner=subprocess.run, package=PACKAGE):
    manifest, payload = build(host, work_root, task_id, inputs, package)
    result = runner(transport_argv(host, ['python3', '-B', '-c', RECEIVER]),
                    input=json.dumps(payload), capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError('Runtime publication failed; retain evidence, do not overwrite: ' + result.stderr)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--work-root', required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--input', action='append', default=[], type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    # Reserve the local receipt before touching the remote host; never replace evidence.
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x', encoding='utf-8') as stream:
        try:
            manifest, _ = build(args.host, args.work_root, args.task_id, args.input)
            if not args.plan_only:
                manifest = publish(args.host, args.work_root, args.task_id, args.input)
            record = dict(status='PLANNED' if args.plan_only else 'PUBLISHED', **manifest)
        except BaseException as error:
            json.dump(dict(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                           host=args.host, task_id=args.task_id,
                           error=str(error)), stream, indent=2)
            raise
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
