"""Fixed host-side launch/status RPC for a prepared job; no arbitrary command input."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from stage_runtime import digest, locked, save
from task_runtime import read_task
from service_process import ticks
from task_events import read_events, emit


def prepared(root, expected):
    spec = read_task(root)
    resolved = json.loads((root/'resolved-cluster.json').read_text())
    if digest(spec) != expected or resolved['plan_sha256'] != expected:
        raise ValueError('Prepared plan identity mismatch')
    if spec['cleanup']['task_end'] != 'stop_container':
        raise ValueError('Group execution requires verified task-end container release')
    if set(resolved['deployments']) != {d['id'] for d in spec['deployments']}:
        raise ValueError('Incomplete prepared deployment set')
    resources = set()
    streams = []
    for dep in resolved['deployments'].values():
        if not dep['groups']:
            raise ValueError('Empty prepared node set')
        for group in dep['groups'].values():
            resources.add(group['host'].split('@')[-1])
            streams.append({'host': group['host'], 'run_dir': group['run_dir'],
                            'script': str(Path(group['script']).with_name('group_worker.py'))})
    return spec, resolved, sorted(resources), streams


def terminal(root, spec, resolved):
    """A failed measurement may release dependencies; missing evidence never does."""
    summary = json.loads((root/'summary.json').read_text())
    expected = {t['id'] for t in spec['sub_tasks']}
    if set(summary.get('sub_tasks', {})) != expected:
        raise ValueError('Incomplete subtask results')
    for key, result in summary['sub_tasks'].items():
        if result.get('status') not in ['PASSED', 'FAILED', 'NOT_RUN']:
            raise ValueError('Subtask not terminal: '+key)
        if result['status'] != 'NOT_RUN':
            record = json.loads((root/'sub_tasks'/key/'result.json').read_text())
            if record != result:
                raise ValueError('Subtask evidence differs from summary: '+key)
    releases = json.loads((root/'release-results.json').read_text())
    safe = summary.get('cleanup', {}).get('status') == 'PASSED'
    if set(releases) != set(resolved['deployments']):
        safe = False
    for name, dep in resolved['deployments'].items():
        groups = releases.get(name, {})
        safe = safe and set(groups) == set(dep['groups']) and all(v.get('status') == 'PASSED' for v in groups.values())
    passed = safe and summary.get('status') == 'PASSED' and all(
        v.get('status') == 'PASSED' and not v.get('release_failed') for v in summary['sub_tasks'].values())
    return {'state': 'DONE' if safe else 'BLOCKED', 'result': 'PASSED' if passed else 'FAILED',
            'cleanup_verified': bool(safe), 'summary': str(root/'summary.json')}


def status(root, expected):
    spec, resolved, resources, streams = prepared(root, expected)
    marker = root/'group-worker.json'
    base = {'resources': resources, 'streams': streams, 'job_id': spec['job_id']}
    if not marker.exists():
        if (root/'task-state.json').exists() or (root/'summary.json').exists():
            return {**base, 'state': 'BLOCKED', 'error': 'Existing execution without group registration'}
        return {**base, 'state': 'PENDING'}
    identity = json.loads(marker.read_text())
    if identity['plan_sha256'] != expected:
        raise ValueError('Worker registration differs')
    if identity.get('start_ticks') and ticks(identity['pid']) == identity['start_ticks']:
        return {**base, 'state': 'RUNNING'}
    # The wrapper writes an exit record only after its child exits.
    if not (root/'group-worker-exit.json').exists():
        return {**base, 'state': 'BLOCKED', 'error': 'Worker interrupted or launch uncertain; inspect owned processes'}
    exit_record = json.loads((root/'group-worker-exit.json').read_text())
    if exit_record.get('plan_sha256') != expected:
        raise ValueError('Worker exit identity differs')
    try:
        result = terminal(root, spec, resolved)
        if exit_record.get('returncode') != 0:
            result['result'] = 'FAILED'
    except (ValueError, KeyError, OSError) as exc:
        result = {'state': 'BLOCKED', 'error': str(exc)}
    return {**base, **result}


def launch(root, expected):
    spec, _, _, _ = prepared(root, expected)
    with locked({'target': {'container': spec['job_id']+'-group-launch'}}):
        current = status(root, expected)
        if current['state'] != 'PENDING':
            return current  # Never replay an uncertain launch.
        save(root/'group-worker.json', {'plan_sha256': expected, 'pid': None, 'start_ticks': None})
        with (root/'logs/group-worker.log').open('ab') as log:
            proc = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                '--execute', str(root), '--sha256', expected], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(root/'group-worker.json', {'plan_sha256': expected, 'pid': proc.pid, 'start_ticks': ticks(proc.pid)})
        return {'state': 'RUNNING', 'job_id': spec['job_id']}


def execute(root, expected):
    spec, _, _, _ = prepared(root, expected)
    emit(root, 'task', 'doing', job_id=spec['job_id'])
    try:
        result = subprocess.run(['bash', str(root/'run.sh')], stdin=subprocess.DEVNULL)
        save(root/'group-worker-exit.json', {'returncode': result.returncode, 'plan_sha256': expected})
        emit(root, 'task', 'done' if result.returncode == 0 else 'fail', job_id=spec['job_id'], returncode=result.returncode)
    except Exception as exc:
        save(root/'group-worker-exit.json', {'error': repr(exc), 'plan_sha256': expected})
        emit(root, 'task', 'fail', job_id=spec['job_id'], error=repr(exc))


def rpc(request):
    root = Path(request['run_dir'])
    if not root.is_absolute():
        raise ValueError('Absolute task path required')
    if request['action'] == 'events':
        # Read-only node event transport; limited to the fixed event file.
        return read_events(root, request.get('offset', 0))
    expected = request['plan_sha256']
    if request['action'] == 'start':
        return launch(root, expected)
    if request['action'] in ['probe', 'status']:
        result = status(root, expected)
        result['events'] = read_events(root, request.get('offset', 0))
        return result
    raise ValueError('Unknown worker action')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    if args.execute:
        execute(args.execute, args.sha256)
    else:
        try:
            print(json.dumps({'status': 'PASSED', 'result': rpc(json.load(sys.stdin))}))
        except Exception as exc:
            print(json.dumps({'status': 'FAILED', 'error': repr(exc)}))
            raise SystemExit(2)
