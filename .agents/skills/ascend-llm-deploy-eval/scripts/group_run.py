"""One host process advances prepared jobs; independent hosts can run concurrently."""

if __name__ == '__main__':
    raise SystemExit('Retired lifecycle entry. Use scripts/run.py --mode FULL|scripts; historical frozen jobs retain their own code.')
import argparse
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from stage_runtime import digest, locked, save
from task_events import emit


def validate(plan):
    if set(plan) != {'group_id', 'tasks'}:
        raise ValueError('Group plan requires only group_id and tasks')
    if not re.fullmatch(r'\d{8}-\d{3}(?:-\d{3})*', plan['group_id']):
        raise ValueError('Invalid group_id')
    tasks = plan['tasks']
    ids = [t['id'] for t in tasks]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Empty/duplicate task IDs')
    locations = set()
    for task in tasks:
        if set(task) - {'id', 'host', 'run_dir', 'plan_sha256', 'depends_on'}:
            raise ValueError('Unknown group task fields')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', task['id']):
            raise ValueError('Invalid task ID')
        if not re.fullmatch(r'(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9_.-]*', task['host']):
            raise ValueError('Invalid SSH host')
        if not task['run_dir'].startswith('/') or '..' in Path(task['run_dir']).parts:
            raise ValueError('Absolute Linux task directory required')
        if not re.fullmatch(r'[a-f0-9]{64}', task['plan_sha256']):
            raise ValueError('Frozen plan hash required')
        location = (task['host'].split('@')[-1], task['run_dir'])
        if location in locations:
            raise ValueError('Duplicate task location')
        locations.add(location)
        if set(task.get('depends_on', [])) - set(ids):
            raise ValueError('Unknown dependency')
    remaining = {t['id']: set(t.get('depends_on', [])) for t in tasks}
    while remaining:
        ready = {key for key, deps in remaining.items() if not deps}
        if not ready:
            raise ValueError('Dependency cycle')
        remaining = {key: deps-ready for key, deps in remaining.items() if key not in ready}


def call(task, action, **values):
    script = task.get('script', str(Path(task['run_dir'])/'common/skill-snapshot/scripts/group_worker.py'))
    request = {**values, 'run_dir': task['run_dir'], 'action': action}
    if 'plan_sha256' in task:
        request['plan_sha256'] = task['plan_sha256']
    reply = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', task['host'],
        shlex.join(['python3', '-B', script])], input=json.dumps(request), text=True,
        capture_output=True, timeout=45)
    try:
        data = json.loads(reply.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise RuntimeError('Worker response unavailable: '+reply.stderr[-1000:])
    if reply.returncode or data.get('status') != 'PASSED':
        raise RuntimeError('Worker rejected request: '+str(data))
    return data['result']


def eligible(task, states):
    if states[task['id']]['state'] != 'PENDING':
        return False
    if any(states[d]['state'] != 'DONE' or not states[d].get('cleanup_verified') for d in task.get('depends_on', [])):
        return False
    resources = set(states[task['id']]['resources'])
    return not any(key != task['id'] and value['state'] in ['RUNNING', 'UNKNOWN', 'BLOCKED']
                   and resources.intersection(value['resources']) for key, value in states.items())


def collect(root, task, reply, cursors, transport):
    streams = [(task['id']+'/coordinator', reply['events'])] if 'events' in reply else []
    for stream in reply.get('streams', []):
        key = task['id']+'/'+stream['host']+'/'+stream['run_dir']
        try:
            streams.append((key, transport(stream, 'events', offset=cursors.get(key, 0))))
        except Exception:
            # Node-event transport is optional observability, not task failure.
            continue
    for key, batch in streams:
        for event in batch['events']:
            # Preserve source event ID for dedup if a crash occurs before cursor save.
            emit(root, event['stage'], event['status'], task_id=task['id'],
                 source_event=event, source_stream=key)
        cursors[key] = batch['offset']


def run(root, transport=call, sleeper=time.sleep):
    plan = json.loads((root/'group-plan.json').read_text())
    validate(plan)
    state_path = root/'group-state.json'
    saved = json.loads(state_path.read_text()) if state_path.exists() else {}
    if saved and saved['plan_sha256'] != digest(plan):
        raise ValueError('Frozen group plan changed')
    if (root/'group-summary.json').exists():
        return json.loads((root/'group-summary.json').read_text())
    states = saved.get('tasks', {})
    cursors = saved.get('cursors', {})
    # Inspect every prepared task before starting any of them.
    for task in plan['tasks']:
        if task['id'] not in states:
            states[task['id']] = transport(task, 'probe')
    def persist():
        save(state_path, {'plan_sha256': digest(plan), 'tasks': states, 'cursors': cursors})
    persist()
    emit(root, 'group', 'doing', group_id=plan['group_id'])
    while True:
        for task in plan['tasks']:
            key = task['id']
            old = states[key]
            if old['state'] in ['DONE', 'BLOCKED']:
                continue
            try:
                reply = transport(task, 'status', offset=cursors.get(key+'/coordinator', 0))
                collect(root, task, reply, cursors, transport)
                states[key] = {k: v for k, v in reply.items() if k not in ['events', 'streams']}
                if old['state'] != reply['state']:
                    emit(root, 'task', 'done' if reply['state'] == 'DONE' and reply.get('result') == 'PASSED'
                         else 'fail' if reply['state'] == 'DONE' else 'blocked' if reply['state'] == 'BLOCKED'
                         else 'doing', task_id=key, detail=states[key])
            except Exception as exc:
                count = old.get('connection_failures', 0)+1
                states[key] = {**old, 'state': 'UNKNOWN', 'connection_failures': count, 'error': repr(exc)}
                if count == 3:
                    emit(root, 'task', 'blocked', task_id=key, reason='Three observer failures; execution state unknown')
        for task in plan['tasks']:
            key = task['id']
            if eligible(task, states):
                # Persist before mutation; uncertain transport is resolved by worker identity, never replayed blindly.
                states[key]['state'] = 'UNKNOWN'
                persist()
                try:
                    reply = transport(task, 'start')
                    states[key].update(reply)
                    emit(root, 'task', 'doing', task_id=key, host=task['host'])
                except Exception as exc:
                    states[key]['error'] = repr(exc)
            elif states[key]['state'] == 'PENDING':
                blocked = any(states[d]['state'] == 'BLOCKED' for d in task.get('depends_on', []))
                blocked = blocked or any(v['state'] == 'BLOCKED' and set(v['resources']).intersection(states[key]['resources']) for v in states.values())
                if blocked:
                    states[key].update(state='BLOCKED', error='Dependency or shared-resource cleanup unverified')
                    emit(root, 'task', 'blocked', task_id=key, reason=states[key]['error'])
        persist()
        if all(v['state'] in ['DONE', 'BLOCKED'] for v in states.values()):
            summary = {'status': 'PASSED' if all(v['state'] == 'DONE' and v.get('result') == 'PASSED'
                       and v.get('cleanup_verified') for v in states.values()) else 'FAILED', 'tasks': states}
            save(root/'group-summary.json', summary)
            emit(root, 'group', 'done' if summary['status'] == 'PASSED' else 'fail', group_id=plan['group_id'])
            return summary
        sleeper(30)  # Cheap host-side polling, not an Agent/model call.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    p.add_argument('--start', action='store_true')
    a = p.parse_args()
    root = a.run_dir
    if not root.is_absolute():
        p.error('Absolute run-dir required')
    if a.start:
        if a.plan is None:
            p.error('--start requires --plan')
        plan = json.loads(a.plan.read_text(encoding='utf-8-sig')); validate(plan)
        root.mkdir(parents=True, exist_ok=True)
        with locked({'target': {'container': plan['group_id']+'-group-start'}}):
            if (root/'group-launch.json').exists():
                raise ValueError('Group already submitted; inspect state, do not launch again')
            if (root/'group-plan.json').exists() and json.loads((root/'group-plan.json').read_text()) != plan:
                raise ValueError('Group plan changed')
            for task in plan['tasks']:
                if call(task, 'probe')['state'] != 'PENDING':
                    raise ValueError('Task already executed or state uncertain')
            snapshot = root/'common/skill-snapshot'
            if snapshot.exists():
                raise ValueError('Existing group snapshot; inspect before resubmission')
            shutil.copytree(Path(__file__).resolve().parents[1], snapshot, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            save(root/'group-plan.json', plan)
            (root/'logs').mkdir(exist_ok=True)
            # Show the observer command before submitting the actual process.
            print('Observe on this host: '+shlex.join(['python3', str(snapshot/'scripts/watch_events.py'), '--run-dir', str(root)]), flush=True)
            save(root/'group-launch.json', {'state': 'SUBMITTING'})
            with (root/'logs/coordinator.log').open('ab') as log:
                proc = subprocess.Popen([sys.executable, '-B', str(snapshot/'scripts/group_run.py'), '--run-dir', str(root)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                from service_process import ticks
                save(root/'group-launch.json', {'pid': proc.pid, 'start_ticks': ticks(proc.pid)})
            print(json.dumps({'status': 'SUBMITTED', 'pid': proc.pid, 'run_dir': str(root)}))
        return 0
    plan = json.loads((root/'group-plan.json').read_text())
    with locked({'target': {'container': plan['group_id']+'-group'}}):
        try:
            return 0 if run(root)['status'] == 'PASSED' else 2
        except Exception as exc:
            try: save(root/'group-interrupted.json', {'error': repr(exc)})
            except OSError:
                import tempfile
                fallback=Path(tempfile.gettempdir())/('autobenchmark-interrupted-'+root.name+'.json')
                try: save(fallback, {'run_dir':str(root),'error':repr(exc)})
                except OSError: pass
            emit(root, 'group', 'blocked', error=repr(exc))
            raise


if __name__ == '__main__':
    raise SystemExit(main())
