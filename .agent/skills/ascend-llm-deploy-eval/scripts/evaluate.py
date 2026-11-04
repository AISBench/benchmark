"""One case at a time; attempt budgets and results are scoped by case ID."""
import json
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone
from prepare_benchmark_tool import prepare
from task_events import emit
from stage_runtime import (entry, read_plan, ready, save, service_identity, state,
                           transition, case_state, update_case)


def evaluate(root, case):
    spec = read_plan(root)
    case_id = case['id']
    data = case_state(root, case_id)
    if data['status'] == 'PASSED':
        from validate_case import validate
        folder = Path(data['attempts'][-1]['folder'])
        validate(folder, json.loads((folder/'case.json').read_text()))
        return {'status': 'PASSED', 'message': 'Completed case revalidated; no replay',
                'artifacts': [str(folder/'validated.json')]}
    if data['status'] == 'RUNNING':
        raise ValueError('Interrupted case has unknown client state; cleanup required')
    if not ready(root, spec):
        raise ValueError('Node not READY; coordinator must verify the complete topology')
    limit = 1 + spec['execution']['max_retries']
    for attempt in range(len(data['attempts']) + 1, limit + 1):
        if not ready(root, spec):
            raise ValueError('Service not READY; no blind case replay')
        if spec.get('storage'):
            from disk_guard import check as disk_check
            disk_check(spec['storage']['root'], spec['storage']['subtasks'], Path(spec['storage']['root'])/'sub_tasks'/case_id)
        folder = Path(spec.get('storage', {}).get('root', root))/'sub_tasks'/case_id/f'attempt-{attempt:02}'
        folder.mkdir(parents=True, exist_ok=False)
        case = {**case, '_attempt': attempt}
        save(folder/'case.json', case)
        emit(root, 'attempt', 'doing', attempt=attempt, **case.get('_event_context', {'subcase_id':case_id}))
        item = {'attempt': attempt, 'folder': str(folder), 'status': 'RUNNING', 'stage': 'evaluate'}
        attempts = case_state(root, case_id)['attempts'] + [item]
        update_case(root, case_id, status='RUNNING', attempts=attempts, safe_to_continue=False)
        identity = service_identity(spec, root)
        transition(root, 'TESTING', cache_state='DIRTY')
        error = None
        public = None
        try:
            marker=json.loads((root/'deployment-artifacts.json').read_text())
            job=Path(spec.get('storage', {}).get('root', root))
            public=job/'sub_tasks'/case_id/f'attempt-{attempt:02}'
            prepare(Path(spec['benchmark']['tool_dir']), public/'tool')
            info={'sub_task_id':case_id,'attempt':attempt,'case':case,
                  'deployment_run_id':marker['id'],'deployment_path':marker['path'],
                  'started_at':datetime.now(timezone.utc).isoformat(),'status':'RUNNING'}
            save(public/'sub_task.json',info)
            with (folder/'runner.log').open('w') as log:
                result = subprocess.run(['docker', 'exec', state(root)['container_id'],
                    'python3', str(Path(__file__).with_name('case_runner.py')), str(root), str(folder)],
                    stdout=log, stderr=subprocess.STDOUT)
            returncode = result.returncode
        except Exception as exc:
            returncode = 2
            error = repr(exc)
        cleanup = json.loads((folder/'cleanup.json').read_text()) if (folder/'cleanup.json').exists() else {'released': False, 'error': 'No cleanup evidence'}
        summary = json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {'status': 'FAILED', 'error': error or 'Runner failed; see runner.log'}
        passed = returncode == 0 and summary['status'] == 'PASSED' and cleanup['released']
        if public is not None and (public/'sub_task.json').exists():
            info.update(finished_at=datetime.now(timezone.utc).isoformat(),summary=summary,
                        status='PASSED' if passed else 'FAILED',returncode=returncode,cleanup=cleanup)
            save(public/'sub_task.json',info)
        passed = returncode == 0 and summary['status'] == 'PASSED' and cleanup['released']
        try:
            unchanged = identity == service_identity(spec, root)
        except Exception:
            unchanged = False
        safe = bool(cleanup['released'] and unchanged)
        item.update(status='PASSED' if passed else 'FAILED', returncode=returncode,
                    summary=summary, cleanup=cleanup)
        attempts[-1] = item
        update_case(root, case_id, status=item['status'], attempts=attempts, safe_to_continue=safe)
        transition(root, 'READY' if unchanged else 'UNKNOWN', cache_state='NEEDS_RESET')
        emit(root, 'attempt', 'done' if passed else 'fail', attempt=attempt, **case.get('_event_context', {'subcase_id':case_id}))
        if passed:
            return {'status': 'PASSED', 'message': f'{case_id} passed on attempt {attempt}',
                    'artifacts': [str(folder/'validated.json')]}
        if 'Storage budget exceeded' in (summary.get('error') or ''):
            raise RuntimeError(summary['error'])
        if not safe or not case.get('reset_cache', True):
            raise ValueError('Unsafe to retry: client/service/cache isolation not verified')
        print(json.dumps({'event': 'case_attempt_failed', 'case_id': case_id,
                          'attempt': attempt, 'max_attempts': limit, 'error': summary.get('error')}), flush=True)
        if attempt < limit:
            emit(root, 'attempt', 'retry', attempt=attempt+1, **case.get('_event_context', {'subcase_id':case_id}))
            time.sleep(3)
    raise RuntimeError('Case exhausted retry budget; evidence retained')


if __name__ == '__main__':
    raise SystemExit(entry('evaluate'))
