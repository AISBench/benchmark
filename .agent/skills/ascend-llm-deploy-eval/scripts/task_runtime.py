"""Serial task control. Configuration storage is an event hook, not a catalog."""
from datetime import datetime, timezone
import json
from pathlib import Path
from stage_runtime import digest, locked


def storage_event(root, event, **fields):
    path = root/'.internal/logs/config-storage-events.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'time': datetime.now(timezone.utc).isoformat(),
            'event': event, 'storage': 'not_implemented', **fields})+'\n')


def read_task(root):
    plan = json.loads((root/'common/plan.json').read_text(encoding='utf-8-sig'))
    spec = plan['job_spec']
    if spec['schema_version'] != '3.0' or digest(spec) != plan['job_spec_sha256']:
        raise ValueError('Invalid task plan or hash')
    if plan.get('sequence') != spec['sub_tasks']:
        raise ValueError('Execution sequence differs from frozen sub_tasks')
    return spec




def stage_entry(stage, root, bindings=None):
    spec=read_task(root)
    if stage == 'evaluate':
        from aisbench_cli import require_supported_mode
        for case in spec['cases']:
            require_supported_mode(case)
    from cluster_runtime import preflight, run, finalize
    with locked({'target':{'container':spec['job_id']+'-coordinator'}}):
        if stage=='preflight':
            values=json.loads(Path(bindings or root/'bindings.json').read_text(encoding='utf-8-sig'))
            frozen=root/'resolved-cluster.json'
            if frozen.exists():
                previous=json.loads(frozen.read_text())
                if previous['plan_sha256']!=digest(spec) or previous['bindings_sha256']!=digest(values):
                    raise ValueError('Frozen topology/bindings changed')
                return {'status':'PASSED','message':'Frozen node preparation reused; deploy rechecks resources','artifacts':[str(frozen)]}
            result=preflight(root,spec,values)
            storage_event(root,'bindings_resolved',task_id=spec['job_id'])
            return result
        resolved=json.loads((root/'resolved-cluster.json').read_text())
        if resolved['plan_sha256']!=digest(spec):
            raise ValueError('Topology bindings do not match plan')
        if stage=='evaluate':
            return run(root,spec,resolved)
        if stage=='finalize':
            return finalize(root,spec,resolved)
        raise ValueError('Use generated run.sh for topology deployment and execution')

