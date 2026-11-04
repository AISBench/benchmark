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


def validate_task(spec):
    from topology import validate_deployment
    errors = []
    deployments = {d['id']:d for d in spec['deployments']}
    cases = {c['id']:c for c in spec['cases']}
    if len(deployments) != len(spec['deployments']) or len(cases) != len(spec['cases']):
        errors.append('Duplicate deployment/case ID')
    for dep in deployments.values():
        import re
        for instance in dep['instances']:
            launch=instance.get('launch',{})
            for key in ('ASCEND_SLOG_PRINT_TO_STDOUT','ASCEND_GLOBAL_LOG_LEVEL'):
                if key in launch.get('env',{}) or re.search(r'(?<![A-Za-z0-9_])'+key+r'\s*=',launch.get('bootstrap','')):
                    errors.append('Remove operator logging assignment: '+key)
        errors.extend(validate_deployment(dep))
    if errors:
        return errors
    seen = set()
    for task in spec['sub_tasks']:
        if task['id'] in seen:
            errors.append('Duplicate sub_task ID')
        seen.add(task['id'])
        dep, case = deployments.get(task['deployment_id']), cases.get(task['case_id'])
        if dep is None or case is None:
            errors.append('Unknown deployment/case reference')
            continue
        limits = [i['launch'].get('args',{}).get('max-model-len',0) for i in dep['instances'] if i['role'] != 'proxy']
        if limits and case.get('input_tokens',0)+case['output_tokens'] > min(limits):
            errors.append('Context limit exceeded: '+task['id'])
    for case in cases.values():
        if case.get('formal_requests',case['concurrency']*4) < case['concurrency']:
            errors.append('Requests cannot reach concurrency: '+case['id'])
    if spec['state']=='ready' and spec['unresolved']:
        errors.append('Ready task has unresolved items')
    return errors


def stage_entry(stage, root, bindings=None):
    spec=read_task(root)
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
