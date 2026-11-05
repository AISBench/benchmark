"""Freeze a serial task, fill defaults, and snapshot execution code."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime
if '--validate-only' in sys.argv or '--check-dependencies' in sys.argv:
    sys.dont_write_bytecode = True
from stage_runtime import PACKAGE, digest, save


def load_json(path):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result: raise ValueError('Duplicate JSON key: '+key)
            result[key]=value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=pairs)


def normalize(spec):
    s=copy.deepcopy(spec)
    e=s.setdefault('execution',{})
    for k,v in {'smoke':'default','max_retries':3,'startup_timeout_seconds':1800,'smoke_timeout_seconds':240,
                'enable_cache_reset_api':True}.items():
        e.setdefault(k,v)
    if s.get('schema_version')=='3.0':
        from container_baseline import DEFAULT_CONTAINER, DEVICES
        for dep in s.get('deployments', []):
            if e['enable_cache_reset_api']:
                for instance in dep.get('instances', []):
                    if instance['role'] != 'proxy':
                        instance['launch'].setdefault('env', {})['VLLM_SERVER_DEV_MODE'] = '1'
            container = dep['runtime'].setdefault('container', {})
            for key, value in {**DEFAULT_CONTAINER, 'devices': DEVICES}.items():
                container.setdefault(key, copy.deepcopy(value))
        s.setdefault('cleanup',{'after_case':'keep_service','task_end':'stop_container'})
        for case in s.get('cases',[]):
            case.setdefault('mode', 'perf')
            if case['mode'] != 'perf':
                continue
            for k,v in {'formal_requests':case['concurrency']*4,'dataset_id':'default','num_warmups':1,
                        'steady_state':False,'request_rate':0,'warmup':True,'reset_cache':False,'timeout_seconds':5400}.items():
                case.setdefault(k,v)
    for subtask in s.get('sub_tasks', []):
        subtask.setdefault('after_case', s['cleanup']['after_case'])
        subtask.setdefault('phase_id', 'pp1')
        subtask.setdefault('on_failure', 'record_only')
    return s


def validate(spec):
    from jsonschema import Draft202012Validator
    errors=[f'{e.json_path}: {e.message}' for e in Draft202012Validator(load_json(PACKAGE/'schemas/job_spec.schema.json')).iter_errors(spec)]
    if errors: return errors
    from task_runtime import validate_task
    return validate_task(spec)


def check_dependencies():
    """Check the declared requirements without installing or changing anything."""
    from importlib.metadata import version
    from packaging.requirements import Requirement
    if sys.version_info < (3, 10):
        raise ValueError('Python >=3.10 required; configuration NOT validated')
    for line in (PACKAGE/'requirements.txt').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        installed = version(requirement.name)
        if installed not in requirement.specifier:
            raise ValueError(f'{requirement}: installed {installed}; configuration NOT validated')
    from jsonschema import Draft202012Validator
    Draft202012Validator.check_schema(load_json(PACKAGE/'schemas/job_spec.schema.json'))


def candidate_ids(date):
    if len(date) != 8 or not date.isascii() or not date.isdigit():
        raise ValueError('--date must be YYYYMMDD in the user timezone')
    datetime.strptime(date, '%Y%m%d')
    for number in range(1, 1000):
        yield f'{date}-{number:03d}'



def plan(spec,root):
    s=normalize(spec); errors=validate(s)
    if errors: raise ValueError('; '.join(errors))
    if s['state']!='ready': raise ValueError('Draft requires attended resolution')
    from aisbench_cli import require_supported_mode
    for case in s['cases']:
        require_supported_mode(case)
    result={'job_spec':s,'job_spec_sha256':digest(s),'status':'PLANNED','runtime_verified':False}
    if s['schema_version']=='3.0':
        result['sequence']=s['sub_tasks']
    if (root/'common/plan.json').exists() and load_json(root/'common/plan.json')!=result:
        raise ValueError('Existing plan differs; use a new job_id')
    root.mkdir(parents=True,exist_ok=True)
    (root/'common').mkdir(exist_ok=True)
    for case in s['cases']:
        save(root/'cases'/case['id']/'case.json', case)
    save(root/'common/plan.json',result); save(root/'common/job_spec.json',s)
    # Preserve the complete executable package used by this job.
    if not (root/'common/skill-snapshot').exists():
        shutil.copytree(PACKAGE,root/'common/skill-snapshot',ignore=shutil.ignore_patterns('__pycache__','*.pyc','test_*.py','references','examples','SKILL.md','agents','assets','docs','tests','README.md'))
    if s['schema_version']=='3.0':
        # Relative snapshot entry makes the generated task package transferable.
        (root/'run.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
            'RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
            'exec python3 "$RUN_DIR/common/skill-snapshot/scripts/run_task.py" --run-dir "$RUN_DIR" "$@"\n',encoding='utf-8',newline='\n')
        from task_runtime import storage_event
        storage_event(root,'task_normalized',task_id=s['job_id'],plan_sha256=result['job_spec_sha256'])
    return result


def main():
    p=argparse.ArgumentParser()
    destination=p.add_mutually_exclusive_group()
    destination.add_argument('--run-dir',type=Path)
    destination.add_argument('--jobs-dir',type=Path,help='Auto allocate a job directory when job_id is omitted')
    p.add_argument('--date',help='YYYYMMDD in user timezone, required for automatic IDs')
    p.add_argument('--job-spec',type=Path)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--check-dependencies',action='store_true')
    mode.add_argument('--validate-only',action='store_true')
    a=p.parse_args()
    try:
        try:
            check_dependencies()
        except Exception as exc:
            raise ValueError(f'Dependency check failed; configuration NOT validated: {exc}') from exc
        if a.check_dependencies:
            print(json.dumps({'stage':'plan_run','status':'DEPENDENCIES_OK','artifacts':[]}))
            return 0
        if a.job_spec is None: raise ValueError('--job-spec required')
        spec=load_json(a.job_spec)
        root=a.run_dir
        if a.jobs_dir:
            if not a.jobs_dir.is_absolute(): raise ValueError('Absolute --jobs-dir required')
            if 'job_id' in spec: raise ValueError('Auto allocation requires job_id to be omitted; explicit IDs use --run-dir')
            if not a.date: raise ValueError('--date required for automatic IDs')
            for job_id in candidate_ids(a.date):
                if not (a.jobs_dir/job_id).exists():
                    spec['job_id']=job_id
                    root=a.jobs_dir/job_id
                    break
            else: raise ValueError('No available job ID for date')
        elif a.date:
            raise ValueError('--date requires --jobs-dir')
        errors=validate(normalize(spec))
        if errors: raise ValueError('; '.join(errors))
        if a.validate_only:
            print(json.dumps({'stage':'plan_run','status':'VALIDATED','job_id':spec['job_id'],
                'message':'Configuration valid; no plan frozen, ID not reserved, runtime not checked','artifacts':[]}))
            return 0
        if root is None or not root.is_absolute(): raise ValueError('Absolute --run-dir or --jobs-dir required')
        if spec['state']!='ready': raise ValueError('Draft requires attended resolution')
        if a.jobs_dir:
            # Atomic reservation: concurrent planners cannot share an allocated ID.
            a.jobs_dir.mkdir(parents=True,exist_ok=True)
            for job_id in candidate_ids(a.date):
                root=a.jobs_dir/job_id
                try: root.mkdir()
                except FileExistsError: continue
                spec['job_id']=job_id
                break
            else: raise ValueError('No available job ID for date')
        plan(spec,root)
        print(json.dumps({'stage':'plan_run','status':'PLANNED','job_id':spec['job_id'],'message':'Plan saved; defaults and execution order recorded','artifacts':[str(root/'common/plan.json')]}))
        return 0
    except Exception as exc:
        print(json.dumps({'stage':'plan_run','status':'BLOCKED','message':str(exc),'artifacts':[]})); return 2

if __name__=='__main__': raise SystemExit(main())
