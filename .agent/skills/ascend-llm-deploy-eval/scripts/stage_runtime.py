"""Shared stage IO, ownership checks, service state and container-wide lock."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

PACKAGE = Path(__file__).resolve().parents[1]


def command(argv, timeout=60):
    r=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
    if r.returncode:
        raise RuntimeError(f'{argv[0]} failed ({r.returncode}): {r.stderr[-4000:]} {r.stdout[-1000:]}')
    return r.stdout


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)


def read_plan(root):
    plan=json.loads((root/'plan.json').read_text(encoding='utf-8-sig'))
    s=plan['execution_spec']
    if s.get('schema_version') != 'resolved-1':
        raise ValueError('Expected v3 preflight execution artifact')
    if digest(s)!=plan['execution_sha256']:
        raise ValueError('Plan hash mismatch')
    if root.resolve()!=Path(s['target']['work_dir']).resolve():
        raise ValueError('Run directory differs from target work_dir')
    return s


def ownership_digest(spec):
    return spec.get('container_fingerprint',spec['service_fingerprint'])


def pool_state(spec):
    path=Path(spec['container_registry'])
    if not path.exists(): return {}
    record=json.loads(path.read_text())
    if record.get('fingerprint')!=ownership_digest(spec):
        raise ValueError('Task container registry configuration changed')
    return record


def save_pool(spec, **values):
    record=pool_state(spec);record.update(values,fingerprint=ownership_digest(spec))
    save(spec['container_registry'],record)


def case_state(root,case_id):
    return state(root).get('cases',{}).get(case_id,{'status':'NOT_RUN','attempts':[]})


def update_case(root,case_id,**values):
    data=state(root)
    current=data.setdefault('cases',{}).setdefault(case_id,{'status':'NOT_RUN','attempts':[]})
    current.update(values)
    save(root/'run_state.json',data)
    return current


def state(root):
    path=root/'run_state.json'
    return json.loads(path.read_text()) if path.exists() else {'service':'ABSENT','cases':{},'releases':[]}


def update(root,**values):
    data=state(root); data.update(values); save(root/'run_state.json',data); return data


def transition(root,value,**extra):
    previous=state(root)['service']
    with (root/'logs/state-transitions.jsonl').open('a') as f:
        f.write(json.dumps({'time':datetime.now(timezone.utc).isoformat(),'from':previous,'to':value,**extra})+'\n')
    return update(root,service=value,**extra)


def inspect_owned(spec, expected_id=None):
    c=json.loads(command(['docker','inspect',spec['target']['container']]))[0]
    labels=c['Config'].get('Labels') or {}
    if labels.get('benchmark.job_id')!=spec['job_id'] or labels.get('benchmark.spec_sha256')!=ownership_digest(spec):
        raise ValueError('Container ownership/configuration mismatch')
    if expected_id is not None and c['Id']!=expected_id:
        raise ValueError('Container identity changed')
    return c


@contextmanager
def locked(spec):
    # Same container name maps to the same host-wide lock, regardless of work_dir.
    import fcntl
    directory=Path('/tmp/autobenchmark-locks'); directory.mkdir(exist_ok=True)
    path=directory/(hashlib.sha256(spec['target']['container'].encode()).hexdigest()+'.lock')
    with path.open('a+') as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Container is locked by another stage/case')
        try:
            yield
        finally:
            fcntl.flock(f,fcntl.LOCK_UN)


def http(url,method='GET'):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url,method=method),timeout=10) as r:
        return r.read().decode()


def service_identity(spec,root):
    data=state(root)
    if spec.get('container_registry') and pool_state(spec).get('active_run_dir')!=str(root):
        raise ValueError('Container is assigned to another service')
    c=inspect_owned(spec,data.get('container_id'))
    if not c['State']['Running'] or c['Image']!=data.get('image_id'):
        raise ValueError('Registered container is not running or image changed')
    raw=command(['docker','exec',c['Id'],'python3',str(Path(__file__).with_name('service_process.py')),'identity',str(root)])
    return {'container_id':c['Id'],'container_started_at':c['State']['StartedAt'],'processes':json.loads(raw)}


def ready(root,spec):
    data=state(root)
    if data['service']!='READY':
        return False
    try:
        if data.get('smoke_identity')!=service_identity(spec,root):
            raise ValueError('Service instance changed; new smoke required')
        service=json.loads((root/'scripts/service.json').read_text())
        for url in service['health_urls']:
            http(url)
    except Exception as exc:
        transition(root,'UNKNOWN',reason=str(exc))
        return False
    return True


def record(root,stage,operation):
    started=time.monotonic(); result={'stage':stage,'started_at':datetime.now(timezone.utc).isoformat(),'artifacts':[]}
    result['executor']={'package':str(PACKAGE),'script_sha256':{
        p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (PACKAGE/'scripts').glob('*.py')}}
    try:
        result.update(operation())
    except Exception as exc:
        result.update(status='FAILED',message=repr(exc))
    result.update(seconds=time.monotonic()-started,finished_at=datetime.now(timezone.utc).isoformat())
    (root/'logs').mkdir(parents=True,exist_ok=True)
    with (root/'logs/stages.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(result,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result['status']=='PASSED' else 2


def entry(stage):
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--bindings',type=Path)
    args=p.parse_args()
    plan_path = args.run_dir/'common/plan.json'
    error = ('Absolute run-dir required: '+str(args.run_dir) if not args.run_dir.is_absolute()
             else 'Task plan missing: '+str(plan_path) if not plan_path.is_file() else None)
    if error:
        print(json.dumps({'stage':stage,'status':'BLOCKED','message':error,'artifacts':[]}))
        return 2
    def run():
        from task_runtime import stage_entry
        return stage_entry(stage,args.run_dir,args.bindings)
    return record(args.run_dir,stage,run)
