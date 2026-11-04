"""One owned node operation over JSON stdin; never execute arbitrary received code."""
import hashlib
import json
from pathlib import Path
import sys
import shutil
import time
from datetime import datetime, timezone
from stage_runtime import locked, read_plan, state, service_identity, http, save, transition


def execute(request):
    root = Path(request['run_dir'])
    if not root.is_absolute():
        raise ValueError('Absolute node run directory required')
    spec = read_plan(root)
    if spec['job_id'] != request['job_id'] or spec['service_fingerprint'] != request['fingerprint']:
        raise ValueError('Node identity differs from coordinator registration')
    for name, expected in request['scripts_sha256'].items():
        path=root/'scripts'/name
        if path.parent.resolve() != (root/'scripts').resolve() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen node script changed')
    with locked(spec):
        action = request['action']
        if action == 'check':
            from preflight import check
            return check(root)
        if action == 'start':
            from deploy import start_node
            return start_node(root,action='start_models')
        if action == 'start_proxy':
            from stage_runtime import command
            command(['docker','exec',state(root)['container_id'],'python3',str(Path(__file__).with_name('service_process.py')),'start_proxy',str(root)])
            return {'status':'PASSED'}
        if action == 'identity':
            if not request.get('include_proxy',True) and not spec['deployment']['service']['model_health_urls']:
                from stage_runtime import inspect_owned
                container=inspect_owned(spec,state(root).get('container_id'))
                if not container['State']['Running']:
                    raise ValueError('Proxy container is not running')
                return {'status':'PASSED','identity':{'container_id':container['Id']},'healthy':True}
            identity = service_identity(spec, root)
            try:
                for url in spec['deployment']['service']['health_urls' if request.get('include_proxy',True) else 'model_health_urls']:
                    http(url)
            except Exception as exc:
                return {'status':'PASSED','identity':identity,'healthy':False,'health_error':repr(exc)}
            return {'status': 'PASSED', 'identity': identity, 'healthy':True}
        if action == 'ready':
            identity = service_identity(spec, root)
            if identity != request['identity']:
                raise ValueError('Instance changed after deployment smoke')
            transition(root, 'READY', smoke_identity=identity)
            return {'status': 'PASSED'}
        if action == 'evaluate':
            from evaluate import evaluate
            return evaluate(root, request['case'])
        if action == 'release':
            if request['scope']=='failed_deployment':
                archive=root/'results'/('deployment-failure-'+str(time.time_ns()))
                archive.mkdir(parents=True)
                for path in (root/'logs').glob('*'):
                    if path.is_file(): shutil.copy2(path,archive/path.name)
            from finalize import release
            result = release(root, request['policy'], request['scope'])
            return {'status': 'PASSED', 'release': result}
        if action == 'assess_failure':
            from stage_runtime import inspect_owned, command
            data=state(root)
            if not data.get('container_id'):
                return {'status':'PASSED','service':'ABSENT'}
            container=inspect_owned(spec,data['container_id'])
            if not container['State']['Running']:
                return {'status':'PASSED','service':'DEAD'}
            case=data.get('cases',{}).get(request['case_id'],{})
            if case.get('status')=='RUNNING' or any(not a.get('cleanup',{}).get('released') for a in case.get('attempts',[])):
                return {'status':'PASSED','service':'UNKNOWN','reason':'Client cleanup unverified'}
            probe=json.loads(command(['docker','exec',container['Id'],'python3',str(Path(__file__).with_name('service_process.py')),'probe',str(root)]))
            if not probe['alive']:
                transition(root,'UNKNOWN',smoke_identity=None)
                return {'status':'PASSED','service':'DEAD'}
            identity=service_identity(spec,root)
            if identity!=data.get('smoke_identity'):
                return {'status':'PASSED','service':'UNKNOWN','reason':'Service generation not verified'}
            try:
                for url in spec['deployment']['service']['health_urls']: http(url)
            except Exception as exc:
                return {'status':'PASSED','service':'UNKNOWN','reason':repr(exc)}
            transition(root,'READY',cache_state='NEEDS_RESET')
            return {'status':'PASSED','service':'READY','identity':identity}
        if action == 'state':
            return {'status': 'PASSED', 'state': state(root)}
        raise ValueError('Unknown node action')


if __name__ == '__main__':
    request=json.load(sys.stdin)
    started=time.monotonic()
    try:
        result = execute(request)
    except Exception as exc:
        result = {'status': 'FAILED', 'error': repr(exc), 'error_type': type(exc).__name__}
    try:
        root=Path(request['run_dir'])
        spec=read_plan(root)
        if spec['job_id']==request['job_id'] and spec['service_fingerprint']==request['fingerprint']:
            log=root/'logs/node-operations.jsonl'
            log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as stream:
                stream.write(json.dumps({'time':datetime.now(timezone.utc).isoformat(),'action':request['action'],'seconds':time.monotonic()-started,'result':result})+'\n')
    except Exception as exc:
        result={'status':'FAILED','error':'Node evidence write failed: '+repr(exc),'operation_result':result}
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['status'] == 'PASSED' else 2)
