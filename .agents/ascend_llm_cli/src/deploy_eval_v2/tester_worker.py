"""One supervised tester operation, not a second task runtime or coordinator."""
import argparse
import json
from pathlib import Path
import re
import sys
import threading
import time
import traceback
from .compiler import verify_frozen
from .publication import verify_runtime
from .hashes import digest
from .durable import atomic_json
from .json_io import load_json,unique_object,reject_constant
from .owned_eval import owned_eval_inputs
from .process_guard import run_logged
from .protocol import ProtocolError,envelope,encode
from .remote_guard import exclusive
from .process_io import bounded_command
from .launcher import spawn_owned, process_identity


def verified_inputs(root,request):
    fields={'protocol_version','task_id','operation_id','case_id','runtime_sha256','bundle_sha256','token','action'}
    if (not isinstance(request,dict) or set(request)!=fields or request['protocol_version']!='1.0' or
            request['action'] not in ('run','cancel')):
        raise ProtocolError('INVALID_INPUT: tester request')
    for key in ('task_id','operation_id','case_id'):
        if not isinstance(request[key],str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}',request[key]):
            raise ProtocolError('INVALID_INPUT: tester identity')
    for key in ('runtime_sha256','bundle_sha256','token'):
        if not isinstance(request[key],str) or not re.fullmatch('[a-f0-9]{64}',request[key]):
            raise ProtocolError('INVALID_INPUT: tester digest')
    verify_runtime(root/'runtime',request['runtime_sha256'])
    values=verify_frozen(root/'inputs',request['bundle_sha256'],request['runtime_sha256'])
    plan,bound=values['plan.json'],values['bindings.json']
    if plan['intent']!='deploy' or plan['task_id']!=request['task_id']:
        raise ProtocolError('INVALID_INPUT: remote tester task')
    tester=bound['testers'][plan['evaluation']['test_binding']]
    from .executor_factory import local_machine_id
    if tester.get('machine_id')!=local_machine_id() or root!=Path(tester['work_root'])/plan['task_id']:
        raise ProtocolError('INVALID_INPUT: remote tester machine or root')
    compiled=values['compiled.json']
    service=dict(service_id='svc-'+digest([plan['task_id'],request['bundle_sha256']])[:32],
        endpoint=compiled['topology']['endpoint'],model=compiled['model'],metrics=compiled['topology']['metric_instances'])
    adapted,bindings=owned_eval_inputs(plan,bound,service)
    case=next((c for c in adapted['cases'] if c['case_id']==request['case_id']),None)
    if case is None:
        raise ProtocolError('INVALID_INPUT: remote tester case')
    return adapted,bindings,tester,case


def operation_paths(root,request):
    folder=root/'results'/'tester'/request['operation_id']
    identity={k:v for k,v in request.items() if k!='action'}
    unit=digest(identity)[:32]
    return folder,identity,unit


def execute(root,request,*,evaluator=None):
    plan,bound,tester,case=verified_inputs(root,request)
    folder,identity,unit=operation_paths(root,request)
    if load_json(folder/'request.json')!=dict(identity,action='run'):
        raise ProtocolError('RECOVERY_REQUIRED: tester execution identity')
    if (folder/'cancelled.json').exists():
        raise ProtocolError('RECOVERY_REQUIRED: cancelled before tester start')
    from .evaluate import execute_external_case
    evaluator=evaluator or execute_external_case
    try:
        cancelled=threading.Event()
        def observe_cancel():
            while not cancelled.wait(.1):
                if (folder/'cancelled.json').exists():
                    cancelled.set()
        threading.Thread(target=observe_cancel,daemon=True).start()
        try:
            result=evaluator(plan,bound,request['case_id'],folder/'benchmark',cancelled,verified_local_host=tester['host'])
        finally:
            cancelled.set()
        from .result_contract import evaluation_result
        result=evaluation_result(result,task_id=request['task_id'],operation_id=request['operation_id'],
            service_id=plan['service']['service_id'],case_id=request['case_id'])
        response=envelope(request['task_id'],0)
        response.update(operation_id=request['operation_id'],result=result,result_sha256=digest(result),
                        evidence=dict(host=tester['host'],path=str(folder/'benchmark')),cleanup_status='VERIFIED')
    except Exception:
        with (folder/'error.log').open('w',encoding='utf-8') as stream:
            traceback.print_exc(file=stream)
        response=envelope(request['task_id'],0,'ERROR')
        response.update(operation_id=request['operation_id'],cleanup_status='VERIFIED',
            error=dict(code='EVALUATION_FAILED',message='Remote evaluator failed; inspect evidence',retryable=False),
            evidence=dict(host=tester['host'],path=str(folder/'error.log')))
    if len(encode(response))>8192:
        raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: tester result budget')
    atomic_json(folder/'response.json',response)


def dispatch(root,request,*,runner=bounded_command,logged=run_logged):
    root=Path(root)
    plan,bound,tester,case=verified_inputs(root,request)
    folder,identity,unit=operation_paths(root,request)
    # Acceptance persists before detached launch and is never blindly repeated.
    with exclusive(root/'.runtime/tester-identity.lock'):
        identity_path=root/'.runtime/tester-identity.json'
        task_identity={k:identity[k] for k in ('task_id','runtime_sha256','bundle_sha256','token')}
        if identity_path.exists() and load_json(identity_path)!=task_identity:
            raise ProtocolError('RECOVERY_REQUIRED: tester task execution identity')
        if not identity_path.exists():
            atomic_json(identity_path,task_identity)
    folder.mkdir(parents=True,exist_ok=True)
    with exclusive(folder/'launch.lock'):
        saved=folder/'request.json'
        expected=dict(identity,action='run')
        if saved.exists() and load_json(saved)!=expected:
            raise ProtocolError('REQUEST_CONFLICT: tester operation')
        if request['action']=='cancel':
            # Persist a tombstone even if cancel arrived before run acceptance.
            atomic_json(folder/'cancelled.json',identity)
            if not saved.exists():
                atomic_json(saved,expected)
            record=folder/'process.json'
            if record.exists():
                expected_process=load_json(record)
                deadline=time.monotonic()+10
                while not (folder/'response.json').exists():
                    try:
                        if process_identity(expected_process['pid'])!=expected_process:
                            raise ProtocolError('RECOVERY_REQUIRED: tester identity changed')
                    except OSError:
                        raise ProtocolError('RECOVERY_REQUIRED: tester died without cleanup evidence')
                    if time.monotonic()>=deadline:
                        raise ProtocolError('RECOVERY_REQUIRED: tester cancellation unverified')
                    time.sleep(.1)
            elif (folder/'launch-accepted.json').exists():
                raise ProtocolError('UNKNOWN_OUTCOME: tester launch identity unavailable')
            response=envelope(request['task_id'],0)
            response.update(operation_id=request['operation_id'],cleanup_status='VERIFIED')
            return response
        if (folder/'cancelled.json').exists():
            raise ProtocolError('RECOVERY_REQUIRED: tester operation cancelled')
        if (folder/'response.json').exists():
            return load_json(folder/'response.json',8192)
        if saved.exists():
            raise ProtocolError('UNKNOWN_OUTCOME: tester operation already accepted')
        atomic_json(saved,expected)
        atomic_json(folder/'launch-accepted.json',identity)
        bootstrap='import sys; sys.path.insert(0, '+repr(str(root/'runtime'))+'); from deploy_eval_v2.tester_worker import main; raise SystemExit(main())'
        argv=[tester['python'],'-B','-c',bootstrap,'--root',str(root),'--execute',str(saved)]
        spawn_owned(argv,folder/'worker.log',folder/'process.json')
    deadline=time.monotonic()+case['timeout_seconds']+20
    while not (folder/'response.json').exists():
        if time.monotonic()>=deadline:
            raise ProtocolError('UNKNOWN_OUTCOME: tester deadline without terminal evidence')
        time.sleep(.1)
    if (folder/'cancelled.json').exists():
        raise ProtocolError('RECOVERY_REQUIRED: tester operation cancelled')
    if not (folder/'response.json').exists():
        raise ProtocolError('UNKNOWN_OUTCOME: tester process lacks terminal evidence')
    return load_json(folder/'response.json',8192)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--execute')
    args=parser.parse_args();task=None
    try:
        if args.execute:
            execute(Path(args.root),load_json(args.execute,65536))
            return 0
        raw=sys.stdin.buffer.readline(65537)
        if len(raw)>65536 or not raw.endswith(b'\n'):
            raise ProtocolError('INVALID_INPUT: tester framing')
        request=json.loads(raw,object_pairs_hook=unique_object,parse_constant=reject_constant)
        task=request.get('task_id')
        response=dispatch(args.root,request)
    except Exception as error:
        response=envelope(task if isinstance(task,str) and len(task)<=64 else None,0,'ERROR')
        code=str(error).split(':',1)[0] if isinstance(error,ProtocolError) else 'RECOVERY_REQUIRED'
        response['error']=dict(code=code,message=code,retryable=False)
        response['cleanup_status']='UNVERIFIED'
    sys.stdout.buffer.write(encode(response)+b'\n')
    return 0 if response['status']!='ERROR' else 3


if __name__=='__main__':
    raise SystemExit(main())
