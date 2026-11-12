"""Offline readiness checks. Never connect to hosts or start workloads."""
import re
import shutil
import tempfile
from pathlib import Path
from .compiler import verify_frozen
from .publication import install_archive
from .publication_cli import task_binding,publication_targets
from .protocol import ProtocolError,envelope


def check_ready(request,config):
    fields={'task_ref','runtime_archive','input_directory','runtime_sha256','bundle_sha256'}
    if not isinstance(request,dict) or set(request)!=fields:
        raise ProtocolError('INVALID_INPUT: readiness request')
    task=request['task_ref']
    binding=task_binding(config,task)
    values=verify_frozen(request['input_directory'],request['bundle_sha256'],request['runtime_sha256'])
    plan,bound=values['plan.json'],values['bindings.json']
    if plan['task_id']!=task:
        raise ProtocolError('INVALID_INPUT: readiness task mismatch')
    targets=publication_targets(values,binding)
    with tempfile.TemporaryDirectory(prefix='v2-readiness-') as folder:
        install_archive(request['runtime_archive'],Path(folder)/'runtime',request['runtime_sha256'])
    blockers=[]
    def require(condition,code):
        if not condition and code not in blockers:
            blockers.append(code)
    require(shutil.which('ssh') is not None,'CONTROL_SSH_MISSING')
    for _,target in targets:
        python=target.get('python','/usr/bin/python3')
        require(isinstance(python,str) and python.startswith('/'),'ABSOLUTE_REMOTE_PYTHON_REQUIRED')
    def machine(item):
        value=item.get('machine_id','')
        return value if isinstance(value,str) and re.fullmatch('[a-f0-9]{32}',value) else None
    evaluation=None
    if plan['intent']=='external_eval':
        evaluation=plan
        tester=bound['testers'][plan['test_binding']]
        require(machine(tester) is not None,'TESTER_MACHINE_ID_REQUIRED')
        require(tester['host']==binding['host'],'RUNTIME_MUST_USE_TESTER_HOST')
    elif plan['intent']=='deploy':
        groups=values['compiled.json']['topology']['groups']
        slots={g['node'] for g in groups.values()}
        local={g['node'] for g in groups.values() if (g['host'],g['root'])==(binding['host'],binding['root'])}
        require(len(local)==1,'RUNTIME_NODE_BINDING_REQUIRED')
        identities=[machine(bound['nodes'][slot]) for slot in slots]
        require(all(identities),'NODE_MACHINE_ID_REQUIRED')
        require(len(set(identities))==len(identities),'NODE_MACHINE_IDS_MUST_BE_DISTINCT')
        evaluation=plan.get('evaluation')
        if evaluation:
            tester=bound['testers'][evaluation['test_binding']]
            require(machine(tester) is not None,'TESTER_MACHINE_ID_REQUIRED')
    else:
        require(False,'INTENT_UNSUPPORTED')
    result=envelope(task,0,'ERROR' if blockers else 'EVENTS')
    result.update(readiness=dict(status='BLOCKED' if blockers else 'READY_FOR_HOST_PREFLIGHT',
        blockers=blockers,publication_targets=len(targets),local_payloads_verified=True,
        remote_checks='NOT_RUN',performance_acceptance='NOT_RUN'),started=False)
    if blockers:
        result['error']=dict(code='INVALID_INPUT',message='Readiness blockers; see readiness.blockers',retryable=False)
    return result
