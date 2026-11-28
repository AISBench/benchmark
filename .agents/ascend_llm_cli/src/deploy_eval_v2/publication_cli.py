"""JSON-file CLI contracts for runtime packaging and task publication."""
from pathlib import Path,PurePosixPath
import re
import tempfile
from .compiler import verify_frozen
from .input_publication import build_input_archive
from .protocol import ProtocolError,envelope
from .publication import build_archive,install_archive
from .ssh_publication import publish_task
from .ssh_transport import exchange,ssh_argv


def task_binding(config,task):
    if not isinstance(task,str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}',task):
        raise ProtocolError('INVALID_INPUT: task identity')
    binding=config['tasks'][task]
    root,allowed=PurePosixPath(binding['root']),PurePosixPath(binding['allowed_root'])
    if (not allowed.is_absolute() or '..' in allowed.parts or not root.is_absolute() or
            '..' in root.parts or root==allowed or not root.is_relative_to(allowed) or
            any(c in str(root) for c in '\x00\r\n\\')):
        raise ProtocolError('INVALID_INPUT: task root')
    return binding


def package_runtime(request):
    if not isinstance(request,dict) or set(request)!={'destination'} or not isinstance(request['destination'],str) or not 1<=len(request['destination'])<=512:
        raise ProtocolError('INVALID_INPUT: package runtime request')
    result=build_archive(Path(__file__).parent,request['destination'])
    response=envelope(None,0)
    response['runtime_package']=result
    return response


def publication_targets(values,binding):
    targets=[('runtime',binding)]
    if values['plan.json']['intent']!='deploy':
        return targets
    groups=values['compiled.json']['topology']['groups']
    selected={}
    for group in groups.values():
        slot=group['node']
        if slot in selected:
            if (selected[slot]['host'],selected[slot]['root'])!=(group['host'],group['root']):
                raise ProtocolError('INVALID_INPUT: inconsistent node publication root')
            continue
        selected[slot]=group
        if (group['host'],group['root'])==(binding['host'],binding['root']):
            continue
        target=binding.get('nodes',{}).get(slot)
        if not isinstance(target,dict) or (target.get('host'),target.get('root'))!=(group['host'],group['root']):
            raise ProtocolError('INVALID_INPUT: node publication target differs from frozen binding')
        task_binding({'tasks':{'target':target}},'target')
        targets.append((slot,target))
    evaluation=values['plan.json'].get('evaluation')
    if evaluation:
        tester=values['bindings.json']['testers'][evaluation['test_binding']]
        test_root=str(PurePosixPath(tester['work_root'])/values['plan.json']['task_id'])
        if (tester['host'],test_root) not in {(item['host'],item['root']) for _,item in targets}:
            target=binding.get('tester')
            if not isinstance(target,dict) or (target.get('host'),target.get('root'))!=(tester['host'],test_root):
                raise ProtocolError('INVALID_INPUT: tester publication target differs from frozen binding')
            task_binding({'tasks':{'tester':target}},'tester')
            targets.append(('tester',target))
    return targets


def publish(request,config,*,publisher=publish_task):
    required={'task_ref','runtime_archive','input_directory','runtime_sha256','bundle_sha256'}
    if not isinstance(request,dict) or set(request)!=required or any(not isinstance(v,str) or not 1<=len(v)<=512 for v in request.values()):
        raise ProtocolError('INVALID_INPUT: publication request')
    for key in ('runtime_sha256','bundle_sha256'):
        if not re.fullmatch('[a-f0-9]{64}',request[key]):
            raise ProtocolError('INVALID_INPUT: publication digest')
    task=request['task_ref']
    binding=task_binding(config,task)
    values=verify_frozen(request['input_directory'],request['bundle_sha256'],request['runtime_sha256'])
    if values['plan.json'].get('task_id')!=task:
        raise ProtocolError('INVALID_INPUT: publication task mismatch')
    if values['plan.json'].get('intent') not in ('deploy','external_eval'):
        raise ProtocolError('INVALID_INPUT: unsupported production plan')
    targets=publication_targets(values,binding)
    # Inspect both local payloads before any remote side effect or connection.
    with tempfile.TemporaryDirectory(prefix='deploy-eval-publish-') as temporary:
        temporary=Path(temporary)
        install_archive(request['runtime_archive'],temporary/'runtime',request['runtime_sha256'])
        inputs=temporary/'inputs.zip'
        build_input_archive(request['input_directory'],inputs,request['bundle_sha256'],request['runtime_sha256'])
        completed=0
        for name,target in targets:
            try:
                publisher(target['host'],target.get('python','/usr/bin/python3'),target['root'],
                          request['runtime_archive'],inputs,task,request['runtime_sha256'],request['bundle_sha256'])
                completed+=1
            except (ProtocolError,OSError):
                response=envelope(task,0,'ERROR')
                response.update(publication=dict(completed_targets=completed,total_targets=len(targets),failed_target=name),started=False)
                response['error']=dict(code='UNKNOWN_OUTCOME',message='Publication incomplete; verify existing payloads before retry',retryable=False)
                return response
    response=envelope(task,0)
    response.update(publication=dict(completed_targets=completed,total_targets=len(targets),
                                    runtime_sha256=request['runtime_sha256'],bundle_sha256=request['bundle_sha256']),started=False)
    return response


def start_task(request,config,*,transport=exchange):
    fields={'task_ref','runtime_sha256','bundle_sha256','nonce'}
    if not isinstance(request,dict) or set(request)!=fields:
        raise ProtocolError('INVALID_INPUT: start-task request')
    for key in fields-{'task_ref'}:
        if not isinstance(request[key],str) or not re.fullmatch('[a-f0-9]{64}',request[key]):
            raise ProtocolError('INVALID_INPUT: startup digest')
    binding=task_binding(config,request['task_ref'])
    root=binding['root']
    python=binding.get('python','/usr/bin/python3')
    if not isinstance(python,str) or not python.startswith('/'):
        raise ProtocolError('INVALID_INPUT: absolute startup Python required')
    bootstrap='import sys; sys.path.insert(0, '+repr(root+'/runtime')+'); from deploy_eval_v2.startup import main; raise SystemExit(main())'
    remote=dict(protocol_version='1.0',task_id=request['task_ref'],python=python,
                **{k:request[k] for k in fields-{'task_ref'}})
    return transport(ssh_argv(binding['host'],[python,'-B','-c',bootstrap,'--root',root]),remote,timeout=90)
