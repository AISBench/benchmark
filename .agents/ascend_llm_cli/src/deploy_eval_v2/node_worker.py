"""One SSH node command, constrained to a verified frozen task and runtime.

This process is not another task coordinator. Effects use the host guard and
owned containers; interrupted replies never authorize re-execution.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import threading
import time
from .compiler import verify_frozen
from .docker_node import DockerNode
from .durable import atomic_json
from .json_io import unique_object,reject_constant,load_json
from .protocol import ProtocolError,encode,envelope
from .publication import verify_runtime
from .remote_guard import HostGuard,exclusive
from .release import release_host_resources,release_container

GUARD_ROOT='/var/lib/deploy-eval-v2/guard'


def dispatch(root,request,*,guard_root=GUARD_ROOT,node_factory=DockerNode):
    common={'protocol_version','task_id','node','action','command_id','runtime_sha256','bundle_sha256','token','timeout_seconds'}
    fields={'claim':{'group'},'install':{'group'},'create':{'group'},'start':{'container_id'},
            'dependency':{'group','instance','container_id'},'service':{'group','instance','container_id'},
            'release':{'containers','keep_claims'},'release_unstarted':set(),'log-page':{'log_offset'},'log-ack':{'log_sha256'}}
    if (not isinstance(request,dict) or request.get('action') not in fields or
            set(request)!=common|fields[request['action']] or request['protocol_version']!='1.0'):
        raise ProtocolError('INVALID_INPUT: node request fields')
    if any(not isinstance(request[k],str) or not re.fullmatch('[a-f0-9]{64}',request[k])
           for k in ('runtime_sha256','bundle_sha256','token')):
        raise ProtocolError('INVALID_INPUT: node digests')
    if not isinstance(request['command_id'],str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}',request['command_id']):
        raise ProtocolError('INVALID_INPUT: node command identity')
    timeout=request['timeout_seconds']
    if type(timeout) not in (int,float) or not 0<timeout<=86400:
        raise ProtocolError('INVALID_INPUT: node deadline')
    root=Path(root)
    verify_runtime(root/'runtime',request['runtime_sha256'])
    values=verify_frozen(root/'inputs',request['bundle_sha256'],request['runtime_sha256'])
    if values['plan.json']['task_id']!=request['task_id'] or values['plan.json']['intent']!='deploy':
        raise ProtocolError('INVALID_INPUT: node task identity')
    compiled=values['compiled.json']
    groups={key:g for key,g in compiled['topology']['groups'].items() if g['node']==request['node']}
    if not groups or any(Path(g['root']).resolve()!=root.resolve() for g in groups.values()):
        raise ProtocolError('INVALID_INPUT: node root binding mismatch')
    if node_factory is DockerNode:
        from .executor_factory import local_machine_id
        if values['bindings.json']['nodes'][request['node']].get('machine_id')!=local_machine_id():
            raise ProtocolError('INVALID_INPUT: remote node machine identity mismatch')
    identity={key:request[key] for key in ('task_id','node','runtime_sha256','bundle_sha256','token')}
    with exclusive(root/'.runtime/node-identity.lock'):
        identity_path=root/'.runtime/node-identity.json'
        if identity_path.exists():
            if load_json(identity_path)!=identity:
                raise ProtocolError('RECOVERY_REQUIRED: node execution identity changed')
        else:
            atomic_json(identity_path,identity)
    if request['action'] in ('log-page','log-ack'):
        from .log_transfer import serve
        response=envelope(request['task_id'],0)
        response.update(command_id=request['command_id'],result=serve(root,request['action'],
            offset=request.get('log_offset',0),sha256=request.get('log_sha256')))
        return response
    selected=None
    if 'group' in request:
        selected=groups.get(request['group'])
        if selected is None:
            raise ProtocolError('INVALID_INPUT: group not on node')
    guard=HostGuard(guard_root,request['task_id'],request['token'])
    node=node_factory(guard)
    deadline=time.monotonic()+timeout
    cancelled=threading.Event()
    action=request['action']
    command=request['command_id']
    if action=='release_unstarted':
        resources=sorted({r for g in groups.values() for r in
            ['npu:'+str(n) for n in g['npu_ids']]+['tcp:'+str(p) for p in g['ports']]})
        result=guard.release_unstarted(resources)
    elif action=='claim':
        resources=['npu:'+str(n) for n in selected['npu_ids']]+['tcp:'+str(p) for p in selected['ports']]
        guard.claim(resources)
        result=dict(claimed=True)
    elif action=='install':
        node.install_scripts(selected,compiled['scripts'][request['group']])
        result=dict(installed=True)
    elif action=='create':
        result=node.create(command,selected,deadline,cancelled)
    elif action=='start':
        result=node.start_container(command,request['container_id'],deadline,cancelled)
    elif action in ('dependency','service'):
        matches=[s for s in compiled['scripts'][request['group']]
                 if s['instance']==request['instance'] and s['purpose']==action]
        if len(matches)!=1:
            raise ProtocolError('INVALID_INPUT: script identity')
        method=node.check_dependency if action=='dependency' else node.start_service
        result=method(command,request['container_id'],matches[0],deadline,cancelled)
    else:
        containers=request['containers']
        if (not isinstance(containers,list) or not containers or len(containers)>128 or
                any(not isinstance(cid,str) or not re.fullmatch('[a-f0-9]{64}',cid) for cid in containers) or
                type(request['keep_claims']) is not bool):
            raise ProtocolError('INVALID_INPUT: node release inventory')
        resources=sorted({r for g in groups.values() for r in
            ['npu:'+str(n) for n in g['npu_ids']]+['tcp:'+str(p) for p in g['ports']]})
        evidence=root/'results'/'node'/(command+'.json')
        if request['keep_claims']:
            def clean_known():
                entries=[release_container(cid,request['task_id'],request['token'],node.runner,deadline=deadline)
                         for cid in containers]
                atomic_json(evidence,dict(containers=entries,claims_released=False))
                return dict(cleanup_status='UNVERIFIED',claims_released=False)
            result=guard.command(command,dict(containers=containers,keep_claims=True),clean_known)
        else:
            result=release_host_resources(guard,command,containers,resources,evidence,
                                          timeout=min(timeout,3600),runner=node.runner)
        result['evidence']=dict(host=next(iter(groups.values()))['host'],path=str(evidence))
    response=envelope(request['task_id'],0)
    response.update(command_id=command,result=result)
    if len(encode(response))>8192:
        raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: node response budget')
    return response


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    args=parser.parse_args()
    task_id=None
    try:
        raw=sys.stdin.buffer.readline(65537)
        if len(raw)>65536 or not raw.endswith(b'\n'):
            raise ProtocolError('INVALID_INPUT: node request budget')
        request=json.loads(raw,object_pairs_hook=unique_object,parse_constant=reject_constant)
        if isinstance(request,dict) and isinstance(request.get('task_id'),str) and len(request['task_id'])<=64:
            task_id=request['task_id']
        response=dispatch(args.root,request)
        code=0
    except Exception as error:
        message=str(error).split(':',1)[0] if isinstance(error,ProtocolError) else 'RECOVERY_REQUIRED'
        response=envelope(task_id,0,'ERROR')
        response['error']=dict(code=message,message=message,retryable=False)
        code=3 if message in ('UNKNOWN_OUTCOME','TRANSPORT_PROTOCOL_ERROR') else 2
    sys.stdout.buffer.write(encode(response)+b'\n')
    return code


if __name__=='__main__':
    raise SystemExit(main())
