"""Task-runtime adapter to the frozen node worker over bounded SSH JSON."""
import threading
import time
from .compiler import digest
from .protocol import ProtocolError
from .process_guard import remaining,Cancelled
from .ssh_transport import exchange,ssh_argv


class SSHNode:
    remote=True

    def __init__(self,task_id,node,compiled,bundle_digest,runtime_digest,token,python='/usr/bin/python3',transport=exchange,coordinator_root=None):
        self.task_id,self.node=task_id,node
        self.groups={k:g for k,g in compiled['topology']['groups'].items() if g['node']==node}
        if not self.groups:
            raise ProtocolError('INVALID_INPUT: missing SSH node groups')
        group=next(iter(self.groups.values()))
        self.root,self.host=group['root'],group['host']
        self.bundle_digest,self.runtime_digest,self.token=bundle_digest,runtime_digest,token
        self.transport=transport
        self.coordinator_root=coordinator_root
        # No shell config/body interpolation; ssh_argv quotes each argument.
        bootstrap='import sys; sys.path.insert(0, '+repr(self.root+'/runtime')+'); from deploy_eval_v2.node_worker import main; raise SystemExit(main())'
        self.argv=ssh_argv(self.host,[python,'-B','-c',bootstrap,'--root',self.root])
        self.guard=_RemoteGuard(self)

    def _call(self,action,command_id,deadline,cancelled,**fields):
        if cancelled.is_set():
            raise Cancelled('Remote node operation cancelled')
        seconds=remaining(deadline)
        request=dict(protocol_version='1.0',task_id=self.task_id,node=self.node,action=action,
            command_id=command_id,runtime_sha256=self.runtime_digest,bundle_sha256=self.bundle_digest,
            token=self.token,timeout_seconds=seconds,**fields)
        try:
            response=self.transport(self.argv,request,timeout=seconds,cancelled=cancelled)
        finally:
            if self.coordinator_root is not None and action not in ('log-page','log-ack'):
                from .log_transfer import collect
                from .task_log import diagnostic
                log_deadline=time.monotonic()+15
                try:
                    collect(self.coordinator_root,self.host+':'+self.root,
                        lambda kind,**extra:self._call(kind,command_id,log_deadline,threading.Event(),**extra))
                except Exception as error:
                    diagnostic(__import__('pathlib').Path(self.coordinator_root)/'task.jsonl',
                               'Remote log retained; collection pending',event='remote.log.pending',
                               remote_source=self.host+':'+self.root,error_type=type(error).__name__)

        if response['status']=='ERROR':
            raise ProtocolError(response['error']['code'])
        if response.get('command_id')!=command_id or not isinstance(response.get('result'),dict):
            raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: node response identity')
        return response['result']

    def _group(self,group):
        matches=[key for key,value in self.groups.items() if value==group]
        if len(matches)!=1:
            raise ProtocolError('INVALID_INPUT: node group mismatch')
        return matches[0]

    def install_scripts(self,group,scripts):
        key=self._group(group)
        return self._call('install','install-'+digest(key)[:32],time.monotonic()+30,threading.Event(),group=key)

    def prepare(self,group,scripts,deadline,cancelled):
        key=self._group(group)
        self._call('claim','claim-'+digest(key)[:32],deadline,cancelled,group=key)
        return self._call('install','install-'+digest(key)[:32],deadline,cancelled,group=key)

    def create(self,command_id,group,deadline,cancelled):
        return self._call('create',command_id,deadline,cancelled,group=self._group(group))

    def start_container(self,command_id,cid,deadline,cancelled):
        return self._call('start',command_id,deadline,cancelled,container_id=cid)

    def _script(self,action,command_id,cid,script,deadline,cancelled):
        matches=[key for key,g in self.groups.items() if any(c['name']==script['instance'] for c in g['commands'])]
        if len(matches)!=1:
            raise ProtocolError('INVALID_INPUT: remote script identity')
        return self._call(action,command_id,deadline,cancelled,group=matches[0],instance=script['instance'],container_id=cid)

    def check_dependency(self,command_id,cid,script,deadline,cancelled):
        return self._script('dependency',command_id,cid,script,deadline,cancelled)

    def start_service(self,command_id,cid,script,deadline,cancelled):
        return self._script('service',command_id,cid,script,deadline,cancelled)

    def release_resources(self,command_id,containers,*,keep_claims=False):
        return self._call('release',command_id,time.monotonic()+120,threading.Event(),
                          containers=containers,keep_claims=keep_claims)


class _RemoteGuard:
    def __init__(self,node):
        self.node=node
        self.owner=dict(task_id=node.task_id,token=node.token)

    def claim(self,resources):
        matches=[key for key,g in self.node.groups.items() if set(resources)==
                 set(['npu:'+str(n) for n in g['npu_ids']]+['tcp:'+str(p) for p in g['ports']])]
        if len(matches)!=1:
            raise ProtocolError('INVALID_INPUT: remote claims differ from frozen group')
        return self.node._call('claim','claim-'+digest(matches[0])[:32],time.monotonic()+30,
                               threading.Event(),group=matches[0])

    def release_claims(self,resources,*,resources_verified_absent):
        raise ProtocolError('RECOVERY_REQUIRED: remote claims require node inventory verification')

    def release_unstarted(self,resources):
        expected={r for g in self.node.groups.values() for r in
            ['npu:'+str(n) for n in g['npu_ids']]+['tcp:'+str(p) for p in g['ports']]}
        if set(resources)!=expected:
            raise ProtocolError('INVALID_INPUT: node cleanup claim inventory')
        return self.node._call('release_unstarted','release-unstarted',time.monotonic()+30,threading.Event())
