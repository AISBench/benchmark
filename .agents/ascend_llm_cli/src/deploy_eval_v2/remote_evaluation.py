"""Owned-task remote tester adapter. Task coordinator remains on its main node."""
from pathlib import PurePosixPath,Path
import threading
from .hashes import digest
from .durable import atomic_json
from .protocol import ProtocolError
from .process_guard import Cancelled
from .ssh_transport import exchange,ssh_argv


class RemoteEvaluation:
    def __init__(self,task_id,tester,bundle_digest,runtime_digest,token,*,transport=exchange):
        self.task_id,self.tester=task_id,tester
        self.bundle_digest,self.runtime_digest,self.token=bundle_digest,runtime_digest,token
        self.root=str(PurePosixPath(tester['work_root'])/task_id)
        bootstrap='import sys; sys.path.insert(0, '+repr(self.root+'/runtime')+'); from deploy_eval_v2.tester_worker import main; raise SystemExit(main())'
        self.argv=ssh_argv(tester['host'],[tester['python'],'-B','-c',bootstrap,'--root',self.root])
        self.transport=transport

    def __call__(self,plan,bindings,case_id,folder,cancelled,*,verified_local_host):
        if verified_local_host!=self.tester['host'] or bindings['testers'][plan['test_binding']]!=self.tester:
            raise ProtocolError('INVALID_INPUT: remote tester binding mismatch')
        operation=Path(folder).parent.name
        case=next(c for c in plan['cases'] if c['case_id']==case_id)
        request=dict(protocol_version='1.0',task_id=self.task_id,operation_id=operation,case_id=case_id,
            runtime_sha256=self.runtime_digest,bundle_sha256=self.bundle_digest,token=self.token,action='run')
        try:
            response=self.transport(self.argv,request,timeout=min(86400,case['timeout_seconds']+30),cancelled=cancelled)
            if response.get('status')=='ERROR':
                raise ProtocolError(response['error']['code'])
            result=response.get('result')
            if (response.get('operation_id')!=operation or not isinstance(result,dict) or
                    response.get('result_sha256')!=digest(result)):
                raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: tester response identity')
        except Exception as error:
            # An interrupted observer is not proof the remote client stopped.
            # Explicitly stop the exactly identified unit before service cleanup.
            try:
                stopped=self.transport(self.argv,dict(request,action='cancel'),timeout=30)
                if stopped.get('status')!='EVENTS' or stopped.get('cleanup_status')!='VERIFIED':
                    raise ProtocolError('RECOVERY_REQUIRED: remote tester stop unverified')
            except Exception as cleanup_error:
                raise RemoteCleanupUnknown('Remote tester cleanup unverified') from cleanup_error
            if cancelled.is_set():
                raise Cancelled('Remote evaluation cancelled after verified client stop') from error
            raise
        Path(folder).mkdir(parents=True,exist_ok=False)
        atomic_json(Path(folder)/'remote-result.json',response)
        result=dict(result,remote_evidence=response.get('evidence'))
        return result


class RemoteCleanupUnknown(RuntimeError):
    """Prevents releasing/reusing a service while an unknown remote load may live."""
