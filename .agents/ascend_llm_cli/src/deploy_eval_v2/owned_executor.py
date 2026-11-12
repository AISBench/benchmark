"""Frozen owned-service deployment/release executor for the stage runtime."""
from pathlib import Path
import threading
import traceback
from .compiler import verify_frozen,digest
from .deploy import execute_deployment
from .durable import atomic_json
from .json_io import load_json
from .protocol import ProtocolError,now
from .release import release_host_resources
from .owned_eval import owned_eval_inputs
from .evaluate import execute_external_case
from .process_guard import Cancelled
from .service_contract import validate_service
from .result_contract import evaluation_result


class OwnedExecutor:
    def __init__(self,root,frozen,bundle_digest,runtime_digest,nodes,*,deployer=execute_deployment,
                 evaluator=execute_external_case,verified_tester_host=None):
        self.root=Path(root)
        self.frozen=Path(frozen)
        self.bundle_digest=bundle_digest
        self.runtime_digest=runtime_digest
        self.nodes=nodes
        self.deployer=deployer
        self.evaluator=evaluator
        self.verified_tester_host=verified_tester_host
        values=self.inputs()
        self.task_id=values['plan.json']['task_id']
        self.idle_policy=values['plan.json']['policy']
        self.service_id='svc-'+digest([self.task_id,bundle_digest])[:32]
        slots={g['node'] for g in values['compiled.json']['topology']['groups'].values()}
        if set(nodes)!=slots or any(node.guard.owner['task_id']!=self.task_id for node in nodes.values()):
            raise ProtocolError('INVALID_INPUT: node ownership identity')
        if (self.root/'service.json').exists():
            self.service()

    def inputs(self):
        values=verify_frozen(self.frozen,self.bundle_digest,self.runtime_digest)
        if values['plan.json'].get('intent')!='deploy':
            raise ProtocolError('INVALID_INPUT: owned executor requires deploy plan')
        return values

    def service(self):
        value=load_json(self.root/'service.json')
        validate_service(value)
        values=self.inputs()
        compiled=values['compiled.json']
        if (value.get('service_id')!=self.service_id or value.get('task_id')!=self.task_id or
                value.get('bundle_sha256')!=self.bundle_digest or value.get('ownership')!='owned' or
                value['runtime_sha256']!=self.runtime_digest or
                value['policy']!=values['plan.json']['policy'] or
                value['endpoint']!=compiled['topology']['endpoint'] or value['model']!=compiled['model'] or
                set(value['containers'])!=set(compiled['topology']['groups']) or
                value.get('metrics',[])!=compiled['topology']['metric_instances']):
            raise ProtocolError('RECOVERY_REQUIRED: service identity mismatch')
        return value

    def validate(self,command):
        values=self.inputs()
        if command['task_id']!=self.task_id:
            raise ProtocolError('INVALID_INPUT: task identity mismatch')
        if command['kind']=='cancel':
            return
        if command['kind']=='deploy':
            if command['plan_ref']!=self.bundle_digest:
                raise ProtocolError('INVALID_INPUT: frozen plan_ref digest mismatch')
        elif command['kind'] in ('release','eval'):
            if command['service_id']!=self.service_id:
                raise ProtocolError('INVALID_INPUT: service identity mismatch')
            self.service()
            if command['kind']=='eval':
                plan,bound=owned_eval_inputs(values['plan.json'],values['bindings.json'],self.service())
                if (bound['testers'][plan['test_binding']]['host']!=self.verified_tester_host or
                        not any(c['case_id']==command['case_id'] for c in plan['cases'])):
                    raise ProtocolError('INVALID_INPUT: owned evaluation case or tester identity')
        else:
            raise ProtocolError('INVALID_INPUT: owned evaluation bindings not integrated')

    def __call__(self,command,operation,cancelled):
        self.validate(command)
        values=self.inputs()
        folder=self.root/'results'/operation
        folder.mkdir(parents=True,exist_ok=False)
        target=folder/'stage-result.json'
        reference=target.relative_to(self.root).as_posix()
        if command['kind']=='eval':
            plan,bound=owned_eval_inputs(values['plan.json'],values['bindings.json'],self.service())
            try:
                result=self.evaluator(plan,bound,command['case_id'],folder/'benchmark',cancelled,
                                      verified_local_host=self.verified_tester_host)
                result=evaluation_result(result,task_id=self.task_id,operation_id=operation,
                    service_id=self.service_id,case_id=command['case_id'])
                result['ownership']='owned'
                atomic_json(target,result)
                return 'evaluation.completed','EVALUATED',dict(service_id=self.service_id,case_id=command['case_id'],
                    result_ref=reference,result_sha256=digest(result),execution_status=result['execution_status'],
                    evidence_status=result['evidence_status'],performance_validity=result['performance_validity'])
            except Exception as error:
                with (folder/'evaluation-error.log').open('w',encoding='utf-8') as stream:
                    traceback.print_exc(file=stream)
                # Release is the frozen owning-task failure policy, independent of
                # the evaluation adapter's permissions and cancellation signal.
                from .remote_evaluation import RemoteCleanupUnknown
                if isinstance(error,RemoteCleanupUnknown):
                    cleanup=dict(cleanup_status='UNVERIFIED',reason='remote tester may still be active; service and claims retained')
                else:
                    cleanup=self._release(values,self.service(),operation,folder)
                result=dict(execution_status='CANCELLED' if isinstance(error,Cancelled) else 'FAILED',
                            cleanup_status=cleanup['cleanup_status'],cleanup=cleanup,
                            error_ref=(folder/'evaluation-error.log').relative_to(self.root).as_posix())
                result=evaluation_result(dict(result,evidence_status='PARTIAL',performance_validity='UNVERIFIED'),
                    task_id=self.task_id,operation_id=operation,service_id=self.service_id,case_id=command['case_id'])
                atomic_json(target,result)
                return ('operation.cancelled' if isinstance(error,Cancelled) else 'operation.failed',
                        'CANCELLED' if isinstance(error,Cancelled) else 'FAILED',
                        dict(service_id=self.service_id,evidence_ref=reference,cleanup_status=result['cleanup_status'],
                             error_code='CANCELLED' if isinstance(error,Cancelled) else 'EVALUATION_FAILED'))
        if command['kind']=='deploy':
            result=self.deployer(values['compiled.json'],self.nodes,values['plan.json']['execution'],target,cancelled)
            if result['execution_status']=='SUCCEEDED':
                service=dict(schema_version='1.0',task_id=self.task_id,service_id=self.service_id,
                    ownership='owned',endpoint=result['endpoint'],model=values['compiled.json']['model'],
                    bundle_sha256=self.bundle_digest,runtime_sha256=self.runtime_digest,
                    health_time=now(),health_evidence_ref=reference,policy=values['plan.json']['policy'],
                    deployment_summary=dict(container_groups=len(result['containers'])),
                    metrics=values['compiled.json']['topology']['metric_instances'],
                    containers=result['containers'])
                validate_service(service)
                atomic_json(self.root/'service.json',service)
                atomic_json(folder/'service.json',service)
                return 'deployment.ready','READY',dict(service_id=self.service_id,endpoint=service['endpoint'],
                    model=service['model'],evidence_ref=reference,result_sha256=digest(result),
                    service_ref=(folder/'service.json').relative_to(self.root).as_posix())
            stopped=result['execution_status']=='CANCELLED'
            return ('operation.cancelled' if stopped else 'operation.failed',
                    'CANCELLED' if stopped else 'FAILED',dict(evidence_ref=reference,
                    error_code='CANCELLED' if stopped else 'DEPLOYMENT_FAILED',phase=result.get('phase'),
                    cleanup_status=result['cleanup_status']))
        result=self._release(values,self.service(),operation,folder)
        atomic_json(target,result)
        verified=result['cleanup_status']=='VERIFIED'
        return ('release.completed' if verified else 'release.unverified','RELEASED' if verified else 'RECOVERY_REQUIRED',
                dict(service_id=self.service_id,resource_results_ref=reference,result_sha256=digest(result)))

    def _release(self,values,service,operation,folder):
        groups=values['compiled.json']['topology']['groups']
        results={}
        for name,node in self.nodes.items():
            keys=[k for k,g in groups.items() if g['node']==name]
            resources=sorted({r for k in keys for r in
                ['npu:'+str(n) for n in groups[k]['npu_ids']]+['tcp:'+str(p) for p in groups[k]['ports']]})
            try:
                containers=[service['containers'][k]['container_id'] for k in keys]
                if getattr(node,'remote',False) is True:
                    results[name]=node.release_resources('release-'+operation,containers)
                    continue
                results[name]=release_host_resources(node.guard,'release-'+operation,containers,resources,
                                                    folder/(name+'-resources.json'),runner=node.runner)
            except Exception:
                with (folder/(name+'-error.log')).open('w',encoding='utf-8') as stream:
                    traceback.print_exc(file=stream)
                results[name]=dict(cleanup_status='UNVERIFIED',claims_released=False)
        verified=all(r['cleanup_status']=='VERIFIED' for r in results.values())
        result=dict(schema_version='1.0',service_id=self.service_id,resources=results,
                    cleanup_status='VERIFIED' if verified else 'UNVERIFIED',time=now())
        return result

    def cancel_idle(self,operation):
        if not (self.root/'service.json').exists():
            return 'operation.cancelled','CANCELLED',dict(cleanup_status='VERIFIED')
        kind,status,data=self(dict(task_id=self.task_id,kind='release',service_id=self.service_id),
                              operation,threading.Event())
        verified=kind=='release.completed'
        data['cleanup_status']='VERIFIED' if verified else 'UNVERIFIED'
        return 'operation.cancelled','CANCELLED' if verified else 'RECOVERY_REQUIRED',data
