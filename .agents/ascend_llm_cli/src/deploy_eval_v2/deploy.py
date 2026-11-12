"""Deployment phase ordering across node adapters, with independent failure cleanup."""
import time
from pathlib import Path
import traceback
from .task_log import DiagnosticStream, task_root, record_exception
from .compiler import digest
from .durable import atomic_json
from .process_guard import remaining, Cancelled
from .release import release_host_resources
from .protocol import ProtocolError


def execute_deployment(compiled, nodes, execution, evidence_path, cancelled, *, health=None, smoke=None):
    """Retry only confirmed failures after verified cleanup, within one deadline."""
    target=Path(evidence_path)
    # Standalone adapter tests also get one root, not a log per retry.
    root=task_root(target)
    (root/'.runtime').mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+execution['startup_timeout_seconds']
    attempts=[]
    for attempt in range(execution.get('max_retries',0)+1):
        path=target.parent/'attempts'/str(attempt)/target.name
        path.parent.mkdir(parents=True,exist_ok=True)
        result=_execute_attempt(compiled,nodes,execution,path,cancelled,health=health,smoke=smoke,
                                deadline=deadline,attempt=attempt)
        attempts.append(dict(attempt=attempt,evidence_ref=path.relative_to(target.parent).as_posix(),
                             execution_status=result['execution_status'],cleanup_status=result['cleanup_status']))
        if (result['execution_status']!='FAILED' or result['cleanup_status']!='VERIFIED' or
                not result.get('retryable',False) or cancelled.is_set() or time.monotonic()>=deadline):
            break
    result=dict(result,attempts=attempts)

    atomic_json(target,result)
    return result


def _execute_attempt(compiled, nodes, execution, evidence_path, cancelled, *, health=None, smoke=None,deadline,attempt):
    """Run frozen topology. nodes are local/SSH adapters keyed by bound node.

    Health/smoke adapters receive the single phase deadline and cancellation event.
    Cleanup deliberately uses its own budget and ignores the cancelled work flag.
    No failure is reported as verified cleanup when create outcome is uncertain.
    """
    topology, scripts = compiled['topology'], compiled['scripts']
    from .http_probe import wait_health, smoke_completion
    health = health or wait_health
    smoke = smoke or (lambda endpoint, deadline, flag: smoke_completion(endpoint,compiled['model'],deadline,flag))
    groups = topology['groups']
    created, attempted, phase = {}, set(), 'claim'
    evidence_path = Path(evidence_path)
    def command(group, kind, instance=''):
        return kind+'-'+digest([attempt,group,instance])[:32]
    def check():
        remaining(deadline)
        if cancelled.is_set():
            raise Cancelled('Deployment cancelled')
    try:
        for key, group in groups.items():
            check()
            node = nodes[group['node']]
            if getattr(node,'remote',False) is True:
                node.prepare(group,scripts[key],deadline,cancelled)
                continue
            node.guard.claim(['npu:'+str(n) for n in group['npu_ids']]+['tcp:'+str(p) for p in group['ports']])
            node.install_scripts(group, scripts[key])
        phase = 'create'
        for key, group in groups.items():
            check()
            attempted.add(key)
            created[key] = nodes[group['node']].create(command(key,'create'),group,deadline,cancelled)
        phase = 'container_start'
        for key, result in created.items():
            check()
            nodes[groups[key]['node']].start_container(command(key,'start'),result['container_id'],deadline,cancelled)
        phase = 'dependencies'
        dependency_deadline = min(deadline,time.monotonic()+execution['dependency_timeout_seconds'])
        for key, group in groups.items():
            for script in scripts[key]:
                if script['purpose'] == 'dependency':
                    check()
                    nodes[group['node']].check_dependency(command(key,'dependency',script['instance']),
                        created[key]['container_id'],script,dependency_deadline,cancelled)
        phase = 'services'
        for role in ('models','proxies'):
            for key, group in groups.items():
                for item in group['commands']:
                    if (item['role']=='proxy') != (role=='proxies'):
                        continue
                    check()
                    if role=='proxies' and execution['proxy_start_after_models_healthy']:
                        health(item['startup_health_urls'],deadline,cancelled)
                    script = next(s for s in scripts[key] if s['instance']==item['name'] and s['purpose']=='service')
                    nodes[group['node']].start_service(command(key,'service',item['name']),
                        created[key]['container_id'],script,deadline,cancelled)
        phase = 'health'
        health_evidence = health([url for group in groups.values() for url in group['health_urls']],deadline,cancelled)
        phase = 'smoke'
        smoke_evidence = smoke(topology['endpoint'],min(deadline,time.monotonic()+execution['smoke_timeout_seconds']),cancelled)
        report = dict(schema_version='1.0',execution_status='SUCCEEDED',containers=created,
                      endpoint=topology['endpoint'],health=health_evidence,smoke=smoke_evidence,cleanup_status='NOT_REQUESTED')
        atomic_json(evidence_path,report)
        return report
    except Exception as error:
        evidence_path.parent.mkdir(parents=True,exist_ok=True)
        with DiagnosticStream(evidence_path.with_suffix('.error.log')) as stream:
            traceback.print_exc(file=stream)
        cleanup = {}
        # Aggregate a physical host before unlocking any of its resource claims.
        for name,node in nodes.items():
            keys = [k for k,g in groups.items() if g['node']==name]
            known = [created[k]['container_id'] for k in keys if k in created]
            uncertain = any(k in attempted and k not in created for k in keys)
            resources = sorted({r for k in keys for r in
                ['npu:'+str(n) for n in groups[k]['npu_ids']]+['tcp:'+str(p) for p in groups[k]['ports']]})
            try:
                if known:
                    if getattr(node,'remote',False) is True:
                        cleanup[name]=node.release_resources(command(name,'cleanup'),known,keep_claims=uncertain)
                        continue
                    # Unknown create may have consumed the same claims: clean known
                    # containers but never unlock this host's entire inventory.
                    if uncertain:
                        from .release import release_container
                        for cid in known:
                            release_container(cid,node.guard.owner['task_id'],node.guard.owner['token'],
                                              node.runner,deadline=time.monotonic()+30)
                        cleanup[name] = dict(cleanup_status='UNVERIFIED',claims_released=False)
                    else:
                        cleanup[name] = release_host_resources(node.guard,command(name,'cleanup'),known,resources,
                            evidence_path.parent/(name+'-cleanup.json'),runner=node.runner)
                elif not uncertain:
                    cleanup[name]=node.guard.release_unstarted(resources)
                else:
                    cleanup[name] = dict(cleanup_status='UNVERIFIED',claims_released=False)
            except Exception:
                cleanup[name] = dict(cleanup_status='UNVERIFIED',claims_released=False)
        report = dict(schema_version='1.0',execution_status='CANCELLED' if isinstance(error,Cancelled) else 'FAILED',
                      phase=phase,containers=created,cleanup=cleanup,
                      error_ref='task.jsonl',
                      cleanup_status='VERIFIED' if all(r['cleanup_status']=='VERIFIED' for r in cleanup.values()) else 'UNVERIFIED')
        error_code=str(error).split(':',1)[0] if isinstance(error,ProtocolError) else type(error).__name__
        report.update(error_code=error_code,retryable=phase in ('dependencies','health','smoke') and
                      error_code in ('INVALID_INPUT','TimeoutError') and report['cleanup_status']=='VERIFIED')
        atomic_json(evidence_path,report)
        return report
