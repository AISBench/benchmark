"""Serial topology coordinator; node operations use owned local lifecycle scripts."""
import copy
import json
from pathlib import Path
import shlex
import re
import subprocess
import time
from stage_runtime import save, digest, http, command
from remote_node import RemoteNode, PortConflict, transport_argv
from topology import compile_deployment, context_limit
from task_events import emit
from node_identity import node_job_id, node_root as physical_node_root


def prepare_call(binding, request):
    if not re.fullmatch(r'(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9_.-]*',binding['host']):
        raise ValueError('Invalid SSH destination')
    if not Path(binding['package']).is_absolute():
        raise ValueError('Absolute node package path required')
    script = str(Path(binding['package'])/'scripts/node_prepare.py')
    result = subprocess.run(transport_argv(binding['host'], ['python3', '-B', script]), input=json.dumps(request), capture_output=True, text=True, timeout=240)
    try:
        response = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise RuntimeError('Preparation transport failed: '+result.stderr[-2000:])
    if result.returncode or response['status'] != 'PASSED':
        raise RuntimeError('Node preparation failed: '+json.dumps(response))
    return response['result']


def container_pools(spec, bindings):
    """One compatible task container per host/runtime; union known mounts/devices."""
    pools, assignments, host_counts = {}, {}, {}
    for dep in spec['deployments']:
        for slot in sorted({i['placement']['node'] for i in dep['instances']}):
            binding=bindings['nodes'][slot]
            key=(binding['host'],digest({'node_id':slot,'runtime':dep['runtime'],'work_dir':binding['work_dir'],
                'mounts':binding.get('mounts',[])}))
            if key not in pools:
                number=host_counts.get(binding['host'],0)+1;host_counts[binding['host']]=number
                name=node_job_id(spec['job_id'],slot)+(f'-{number:03}' if number>1 else '')
                pools[key]={'name':name,'job_id':node_job_id(spec['job_id'],slot),'parent_job_id':spec['job_id'],'node_id':slot,'node_root':str(physical_node_root(binding,spec['job_id'],slot)),'task_root':str(Path(binding['work_dir'])/spec['job_id']),
                    'devices':set(),'model_paths':set(),'plan_sha256':digest(spec)}
            pool=pools[key]
            pool['devices'].update(d for i in dep['instances'] if i['placement']['node']==slot for d in i['placement']['devices'])
            pool['model_paths'].add(binding['model_paths'][dep['model']['id']])
            assignments[(dep['id'],slot)]=pool
    for pool in pools.values():
        pool['devices']=sorted(pool['devices']);pool['model_paths']=sorted(pool['model_paths'])
    return assignments


def preflight(root, spec, bindings, *, deployment_id=None, generation=None):
    pools=container_pools(spec,bindings)
    result = {'plan_sha256': digest(spec), 'bindings_sha256': digest(bindings), 'deployments': {}}
    if generation is not None:
        result = json.loads((root/'resolved-cluster.json').read_text())
    failures=[]
    for dep_index, dep in enumerate(spec['deployments'], 1):
        if deployment_id is not None and dep['id'] != deployment_id:
            continue
        used = sorted({i['placement']['node'] for i in dep['instances']})
        nodes = {}
        for slot in used:
            binding = bindings['nodes'][slot]
            request_ports = {i['id']+'.'+p: width for i in dep['instances'] if i['placement']['node'] == slot
                             for p, width in i['ports'].items()}
            node_root = str(physical_node_root(binding,spec['job_id'],slot)/'.internal'/dep['id'])
            if generation is not None:
                node_root = str(Path(node_root)/'port-generations'/generation)
            try:
                discovered = prepare_call(binding, {'action': 'discover', 'run_dir': node_root,
                    'job_id': node_job_id(spec['job_id'],slot)+f'-{dep_index:03}', 'image': dep['runtime']['image'],
                    'bindings': binding, 'port_requests': request_ports})
                nodes[slot] = {**discovered, 'root': node_root, 'ports': {i['id']:
                    {name: discovered['port_bases'][i['id']+'.'+name] for name in i['ports']}
                    for i in dep['instances'] if i['placement']['node'] == slot}}
            except Exception as exc:
                failures.append({'deployment':dep['id'],'node':slot,'error':repr(exc)})
        if len(nodes)!=len(used):
            continue
        compiled = compile_deployment(dep, nodes)
        registered = {'compiled': compiled, 'nodes': nodes, 'groups': {}}
        result['deployments'][dep['id']] = registered
        # Save discoveries before rendering so failures remain diagnosable.
        save(root/'preflight-progress.json', result)
        for group_index, (key, group) in enumerate(compiled['groups'].items(), 1):
            node, binding = nodes[group['node']], bindings['nodes'][group['node']]
            first_port = group['ports'][0]
            local_endpoint = group['health_urls'][0].rsplit('/', 1)[0]
            service = {'health_urls': group['health_urls'], 'model_health_urls':group['model_health_urls'], 'endpoint': local_endpoint,
                'reset_ports': [first_port], 'prefill_port': first_port}
            if key == compiled['endpoint_group']:
                service.update(endpoint=compiled['endpoint'], reset_urls=compiled['reset_urls'],
                               prefill_url=compiled['prefill_url'], metric_instances=compiled['metric_instances'])
            max_length = min(context_limit(i['launch']['args']) for i in dep['instances'] if i['role'] != 'proxy')
            flat = {'id': dep['id']+'-'+key.replace('/', '-'), 'model': {**dep['model'], 'max_model_len': max_length},
                'container': dep['runtime']['container'], 'npu_ids': group['npu_ids'],
                'commands': group['commands'], 'service': service, 'prefill_dp': compiled['prefill_dp']}
            local_spec = {'job_id': node_job_id(spec['job_id'],group['node'])+f'-{dep_index:03}', 'target': {'host': binding['host']},
                          'execution': spec['execution'], 'cleanup': spec['cleanup'],
                          'benchmark_cases': [c for c in spec.get('cases', []) if any(t['deployment_id']==dep['id'] and t['case_id']==c['id'] for t in spec['sub_tasks'])],
                          'storage': {'root':pools[(dep['id'],group['node'])]['node_root'],
                           'subtasks':sum(1 for t in spec['sub_tasks'] if any(bindings['nodes'][i['placement']['node']]['host']==binding['host'] for d in spec['deployments'] if d['id']==t['deployment_id'] for i in d['instances']))},
                          'container_pool':pools[(dep['id'],group['node'])]}
            try:
                reply = prepare_call({**binding,'package':node['package']}, {'action': 'render', 'run_dir': node['root'],
                    'job_id': local_spec['job_id'], 'spec': local_spec, 'deployment': flat, 'index': group_index,
                    'bindings': binding, 'ports': group['ports']})
                registered['groups'][key] = {**reply, 'host': binding['host'],
                    'script': str(Path(node['package'])/'scripts/node_rpc.py')}
                save(root/'preflight-progress.json', result)
            except Exception as exc:
                failures.append({'deployment':dep['id'],'group':key,'error':repr(exc)})
    save(root/'preflight-findings.json', {'status':'REVIEW_REQUIRED' if failures else 'PASSED','findings':failures})
    if failures: raise ValueError('Attended preflight decision required; see preflight-findings.json')
    save(root/'resolved-cluster.json', result)
    return {'status': 'PASSED', 'message': 'All topology nodes prepared', 'artifacts': [str(root/'resolved-cluster.json')]}


def clients(deployment):
    return {key: RemoteNode(item['host'], item['script'], item['run_dir'], item['job_id'], item['service_fingerprint'], item['scripts_sha256'])
            for key, item in deployment['groups'].items()}


def release_all(deployment, policy, scope):
    outcomes = {}
    for key, client in reversed(list(clients(deployment).items())):
        try:
            outcomes[key] = client.call('release', policy=policy, scope=scope)
        except Exception as exc:
            outcomes[key] = {'status': 'FAILED', 'error': repr(exc)}
    return outcomes


def deploy(root, spec, deployment):
    nodes = clients(deployment)
    state_path = root/'service-state.json'
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    if previous.get('status') == 'READY':
        responses = {key:node.call('identity') for key,node in nodes.items()}
        identities = {key:r['identity'] for key,r in responses.items()}
        if all(r.get('healthy',True) for r in responses.values()) and identities == previous['identities']:
            if spec.get('_event_context'):
                emit(Path(spec['_event_root']), 'smoke', 'skipped', reason='Verified READY service reused', **spec['_event_context'])
            return
        raise ValueError('Registered service generation changed; finalize before retry')
    generations=[int(p.name.split('-')[-1]) for p in root.glob('deploy-*') if p.name.split('-')[-1].isdigit()]
    offset=max(generations,default=0)
    for attempt in range(1, spec['execution']['max_retries']+2):
        folder = root/f'deploy-{offset+attempt:02}'
        folder.mkdir(parents=True, exist_ok=True)
        smoke_started = False
        if spec.get('_event_context'):
            emit(Path(spec['_event_root']), 'deploy_attempt', 'doing', attempt=attempt, **spec['_event_context'])
        try:
            for node in nodes.values():
                node.call('start')
            deadline = time.monotonic()+spec['execution']['startup_timeout_seconds']
            while True:
                responses = {key:node.call('identity',include_proxy=False) for key,node in nodes.items()}
                if all(r.get('healthy', True) for r in responses.values()):
                    identities = {key:r['identity'] for key,r in responses.items()}
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('Topology startup deadline exceeded')
                time.sleep(5)
            nodes[deployment['compiled']['endpoint_group']].call('start_proxy')
            while True:
                responses={key:node.call('identity') for key,node in nodes.items()}
                if all(r.get('healthy',True) for r in responses.values()):
                    identities={key:r['identity'] for key,r in responses.items()}
                    break
                if time.monotonic()>=deadline:
                    raise TimeoutError('Proxy health deadline exceeded')
                time.sleep(2)
            event_context = spec.get('_event_context', {})
            if event_context:
                emit(Path(spec['_event_root']), 'smoke', 'doing', attempt=attempt, **event_context)
            smoke_started = True
            request = {'model': spec['_served_name'], 'messages': [{'role':'user','content':'Reply with OK.'}],
                       'max_tokens':8, 'stream':False, 'temperature':0}
            save(folder/'smoke-request.json', request)
            endpoint = deployment['compiled']['endpoint']
            command(['curl','--noproxy','*','--silent','--show-error','--fail-with-body','--max-time',
                str(spec['execution']['smoke_timeout_seconds']),'-H','Content-Type: application/json',
                '--data-binary','@'+str(folder/'smoke-request.json'),'-o',str(folder/'smoke-response.json'),
                endpoint+'/v1/chat/completions'], timeout=spec['execution']['smoke_timeout_seconds']+10)
            response = json.loads((folder/'smoke-response.json').read_text())
            message = response.get('choices', [{}])[0].get('message', {})
            if response.get('error') or not any(message.get(k) for k in ['content','reasoning','reasoning_content']):
                raise ValueError('Smoke returned no generated content')
            for key, node in nodes.items():
                node.call('ready', identity=identities[key])
            if event_context:
                emit(Path(spec['_event_root']), 'smoke', 'done', attempt=attempt, **event_context)
                emit(Path(spec['_event_root']), 'deploy_attempt', 'done', attempt=attempt, **event_context)
            save(state_path, {'status':'READY','identities':identities})
            return
        except Exception as exc:
            if smoke_started and spec.get('_event_context'):
                emit(Path(spec['_event_root']), 'smoke', 'fail', attempt=attempt, error=repr(exc), **spec['_event_context'])
            if spec.get('_event_context'):
                emit(Path(spec['_event_root']), 'deploy_attempt', 'fail', attempt=attempt, error=repr(exc), **spec['_event_context'])
            released = release_all(deployment, 'stop_container', 'failed_deployment')
            save(folder/'failure.json', {'error':repr(exc),'releases':released})
            if any(x['status'] != 'PASSED' for x in released.values()):
                raise RuntimeError('Node cleanup unverified; no retry') from exc
            if isinstance(exc, PortConflict) and attempt < spec['execution']['max_retries']+1:
                task_root = Path(spec['_event_root'])
                bindings = json.loads((task_root/'common/inputs/bindings.json').read_text(encoding='utf-8-sig'))
                clean_spec = {k:v for k,v in spec.items() if not k.startswith('_')}
                dep_id = spec['_event_context']['deployment_id']
                generation = str(time.time_ns())
                save(folder/'port-reallocation.json', {'status':'PREPARING','generation':generation,'previous':deployment})
                preflight(task_root, clean_spec, bindings, deployment_id=dep_id, generation=generation)
                replacement = json.loads((task_root/'resolved-cluster.json').read_text())['deployments'][dep_id]
                deployment.clear(); deployment.update(replacement)
                nodes = clients(deployment)
                save(folder/'port-reallocation.json', {'status':'PASSED','generation':generation,'groups':deployment['groups']})
            if isinstance(exc,ValueError) or attempt == spec['execution']['max_retries']+1:
                raise
            if spec.get('_event_context'):
                emit(Path(spec['_event_root']), 'deploy_attempt', 'retry', attempt=attempt+1, **spec['_event_context'])


def finalize(root, spec, resolved):
    results = {key: release_all(dep, spec['cleanup']['task_end'], 'job') for key, dep in resolved['deployments'].items()}
    save(root/'release-results.json', results)
    failed = any(r['status'] != 'PASSED' for groups in results.values() for r in groups.values())
    return {'status':'FAILED' if failed else 'PASSED', 'message':'All registered node releases checked', 'artifacts':[str(root/'release-results.json')], 'releases':results}


def reconcile_failure(root, deployment, case_id):
    releases=release_all(deployment, 'stop_container', 'failed_sub_task')
    safe=bool(releases) and all(v['status']=='PASSED' for v in releases.values())
    save(root/'service-state.json', {'status':'RELEASED' if safe else 'UNKNOWN'})
    return {'continue':safe, 'releases':releases}


def run(root, spec, resolved):
    if (root/'summary.json').exists():
        summary=json.loads((root/'summary.json').read_text())
        summary['cleanup']=finalize(root,spec,resolved)
        if summary['cleanup']['status']!='PASSED':summary['status']='FAILED'
        save(root/'summary.json',summary)
        return summary
    progress=root/'task-state.json'
    results=json.loads(progress.read_text()) if progress.exists() else {}
    if any(v['status']=='RUNNING' for v in results.values()):
        raise ValueError('Interrupted topology task; finalize and inspect before a new run')
    try:
        for index, subtask in enumerate(spec['sub_tasks']):
            if subtask['id'] in results:
                continue
            blocker = higher_concurrency_blocker(spec, subtask, results)
            if blocker:
                results[subtask['id']]={'status':'NOT_RUN','blocked_by':blocker,
                    'reason':'Higher concurrency branch paused after failure'}
                save(progress,results)
                continue
            results[subtask['id']]={'status':'RUNNING'}
            save(progress,results)
            dep_id = subtask['deployment_id']
            dep = resolved['deployments'][dep_id]
            definition = next(d for d in spec['deployments'] if d['id'] == dep_id)
            folder = root/'sub_tasks'/subtask['id']; folder.mkdir(parents=True, exist_ok=True)
            deployment_root = root/'.internal/services'/dep_id; deployment_root.mkdir(parents=True, exist_ok=True)
            # case ID in result artifacts identifies the execution, not the reusable load definition.
            case = copy.deepcopy(next(c for c in spec['cases'] if c['id'] == subtask['case_id']))
            case['id'] = subtask['id']
            context = dict(job_id=spec['job_id'], subcase_id=subtask['id'], case_id=subtask['case_id'], deployment_id=dep_id, phase_id=subtask['phase_id'])
            case['_event_context'] = context
            emit(root, 'deploy', 'doing', **context)
            failure_stage='deploy'
            try:
                deploy(deployment_root, {**spec,'_served_name':definition['model']['served_name'], '_event_root':str(root), '_event_context':context}, dep)
                emit(root, 'deploy', 'done', **context)
                from task_runtime import storage_event
                storage_event(root,'deployment_verified',deployment_id=dep_id)
                client = clients(dep)[dep['compiled']['endpoint_group']]
                before={key:node.call('identity')['identity'] for key,node in clients(dep).items()}
                failure_stage='testing'
                emit(root, 'testing', 'doing', **context)
                result = client.call('evaluate', timeout=(case['timeout_seconds']+120)*(spec['execution']['max_retries']+1), case=case)
                after={key:node.call('identity')['identity'] for key,node in clients(dep).items()}
                if before!=after:
                    raise RuntimeError('A topology service changed during evaluation')
                results[subtask['id']] = result
                if result['status'] != 'PASSED':
                    save(folder/'failed-evaluation.json', result)
                    raise RuntimeError('Evaluation failed; see failed-evaluation.json')
                emit(root, 'testing', 'done' if result['status']=='PASSED' else 'fail', **context)
            except Exception as exc:
                emit(root, failure_stage, 'fail', error=repr(exc), **context)
                emit(root, 'reconcile', 'doing', **context)
                recovery=reconcile_failure(deployment_root,dep,subtask['id'])
                emit(root, 'reconcile', 'done' if recovery['continue'] else 'blocked', **context)
                results[subtask['id']] = {'status':'FAILED','error':repr(exc),'failure_stage':failure_stage,
                    'on_failure':subtask['on_failure'],'phase_id':subtask['phase_id'],'reconciliation':recovery}
                if not recovery['continue']:
                    results[subtask['id']]['blocked']=True
                    break
            finally:
                save(folder/'result.json', results[subtask['id']])
                save(progress,results)
                from task_runtime import storage_event
                storage_event(root,'case_finished',sub_task_id=subtask['id'],case_id=subtask['case_id'],deployment_id=dep_id,status=results[subtask['id']]['status'])
            next_dep = spec['sub_tasks'][index+1]['deployment_id'] if index+1 < len(spec['sub_tasks']) else None
            if results[subtask['id']].get('reconciliation', {}).get('continue'):
                continue
            policy = subtask['after_case']
            if next_dep and next_dep != dep_id and policy=='keep_service':
                policy='stop_service'
            emit(root, 'release', 'doing', policy=policy, **context)
            releases = release_all(dep, policy, 'sub_task:'+subtask['id'])
            save(folder/'release.json', releases)
            emit(root, 'release', 'done' if all(r['status']=='PASSED' for r in releases.values()) else 'fail', policy=policy, **context)
            if any(r['status'] != 'PASSED' for r in releases.values()):
                results[subtask['id']]['release_failed'] = True
                save(folder/'result.json',results[subtask['id']])
                save(progress,results)
                break
            if policy != 'keep_service':
                save(deployment_root/'service-state.json', {'status':'RELEASED'})
    finally:
        emit(root, 'job_release', 'doing', job_id=spec['job_id'])
        cleanup = finalize(root, spec, resolved)
        emit(root, 'job_release', 'done' if cleanup['status']=='PASSED' else 'fail', job_id=spec['job_id'])
        for subtask in spec['sub_tasks']:
            results.setdefault(subtask['id'], {'status':'NOT_RUN'})
        passed = cleanup['status']=='PASSED' and all(v['status']=='PASSED' and not v.get('release_failed') for v in results.values())
        summary = {'status':'PASSED' if passed else 'FAILED','message':'Serial topology task finished; inspect per-node evidence','artifacts':[str(root/'summary.json')],'sub_tasks':results,'cleanup':cleanup}
        save(root/'summary.json', summary)
    return summary



from task_model import higher_concurrency_blocker
