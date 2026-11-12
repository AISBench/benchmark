"""AISBench external-eval adapter. Caller binds execution to the verified tester."""
import json
import os
from pathlib import Path
import time
from . import aisbench_cli
from .plan_model import normalize_external_plan, validate_eval_bindings
from .process_guard import remaining, run_logged
from .protocol import ProtocolError, now
from .http_probe import wait_health
from .native_results import validate_native_result
from .durable import atomic_json
from .json_io import load_json
from . import metric_evidence


def require_linux():
    if __import__('sys').platform != 'linux':
        raise ProtocolError('INVALID_INPUT: production evaluator requires Linux')


def prepare_case(plan, bindings, case_id, folder, check_files=True):
    normalized, derived = normalize_external_plan(plan)
    validate_eval_bindings(plan, bindings)
    if derived or plan['unresolved']:
        raise ProtocolError('INVALID_INPUT: case must be normalized and reviewed')
    cases = [case for case in normalized['cases'] if case['case_id'] == case_id]
    if len(cases) != 1:
        raise ProtocolError('INVALID_INPUT: unknown case')
    case = cases[0]
    tester = bindings['testers'][plan['test_binding']]
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    spec = dict(model={'path': tester['model_path'], 'served_name': plan['service']['model']},
                benchmark={'datasets': tester['datasets']})
    config = aisbench_cli.write_config(spec, case, plan['service'], folder, check_files=check_files)
    (folder/'native-expectations.json').write_text(json.dumps([['service', 'dataset']]), encoding='utf-8')
    argv = aisbench_cli.commands(tester['aisbench'], config, case, folder)[0][1]
    return case, tester, argv


def execute_external_case(plan, bindings, case_id, folder, cancelled, *, verified_local_host):
    """Must run on the explicitly verified tester; never choose local by hostname guess."""
    normalized, derived = normalize_external_plan(plan)
    validate_eval_bindings(plan, bindings)
    if derived or plan['unresolved']:
        raise ProtocolError('INVALID_INPUT: reviewed normalized plan required')
    if cancelled.is_set():
        from .process_guard import Cancelled
        raise Cancelled('Operation cancelled')
    tester = bindings['testers'][plan['test_binding']]
    if tester['host'] != verified_local_host:
        raise ProtocolError('INVALID_INPUT: tester identity mismatch')
    require_linux()
    case = next((c for c in plan['cases'] if c['case_id'] == case_id), None)
    if case is None:
        raise ProtocolError('INVALID_INPUT: unknown case')
    deadline = time.monotonic()+case['timeout_seconds']
    # Health is rechecked now; no previous deployment health claim is reused.
    url = plan['service']['endpoint'].rstrip('/')+'/health'
    health_evidence = wait_health([url],deadline,cancelled)
    # Imports and dataset checks execute in the bound tester interpreter, under
    # the same stage deadline, rather than in the coordinator's Python environment.
    case, tester, argv = prepare_case(plan, bindings, case_id, folder,check_files=False)
    folder = Path(folder)
    (folder/'health.json').write_text(json.dumps(health_evidence), encoding='utf-8')
    if not Path(tester['aisbench']).is_file() or not Path(tester['model_path']).is_dir():
        raise ProtocolError('INVALID_INPUT: AISBench or tokenizer assets unavailable')
    environment = dict(os.environ)
    environment.update(tester.get('environment', {}))
    atomic_json(folder/'tester-environment.json', tester.get('environment', {}))
    preflight_request=folder/'preflight-request.json'
    atomic_json(preflight_request,dict(tester=tester,case=case,config=str(folder/'aisbench-config.py')))
    bootstrap='import sys; sys.path.insert(0, '+repr(str(Path(__file__).parent.parent))+'); from deploy_eval_v2.benchmark_preflight import main; main()'
    preflight=run_logged([tester['python'],'-B','-c',bootstrap,'--request',str(preflight_request),
                         '--result',str(folder/'preflight.json')],folder/'preflight.log',deadline,cancelled,env=environment)
    if preflight or load_json(folder/'preflight.json').get('status')!='VERIFIED':
        raise ProtocolError('INVALID_INPUT: benchmark environment preflight failed')
    if case['reset_cache']:
        from .cache_reset import reset_instances
        reset_instances(plan['service'],folder,deadline,cancelled)
    metrics=plan['service'].get('metrics',[])
    metric_evidence.capture(metrics,folder,'before',deadline,cancelled)
    returncode = run_logged(argv, folder/'aisbench.log', deadline, cancelled,env=environment)
    metric_evidence.capture(metrics,folder,'after',deadline,cancelled)
    prefix=metric_evidence.summarize(folder,case)
    (folder/'execution.json').write_text(json.dumps({'returncode': returncode}), encoding='utf-8')
    native=validate_native_result(folder, case,deadline=deadline,cancelled=cancelled)
    synthetic=None
    if tester['datasets'][case['dataset_id']]['type']=='aisbench_synthetic':
        postflight=folder/'postflight-request.json'
        atomic_json(postflight,dict(folder=str(folder),case=case,model_path=tester['model_path']))
        script='import sys; sys.path.insert(0, '+repr(str(Path(__file__).parent.parent))+'); from deploy_eval_v2.native_postflight import main; main()'
        code=run_logged([tester['python'],'-B','-c',script,'--request',str(postflight)],folder/'postflight.log',deadline,cancelled,env=environment)
        if not code and (folder/'synthetic-evidence.json').exists():
            synthetic=load_json(folder/'synthetic-evidence.json')
        else:
            synthetic=dict(status='UNVERIFIED',reason='POSTFLIGHT_PROCESS_FAILED')
            atomic_json(folder/'synthetic-evidence.json',synthetic)
    from .performance_evidence import summarize
    performance=summarize(folder,case,native,deadline=deadline,cancelled=cancelled)
    verified=performance['status']=='VERIFIED' and (not metrics or prefix['status']=='VERIFIED')
    # Prefix routing coverage is separate from native request performance.
    if 'shared_prefix_ratio' in case:
        dataset_verified=(synthetic is not None and synthetic['status']=='MEASURED') or tester['datasets'][case['dataset_id']].get('generation_manifest') is not None
        verified=verified and dataset_verified and prefix['routing_coverage']=='VERIFIED' and prefix['warmup_route_coverage'] in ('VERIFIED','NOT_REQUESTED')
    result = dict(schema_version='1.0', service_id=plan['service']['service_id'],
                  case_id=case_id, execution_status='SUCCEEDED', evidence_status='COMPLETE' if verified else 'PARTIAL',
                  performance_validity='VERIFIED' if verified else 'UNVERIFIED',
                  performance=performance,
                  performance_evidence=dict(host=verified_local_host,path=str(folder/'performance-evidence.json')),
                  prefix_metric_status=prefix['status'],prefix_evidence=dict(host=verified_local_host,path=str(folder/'prefix-evidence.json')),
                  reason='Native request performance evidence verified' if verified else 'See performance and prefix evidence for unverified conditions',
                  time=now(), actual_load=case)
    if synthetic is not None:
        result['synthetic_evidence']=dict(status=synthetic['status'],host=verified_local_host,path=str(folder/'synthetic-evidence.json'))
    (folder/'result.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    return result
