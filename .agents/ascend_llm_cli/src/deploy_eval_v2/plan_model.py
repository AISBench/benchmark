"""External-eval contract normalization, before review; no network access."""
import copy
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from jsonschema import Draft202012Validator
from .json_io import load_json
from .protocol import ProtocolError


def normalize_external_plan(plan):
    schema = load_json(Path(__file__).parent/'schemas/external-eval-plan.schema.json')
    if next(Draft202012Validator(schema).iter_errors(plan), None) is not None:
        raise ProtocolError('INVALID_INPUT: external eval plan schema')
    service = plan['service']
    targets=service.get('reset_targets',[])
    if (len({v['id'] for v in targets})!=len(targets) or
            len({v['url'] for v in targets})!=len(targets)):
        raise ProtocolError('INVALID_INPUT: duplicate reset target')
    for target in targets:
        for key in ('url','metrics_url'):
            try:
                parsed=urlsplit(target[key])
                if (parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username is not None or
                    parsed.password is not None or parsed.query or parsed.fragment or parsed.port==0 or
                    any(ord(c)<=32 for c in target[key])):
                    raise ValueError('reset URL')
            except ValueError as error:
                raise ProtocolError('INVALID_INPUT: reset endpoint') from error
        reset,metrics_url=urlsplit(target['url']),urlsplit(target['metrics_url'])
        if (reset.scheme,reset.netloc)!=(metrics_url.scheme,metrics_url.netloc):
            raise ProtocolError('INVALID_INPUT: reset and idle evidence must address same instance')
    metrics=service.get('metrics',[])
    if len({item['id'] for item in metrics})!=len(metrics):
        raise ProtocolError('INVALID_INPUT: duplicate metric instance')
    for item in metrics:
        try:
            target=urlsplit(item['url'])
            if (target.scheme not in ('http','https') or not target.hostname or target.username is not None or
                    target.password is not None or target.query or target.fragment or target.port==0 or
                    any(ord(c)<=32 for c in item['url'])):
                raise ValueError('metric URL')
        except ValueError as error:
            raise ProtocolError('INVALID_INPUT: metric endpoint') from error
    try:
        endpoint = urlsplit(service['endpoint'])
        port = endpoint.port
        if (endpoint.scheme not in ('http', 'https') or not endpoint.hostname or
                endpoint.username is not None or endpoint.password is not None or
                endpoint.query or endpoint.fragment or port == 0 or
                any(ord(c) <= 32 for c in service['endpoint'])):
            raise ValueError('endpoint')
    except ValueError as exc:
        raise ProtocolError('INVALID_INPUT: service endpoint') from exc
    ids = [case['case_id'] for case in plan['cases']]
    if len(ids) != len(set(ids)):
        raise ProtocolError('INVALID_INPUT: duplicate case identity')
    normalized = copy.deepcopy(plan)
    derived = []
    for case in normalized['cases']:
        if case['reset_cache'] and not service['permissions']['reset_cache']:
            raise ProtocolError('INVALID_INPUT: cache reset not authorized')
        if case['reset_cache'] and not targets:
            raise ProtocolError('INVALID_INPUT: explicit cache-reset target adapter requires targets')
        if 'formal_requests' not in case:
            multiplier = 10 if case['steady_state'] else 4
            case['formal_requests'] = multiplier*case['concurrency']
            derived.append({'case_id': case['case_id'], 'field': 'formal_requests',
                            'value': case['formal_requests'], 'source': f'concurrency*{multiplier}'})
    return normalized, derived


def validate_eval_bindings(plan, bindings):
    """Validate references and Linux paths, without probing the bound machine."""
    normalize_external_plan(plan)
    schema = load_json(Path(__file__).parent/'schemas/eval-bindings.schema.json')
    if next(Draft202012Validator(schema).iter_errors(bindings), None) is not None:
        raise ProtocolError('INVALID_INPUT: eval bindings schema')
    if bindings['task_id'] != plan['task_id']:
        raise ProtocolError('INVALID_INPUT: binding task identity mismatch')
    if plan['test_binding'] not in bindings['testers']:
        raise ProtocolError('INVALID_INPUT: missing test binding')

    def path(value):
        parsed = PurePosixPath(value)
        if (not parsed.is_absolute() or '..' in parsed.parts or '\\' in value or
                any(ord(c) < 32 for c in value) or str(parsed) == '/'):
            raise ProtocolError('INVALID_INPUT: Linux binding path')

    for tester in bindings['testers'].values():
        for key in ('work_root', 'python', 'aisbench', 'model_path'):
            path(tester[key])
        for dataset in tester['datasets'].values():
            if dataset['type'] == 'aisbench_synthetic':
                continue
            path(dataset['path'])
            if PurePosixPath(dataset['path']).suffix.lower() != '.'+dataset['type']:
                raise ProtocolError('INVALID_INPUT: dataset format and path disagree')
            if 'generation_manifest' in dataset:
                path(dataset['generation_manifest'])
    tester = bindings['testers'][plan['test_binding']]
    for case in plan['cases']:
        if case['dataset_id'] not in tester['datasets']:
            raise ProtocolError('INVALID_INPUT: missing case dataset binding')
        dataset = tester['datasets'][case['dataset_id']]
        if 'shared_prefix_ratio' in case and dataset['type'] != 'aisbench_synthetic' and 'generation_manifest' not in dataset:
            raise ProtocolError('INVALID_INPUT: prefix dataset requires generation evidence')
