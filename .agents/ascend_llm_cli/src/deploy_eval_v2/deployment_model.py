"""Deployment intent checks; full bound topology compilation remains separate."""
from pathlib import Path
from jsonschema import Draft202012Validator
from .json_io import load_json
from .protocol import ProtocolError


def validate_deploy_plan(plan):
    schema = load_json(Path(__file__).parent/'schemas/deploy-plan.schema.json')
    if next(Draft202012Validator(schema).iter_errors(plan), None) is not None:
        raise ProtocolError('INVALID_INPUT: deploy plan schema')
    deployment = plan['deployment']
    instances = deployment['instances']
    by_id = {item['id']: item for item in instances}
    if len(by_id) != len(instances):
        raise ProtocolError('INVALID_INPUT: duplicate instance')
    endpoint = deployment['endpoint']
    if endpoint['instance'] not in by_id or endpoint['port'] not in by_id[endpoint['instance']]['ports']:
        raise ProtocolError('INVALID_INPUT: endpoint reference')
    for item in instances:
        launch = item['launch']
        if launch.get('preserve_reference') is False:
            raise ProtocolError('INVALID_INPUT: reference preservation disabled')
        if item['role'] == 'proxy':
            if launch.get('adapter') != 'external_pd_proxy' or 'program' in launch:
                raise ProtocolError('INVALID_INPUT: proxy launch')
        else:
            if launch.get('program') != 'vllm serve' or 'adapter' in launch or 'proxy_program_key' in launch:
                raise ProtocolError('INVALID_INPUT: model launch')
            if not item['placement']['devices']:
                raise ProtocolError('INVALID_INPUT: model devices missing')
    connections = deployment['connections']
    for link in connections.get('kv_transfer', []):
        if (link['producer'] not in by_id or link['consumer'] not in by_id or
                by_id[link['producer']]['role'] != 'prefill' or by_id[link['consumer']]['role'] != 'decode'):
            raise ProtocolError('INVALID_INPUT: KV role reference')
    for route in connections.get('routing', []):
        if route['proxy'] not in by_id or by_id[route['proxy']]['role'] != 'proxy':
            raise ProtocolError('INVALID_INPUT: proxy route')
        for field, role in [('prefillers', 'prefill'), ('decoders', 'decode')]:
            if len(route[field]) != len(set(route[field])) or any(key not in by_id or by_id[key]['role'] != role for key in route[field]):
                raise ProtocolError('INVALID_INPUT: route role reference')
    roles = {i['role'] for i in instances}
    if deployment['topology'] == 'pd':
        if not {'prefill', 'decode', 'proxy'} <= roles or not connections.get('kv_transfer') or not connections.get('routing'):
            raise ProtocolError('INVALID_INPUT: incomplete PD topology')
    elif roles != {'mixed'} or connections:
        raise ProtocolError('INVALID_INPUT: colocated topology')
