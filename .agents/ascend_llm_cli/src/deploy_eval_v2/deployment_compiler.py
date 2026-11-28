"""Pure bound deployment compilation. No SSH, Docker or filesystem probes."""
import copy
import json
from pathlib import Path, PurePosixPath
import re
import shlex
from jsonschema import Draft202012Validator, FormatChecker
from .deployment_model import validate_deploy_plan
from .json_io import load_json
from .mount_profiles import resolve_mounts
from .container_baseline import DEFAULT_CONTAINER, DEVICES
from .topology import compile_deployment
from .protocol import ProtocolError

TOKEN = re.compile(r'\{\{([A-Za-z_][A-Za-z0-9_.]*)\}\}')


def expand(value, context, shell=False):
    if isinstance(value, dict):
        return {k: expand(v, context, shell) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, context, shell) for v in value]
    if not isinstance(value, str):
        return value
    def lookup(key):
        result = context
        try:
            for part in key.split('.'):
                result = result[part]
        except (KeyError, TypeError) as exc:
            raise ProtocolError('INVALID_INPUT: missing binding '+key) from exc
        return result
    full = TOKEN.fullmatch(value)
    if full and not shell:
        return copy.deepcopy(lookup(full[1]))
    def replace(match):
        result = lookup(match[1])
        if isinstance(result, (list, dict)):
            raise ProtocolError('INVALID_INPUT: embedded structured binding')
        return shlex.quote(str(result)) if shell else str(result)
    return TOKEN.sub(replace, value)


def compile_bound_deployment(plan, bindings):
    validate_deploy_plan(plan)
    schema = load_json(Path(__file__).parent/'schemas/deploy-bindings.schema.json')
    if next(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(bindings), None):
        raise ProtocolError('INVALID_INPUT: deployment bindings schema')
    if bindings['task_id'] != plan['task_id']:
        raise ProtocolError('INVALID_INPUT: deployment task identity')
    deployment = copy.deepcopy(plan['deployment'])
    selected = {}
    contexts = {}
    def linux_path(value):
        if not value.startswith('/') or '..' in PurePosixPath(value).parts or '\\' in value or any(ord(c)<32 for c in value):
            raise ProtocolError('INVALID_INPUT: bound Linux path')
    try:
        for instance in deployment['instances']:
            slot = instance['placement']['node']
            binding = bindings['nodes'][slot]
            node = binding['deployments'][deployment['id']]
            selected[slot] = node
            ports = node['ports'][instance['id']]
            if set(ports) != set(instance['ports']):
                raise ProtocolError('INVALID_INPUT: port binding names differ')
            for value in [binding['work_dir'], *binding['model_paths'].values(), *binding.get('paths', {}).values()]:
                linux_path(value)
            root = str(PurePosixPath(binding['work_dir'])/plan['task_id'])
            context = dict(model={'path': binding['model_paths'][deployment['model']['id']],
                                  'served_name': deployment['model']['served_name']},
                           paths=binding.get('paths', {}), network=node['network'], ports=ports,
                           task_dir=root, run_dir=root+'/logs/'+instance['id'])
            contexts[instance['id']] = context
            launch = instance['launch']
            for field in ('args', 'env', 'kv_transfer_config'):
                if field in launch:
                    launch[field] = expand(launch[field], context)
            if instance['role'] != 'proxy':
                launch['preserve_reference'] = True
                if 'hccl' in ports and 'HCCL_IF_BASE_PORT' in launch.get('env', {}) and str(launch['env']['HCCL_IF_BASE_PORT']) != str(ports['hccl']):
                    raise ProtocolError('INVALID_INPUT: HCCL port conflicts with binding')
        # Different logical nodes sharing a host could hide device/port collisions.
        hosts = [bindings['nodes'][slot]['host'].split('@')[-1].lower() for slot in selected]
        if len(hosts) != len(set(hosts)):
            raise ProtocolError('INVALID_INPUT: duplicate physical host aliases; use one node identity')
        compiled = compile_deployment(deployment, selected)
        for group in compiled['groups'].values():
            binding = bindings['nodes'][group['node']]
            group['host'] = binding['host']
            group['root'] = str(PurePosixPath(binding['work_dir'])/plan['task_id'])
            group['mounts'] = resolve_mounts(binding)
            model_path = binding['model_paths'][deployment['model']['id']]
            for source in (group['root'], model_path):
                matching = [m for m in group['mounts'] if m['target'] == source]
                if matching and (matching[0]['source'] != source or
                                 (source == group['root'] and matching[0].get('read_only'))):
                    raise ProtocolError('INVALID_INPUT: task/model mount conflicts with binding')
                if not matching:
                    group['mounts'].append(dict(source=source, target=source, read_only=source == model_path))
            group['container'] = {**DEFAULT_CONTAINER, 'devices': list(DEVICES),
                                  **deployment['runtime']['container']}
            group['image'] = deployment['runtime']['image']
            for mount in group['mounts']:
                linux_path(mount['source']); linux_path(mount['target'])
            for command in group['commands']:
                context = contexts[command['name']]
                command['argv'] = [json.dumps(expand(a['json'], context), ensure_ascii=False) if isinstance(a, dict)
                                   else str(expand(a, context)) for a in command['argv']]
                command['env'] = expand(command['env'], context)
                if command['role'] != 'proxy' and plan['execution']['enable_cache_reset_api']:
                    command['env']['VLLM_SERVER_DEV_MODE'] = '1'
                if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) for key in command['env']):
                    raise ProtocolError('INVALID_INPUT: environment name')
                command['bootstrap'] = expand(command['bootstrap'], context, shell=True)
        return compiled
    except (KeyError, ValueError) as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError('INVALID_INPUT: bound deployment '+str(exc)) from exc
