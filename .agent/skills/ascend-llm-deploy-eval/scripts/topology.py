"""Compile deployment instances and connections using discovered node bindings.

No SSH or Docker side effects. Node-local execution retains the existing lifecycle.
"""
import copy
import re


def validate_deployment(dep):
    errors = []
    instances = {i['id']: i for i in dep['instances']}
    if len(instances) != len(dep['instances']):
        errors.append('Duplicate instance ID')
    occupied = {}
    for item in instances.values():
        place = item['placement']
        for device in place['devices']:
            key = (place['node'], device)
            if key in occupied:
                errors.append('Device assigned to multiple model instances: '+str(key))
            occupied[key] = item['id']
        if item['role'] == 'proxy':
            if place['devices']:
                errors.append('Proxy must not claim model devices')
            continue
        if 'args' not in item['launch'] or any(k not in item['checks'] for k in ['prefix_cache_reset','metrics']):
            errors.append('Model launch/checks incomplete: '+item['id'])
            continue
        args = item['launch']['args']
        dp, tp = args.get('data-parallel-size', 1), args.get('tensor-parallel-size', 1)
        if not isinstance(dp, int) or not isinstance(tp, int) or dp < 1 or tp < 1 or dp*tp != len(place['devices']):
            errors.append('DP/TP does not match devices: '+item['id'])
        if not isinstance(args.get('max-model-len'), int) or args['max-model-len'] < 1:
            errors.append('Missing positive max-model-len: '+item['id'])
        generated = {'host', 'port', 'served-model-name', 'device-ids', 'kv-transfer-config'}
        if generated.intersection(args):
            errors.append('Runtime-generated arguments cannot be duplicated: '+item['id'])
    endpoint = dep['endpoint']
    if endpoint['instance'] not in instances or endpoint['port'] not in instances[endpoint['instance']]['ports']:
        errors.append('Invalid deployment endpoint')
    kv = dep.get('connections', {}).get('kv_transfer', [])
    routing = dep.get('connections', {}).get('routing', [])
    if dep['topology'] == 'colocated':
        if len(instances) != 1 or next(iter(instances.values()))['role'] != 'mixed' or kv or routing:
            errors.append('Colocated deployment requires one mixed instance and no PD connections')
    else:
        # First multi-host implementation intentionally supports one P, one D and one proxy.
        roles = sorted(i['role'] for i in instances.values())
        if roles != ['decode', 'prefill', 'proxy'] or len(kv) != 1 or len(routing) != 1:
            errors.append('Initial PD executor requires one P, one D, one proxy and one connection each')
        for link in kv:
            for key, role in [('producer', 'prefill'), ('consumer', 'decode')]:
                if instances.get(link[key], {}).get('role') != role:
                    errors.append('Invalid KV '+key)
        for link in routing:
            if instances.get(link['proxy'], {}).get('role') != 'proxy':
                errors.append('Invalid routing proxy')
            for key, role in [('prefillers', 'prefill'), ('decoders', 'decode')]:
                if len(link[key]) != 1 or any(instances.get(i, {}).get('role') != role for i in link[key]):
                    errors.append('Invalid routing '+key)
        if instances.get(endpoint['instance'], {}).get('role') != 'proxy':
            errors.append('PD endpoint must reference proxy')
    return errors


def cli_args(values):
    result = []
    for name, value in values.items():
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', name):
            raise ValueError('Invalid CLI option: '+name)
        if isinstance(value, bool):
            result.append('--'+('' if value else 'no-')+name)
        elif isinstance(value, (dict, list)):
            result.extend(['--'+name, {'json': value}])
        else:
            result.extend(['--'+name, str(value)])
    return result


def compile_deployment(dep, nodes):
    """nodes keyed by logical slot contain network, per-instance ports and paths."""
    errors = validate_deployment(dep)
    if errors:
        raise ValueError('; '.join(errors))
    instances = {i['id']: i for i in dep['instances']}
    addresses = {}
    for item in instances.values():
        node = nodes[item['placement']['node']]
        ports = node['ports'][item['id']]
        addresses[item['id']] = {'host': node['network']['ip'], 'ports': ports}
    def url(instance, port='http'):
        address = addresses[instance]
        return 'http://'+address['host']+':'+str(address['ports'][port])
    groups = {}
    for item in instances.values():
        placement = item['placement']
        key = placement['node']+'/model'
        node = nodes[placement['node']]
        ports = node['ports'][item['id']]
        group = groups.setdefault(key, {'node': placement['node'], 'container_group': 'model',
            'commands': [], 'npu_ids': [], 'ports': [], 'health_urls': [], 'model_health_urls':[], 'instances': []})
        group['instances'].append(item['id'])
        group['npu_ids'].extend(placement['devices'])
        for name, width in item['ports'].items():
            group['ports'].extend(range(ports[name], ports[name]+width))
        group['health_urls'].append(url(item['id'])+item['checks']['health'])
        if item['role']!='proxy':
            group['model_health_urls'].append(url(item['id'])+item['checks']['health'])
        launch = item['launch']
        env = copy.deepcopy(launch.get('env', {}))
        if item['role'] == 'proxy':
            routing = next(x for x in dep['connections']['routing'] if x['proxy'] == item['id'])
            argv = ['python3', '{{paths.proxy_program}}', '--host', node['network']['ip'], '--port', str(ports['http'])]
            for prefix, peers in [('prefiller', routing['prefillers']), ('decoder', routing['decoders'])]:
                argv += ['--'+prefix+'-hosts']+[addresses[p]['host'] for p in peers]
                argv += ['--'+prefix+'-ports']+[str(addresses[p]['ports']['http']) for p in peers]
        else:
            env.update(HCCL_IF_IP=node['network']['ip'], VLLM_HOST_IP=node['network']['ip'],
                       GLOO_SOCKET_IFNAME=node['network']['interface'], TP_SOCKET_IFNAME=node['network']['interface'],
                       HCCL_SOCKET_IFNAME=node['network']['interface'], VLLM_SERVER_DEV_MODE='1')
            args = copy.deepcopy(launch['args'])
            args.update(host=node['network']['ip'], port=ports['http'])
            args['served-model-name'] = dep['model']['served_name']
            args['device-ids'] = ','.join(map(str, placement['devices']))
            if 'rpc' in ports:
                args['data-parallel-rpc-port'] = ports['rpc']
                args['data-parallel-address'] = node['network']['ip']
            if 'hccl' in ports:
                env['HCCL_IF_BASE_PORT'] = str(ports['hccl'])
            if item['role'] in ['prefill', 'decode']:
                link = next(x for x in dep['connections']['kv_transfer'] if item['id'] in [x['producer'], x['consumer']])
                p = instances[link['producer']]['launch']['args']
                d = instances[link['consumer']]['launch']['args']
                args['kv-transfer-config'] = {'kv_connector': link['connector'],
                    'kv_role': 'kv_producer' if item['role'] == 'prefill' else 'kv_consumer',
                    'kv_port': str(ports['kv']), 'engine_id': dep['id']+'-'+item['id'],
                    'kv_connector_extra_config': {'ascend_local_comm_res_path': '{{paths.comm}}',
                        'prefill': {'dp_size': p.get('data-parallel-size', 1), 'tp_size': p.get('tensor-parallel-size', 1)},
                        'decode': {'dp_size': d.get('data-parallel-size', 1), 'tp_size': d.get('tensor-parallel-size', 1)}}}
            argv = ['vllm', 'serve', '{{model.path}}']+cli_args(args)
        group['commands'].append({'name': item['id'], 'role':item['role'], 'argv': argv, 'env': env,
                                  'bootstrap': launch.get('bootstrap', '')})
    prefill = next(i for i in instances.values() if i['role'] in ['prefill', 'mixed'])
    endpoint = dep['endpoint']
    return {'groups': groups, 'endpoint': url(endpoint['instance'], endpoint['port']),
        'endpoint_group': instances[endpoint['instance']]['placement']['node']+'/model',
        'reset_urls': [url(i['id'])+i['checks']['prefix_cache_reset'] for i in instances.values() if i['role'] != 'proxy'],
        'metrics_urls': [url(i['id'])+i['checks']['metrics'] for i in instances.values() if i['role'] != 'proxy'],
        'prefill_url': url(prefill['id']), 'prefill_dp': prefill['launch']['args'].get('data-parallel-size', 1)}
