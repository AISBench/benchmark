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
        if item['launch'].get('preserve_reference') and item['role'] in ('prefill','decode') and not item['launch'].get('engine_id'):
            errors.append('Reference-preserving PD launch requires engine_id: '+item['id'])
        dp, tp = args.get('data-parallel-size', 1), args.get('tensor-parallel-size', 1)
        pcp = args.get('prefill-context-parallel-size', 1)
        if any(type(v) is not int or v < 1 for v in (dp, tp, pcp)) or dp*tp*pcp != len(place['devices']):
            errors.append('DP/TP/PCP does not match devices: '+item['id'])
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
        prefills = {i['id'] for i in instances.values() if i['role']=='prefill'}
        decodes = {i['id'] for i in instances.values() if i['role']=='decode'}
        proxies = {i['id'] for i in instances.values() if i['role']=='proxy'}
        if not prefills or not decodes or len(proxies)!=1 or len(routing)!=1 or any(i['role']=='mixed' for i in instances.values()):
            errors.append('PD requires prefill/decode instances and one proxy routing table')
        pairs = [(x['producer'], x['consumer']) for x in kv]
        if len(set(pairs))!=len(pairs) or set(pairs)!={(p,d) for p in prefills for d in decodes}:
            errors.append('KV connections must cover every routed P/D pair exactly once')
        for link in kv:
            for key, role in [('producer', 'prefill'), ('consumer', 'decode')]:
                if instances.get(link[key], {}).get('role') != role:
                    errors.append('Invalid KV '+key)
        for link in routing:
            if instances.get(link['proxy'], {}).get('role') != 'proxy':
                errors.append('Invalid routing proxy')
            for key, role in [('prefillers', 'prefill'), ('decoders', 'decode')]:
                expected = prefills if role=='prefill' else decodes
                if len(set(link[key])) != len(link[key]) or set(link[key]) != expected:
                    errors.append('Invalid routing '+key)
        for item in instances.values():
            links = [x for x in kv if item['id'] in (x['producer'], x['consumer'])]
            signatures = set()
            for link in links:
                if link['producer'] not in prefills or link['consumer'] not in decodes:
                    continue
                signatures.add((link['connector'], *tuple(
                    instances[link[side]]['launch']['args'].get(name,1)
                    for side in ('producer','consumer')
                    for name in ('data-parallel-size','tensor-parallel-size','prefill-context-parallel-size'))))
            if len(signatures)>1:
                errors.append('Inconsistent KV parallelism or connector for '+item['id'])
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
    allocated = set()
    for item in instances.values():
        node = nodes[item['placement']['node']]
        ports = node['ports'][item['id']]
        for name,width in item['ports'].items():
            base=ports[name]
            if type(base) is not int or base<1 or base+width-1>65535:
                raise ValueError('Invalid bound port: '+item['id']+'.'+name)
            for port in range(base,base+width):
                identity=(node['network']['ip'],port)
                if identity in allocated:
                    raise ValueError('Overlapping bound ports: '+str(identity))
                allocated.add(identity)
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
            extras=copy.deepcopy(launch.get('args',{}))
            managed={'host','port','prefiller-hosts','prefiller-ports','decoder-hosts','decoder-ports'}
            if managed.intersection(extras):
                raise ValueError('Proxy endpoint options belong in confirmed topology/bindings')
            argv += cli_args(extras)
        else:
            expected={'HCCL_IF_IP':node['network']['ip'], 'VLLM_HOST_IP':node['network']['ip'],
                      'GLOO_SOCKET_IFNAME':node['network']['interface'], 'TP_SOCKET_IFNAME':node['network']['interface'],
                      'HCCL_SOCKET_IFNAME':node['network']['interface']}
            if launch.get('preserve_reference'):
                for key,value in expected.items():
                    if key in env and '{{' not in str(env[key]) and env[key]!=value:
                        raise ValueError('Confirmed environment conflicts with binding: '+item['id']+'.'+key)
            env.update(HCCL_IF_IP=node['network']['ip'], VLLM_HOST_IP=node['network']['ip'],
                       GLOO_SOCKET_IFNAME=node['network']['interface'], TP_SOCKET_IFNAME=node['network']['interface'],
                       HCCL_SOCKET_IFNAME=node['network']['interface'])
            args = copy.deepcopy(launch['args'])
            args.update(host=node['network']['ip'], port=ports['http'])
            args['served-model-name'] = dep['model']['served_name']
            args['device-ids'] = ','.join(map(str, placement['devices']))
            if launch.get('preserve_reference'):
                env = copy.deepcopy(launch.get('env',{}))
                visible=','.join(map(str, placement['devices']))
                if 'ASCEND_RT_VISIBLE_DEVICES' in env and str(env['ASCEND_RT_VISIBLE_DEVICES'])!=visible:
                    raise ValueError('Confirmed visible devices conflict with placement: '+item['id'])
                env['ASCEND_RT_VISIBLE_DEVICES']=args.pop('device-ids')
            if 'rpc' in ports:
                for key,value in {'data-parallel-rpc-port':ports['rpc'],'data-parallel-address':node['network']['ip']}.items():
                    if key in args and str(args[key])!=str(value) and '{{' not in str(args[key]):
                        raise ValueError('Confirmed argument conflicts with binding: '+item['id']+'.'+key)
                args['data-parallel-rpc-port'] = ports['rpc']
                args['data-parallel-address'] = node['network']['ip']
            if 'hccl' in ports and not launch.get('preserve_reference'):
                env['HCCL_IF_BASE_PORT'] = str(ports['hccl'])
            if item['role'] in ['prefill', 'decode']:
                link = next(x for x in dep['connections']['kv_transfer'] if item['id'] in [x['producer'], x['consumer']])
                p = instances[link['producer']]['launch']['args']
                d = instances[link['consumer']]['launch']['args']
                args['kv-transfer-config'] = {'kv_connector': link['connector'],
                    'kv_role': 'kv_producer' if item['role'] == 'prefill' else 'kv_consumer',
                    'kv_port': str(ports['kv']), 'engine_id': launch.get('engine_id',dep['id']+'-'+item['id']),
                    'kv_connector_extra_config': {'ascend_local_comm_res_path': '{{paths.comm}}',
                        'prefill': {'dp_size': p.get('data-parallel-size', 1), 'tp_size': p.get('tensor-parallel-size', 1)},
                        'decode': {'dp_size': d.get('data-parallel-size', 1), 'tp_size': d.get('tensor-parallel-size', 1)}}}
                if link['connector']=='MooncakeConnectorV2':
                    # This connector only loads endpoint files when explicitly
                    # configured. Do not force legacy endpoint-file semantics.
                    args['kv-transfer-config']['kv_connector_extra_config'].pop('ascend_local_comm_res_path')
                if 'kv_transfer_config' in launch:
                    exact=copy.deepcopy(launch['kv_transfer_config'])
                    for key in ('kv_connector','kv_role','kv_port','engine_id'):
                        expected=args['kv-transfer-config'][key]
                        if str(exact[key])!=str(expected) and '{{' not in str(exact[key]):
                            raise ValueError('Confirmed KV configuration conflicts with topology: '+item['id']+'.'+key)
                    args['kv-transfer-config']=exact
            argv = ['vllm', 'serve', '{{model.path}}']+cli_args(args)
        group['commands'].append({'name': item['id'], 'role':item['role'], 'argv': argv, 'env': env,
                                  'bootstrap': launch.get('bootstrap', '')})
    prefill = next(i for i in instances.values() if i['role'] in ['prefill', 'mixed'])
    endpoint = dep['endpoint']
    return {'groups': groups, 'endpoint': url(endpoint['instance'], endpoint['port']),
        'endpoint_group': instances[endpoint['instance']]['placement']['node']+'/model',
        'reset_urls': [url(i['id'])+i['checks']['prefix_cache_reset'] for i in instances.values() if i['role'] != 'proxy'],
        'metrics_urls': [url(i['id'])+i['checks']['metrics'] for i in instances.values() if i['role'] != 'proxy'],
        'metric_instances': [{'id':i['id'],'role':i['role'],'node':i['placement']['node'],
            'url':url(i['id'])+i['checks']['metrics'],
            'dp':i['launch']['args'].get('data-parallel-size',1)}
            for i in instances.values() if i['role']!='proxy'],
        'prefill_url': url(prefill['id']), 'prefill_dp': prefill['launch']['args'].get('data-parallel-size', 1)}
