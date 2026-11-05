"""Bind normalized task fields to this host before the serial loop starts."""
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import shlex
import shutil
import socket
import secrets
import subprocess
from stage_runtime import PACKAGE, command, digest, save

TOKEN = re.compile(r'\{\{([a-zA-Z_][a-zA-Z0-9_.]*)\}\}')


def lookup(context, name):
    value = context
    for key in name.split('.'):
        if not isinstance(value, dict) or key not in value:
            raise ValueError('Unresolved binding: '+name)
        value = value[key]
    return value


def expand(value, context, shell=False):
    if isinstance(value, dict):
        return {k: expand(v, context, shell) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, context, shell) for v in value]
    if not isinstance(value, str):
        return value
    match = TOKEN.fullmatch(value)
    if match and not shell:
        return lookup(context, match[1])
    def replace(match):
        result = lookup(context, match[1])
        if isinstance(result, (dict, list)):
            raise ValueError('Structured binding must occupy entire value: '+match[1])
        return shlex.quote(str(result)) if shell else str(result)
    result = TOKEN.sub(replace, value)
    if '{{' in result or '}}' in result:
        raise ValueError('Unresolved or invalid template token')
    return result


def interface_speed(name):
    """Read negotiated link speed in Mbps, never the maximum advertised mode."""
    if not re.fullmatch(r'[A-Za-z0-9_.:-]+', name):
        return None
    try:
        speed = int((Path('/sys/class/net')/name/'speed').read_text().strip())
        if speed > 0:
            return speed
    except (OSError, ValueError):
        pass
    try:
        match = re.search(r'^\s*Speed:\s*(\d+)Mb/s\s*$', command(['ethtool', name]), re.M)
        if match and int(match[1]) > 0:
            return int(match[1])
    except (OSError, RuntimeError):
        pass
    return None


def network_addresses(*args):
    errors = []
    for executable in dict.fromkeys(['ip', '/usr/sbin/ip', '/sbin/ip']):
        try:
            return json.loads(command([executable, '-d', '-j', 'address', 'show', *args]))
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(str(exc))
    raise RuntimeError('No usable ip executable: '+str(errors))


def discover_network(preference=None):
    interfaces = network_addresses()
    choices = []
    for interface in interfaces:
        name = interface['ifname']
        kind = interface.get('linkinfo', {}).get('info_kind')
        flags = interface.get('flags', [])
        if (interface.get('operstate') != 'UP' or name == 'lo' or 'NO-CARRIER' in flags
                or kind in ['bridge', 'veth', 'dummy', 'tun', 'vxlan', 'geneve']):
            continue
        addresses = [a['local'] for a in interface.get('addr_info', [])
                     if a.get('family') == 'inet' and a.get('scope') == 'global'
                     and not any(a.get(k) for k in ['tentative', 'dadfailed', 'deprecated'])]
        if not addresses:
            continue
        speed = interface_speed(name)
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if ip.version == 4 and not (ip.is_loopback or ip.is_link_local or ip.is_unspecified):
                choices.append({'interface': name, 'ip': address, 'speed_mbps': speed})
    if preference:
        candidates = [c for c in choices if c['interface'] == preference]
        reason = 'explicit_verified_interface'
    elif choices and all(c['speed_mbps'] is not None for c in choices):
        fastest = max(c['speed_mbps'] for c in choices)
        candidates = [c for c in choices if c['speed_mbps'] == fastest]
        reason = 'highest_negotiated_speed_with_usable_ipv4'
    else:
        candidates = []
        reason = 'unknown_link_speed'
    if len(candidates) != 1:
        raise ValueError('Network selection requires attended verification (tie, missing speed/IP or invalid override): '
                         +json.dumps(choices))
    return {**candidates[0], 'selection_reason': reason, 'candidates': choices}


def allocate_ports(requests):
    """Allocate disjoint ranges, holding sockets until all ranges are selected."""
    sockets, bases, all_ports = [], {}, []
    try:
        for name, width in requests.items():
            for attempt in range(100):
                first = socket.socket()
                # Keep service ranges outside the host's ephemeral client-port range.
                try:
                    ephemeral_start=int(Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split()[0])
                except (OSError,ValueError):
                    ephemeral_start=32768
                ceiling=min(ephemeral_start,32768)-width
                if ceiling<=1024:
                    first.close()
                    raise ValueError('No non-ephemeral service port range available')
                base=1024+secrets.randbelow(ceiling-1024)
                try:
                    first.bind(('0.0.0.0',base))
                except OSError:
                    first.close()
                    continue
                opened = [first]
                try:
                    if base + width > 65536:
                        raise OSError('Port range exceeds limit')
                    for port in range(base+1, base+width):
                        sock = socket.socket()
                        opened.append(sock)
                        sock.bind(('0.0.0.0', port))
                except OSError:
                    for sock in opened:
                        sock.close()
                    continue
                sockets.extend(opened)
                bases[name] = base
                all_ports.extend(range(base, base+width))
                break
            else:
                raise RuntimeError('Unable to allocate port range '+name)
        return bases, all_ports
    finally:
        for sock in sockets:
            sock.close()


def inspect_image_tools(image, job_id):
    name = job_id+'-000'
    existing = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
    if existing.returncode == 0:
        raise ValueError('Probe name already exists; do not reuse '+name)
    code = (PACKAGE/'scripts/discover_tool.py').read_text(encoding='utf-8-sig')
    try:
        output = command(['docker', 'run', '--rm', '--name', name,
            '--label', 'benchmark.job_id='+job_id, '--label', 'benchmark.role=preflight-probe',
            '--network=none', '--entrypoint', 'python3', image, '-c', code], timeout=120)
        return json.loads(output)
    finally:
        remaining = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
        if remaining.returncode == 0:
            container = json.loads(remaining.stdout)[0]
            labels = container['Config'].get('Labels') or {}
            if labels.get('benchmark.job_id') != job_id or labels.get('benchmark.role') != 'preflight-probe' or container['Image'] != image:
                raise ValueError('Probe ownership changed; refuse cleanup')
            command(['docker', 'rm', '-f', container['Id']])
            if subprocess.run(['docker', 'inspect', container['Id']], capture_output=True).returncode == 0:
                raise RuntimeError('Probe cleanup not verified')


def create_execution(spec, dep, root, index, bindings, network, port_bases, ports, tools, image):
    child = root/'services'/dep['id']
    for name in ['scripts', 'logs', 'results', 'configs']:
        (child/name).mkdir(parents=True, exist_ok=True)
    model_path = Path(bindings['model_paths'][dep['model']['id']])
    if not model_path.is_absolute() or not model_path.is_dir():
        raise ValueError('Missing model location for '+dep['model']['id'])
    model = {k: v for k, v in dep['model'].items() if k != 'id'}
    model['path'] = str(model_path.resolve())
    context = {'model': model, 'network': network, 'ports': port_bases,
               'paths': dict(bindings.get('paths', {})), 'run_dir': str(child), 'task_dir': str(root)}
    if '{{paths.proxy_program}}' in json.dumps(dep['commands']):
        proxy=Path(context['paths'].get('proxy_program', ''))
        if not proxy.is_absolute() or not proxy.is_file():
            raise ValueError('PD deployment requires bindings.paths.proxy_program pointing to an existing reviewed proxy')
    commands = []
    kv_endpoint_files=[]
    for item in dep['commands']:
        argv = []
        for arg in item['argv']:
            argv.append(json.dumps(expand(arg['json'], context), ensure_ascii=False) if isinstance(arg, dict) else str(expand(arg, context)))
        env = expand(item.get('env', {}), context)
        if '--kv-transfer-config' in argv:
            kv=json.loads(argv[argv.index('--kv-transfer-config')+1])
            directory=kv.get('kv_connector_extra_config',{}).get('ascend_local_comm_res_path')
            if directory and kv.get('kv_connector')=='MooncakeConnectorV2':
                device_ids=argv[argv.index('--device-ids')+1].split(',')
                kv_endpoint_files.extend(str(Path(directory)/('ub_endpoint_npu_'+device+'.json')) for device in device_ids)
        if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', k) for k in env):
            raise ValueError('Invalid environment variable name')
        bootstrap = expand(item.get('bootstrap', ''), context, shell=True)
        script = child/'scripts'/('start-'+item['name']+'.sh')
        script.write_text('#!/usr/bin/env bash\nset -e\n'+bootstrap+'\n'+
            ''.join('export '+k+'='+shlex.quote(str(v))+'\n' for k, v in env.items())+
            'exec '+shlex.join(argv)+'\n')
        commands.append({'name': item['name'], 'role':item.get('role','model'), 'argv': ['bash', str(script)]})
    service = expand(dep['service'], context)
    service.update(commands=commands, npu_ids=dep['npu_ids'], ports=ports, network=network)
    service['kv_endpoint_files']=kv_endpoint_files
    if not all(isinstance(p, int) and p in ports for p in service['reset_ports']+[service['prefill_port']]):
        raise ValueError('Service ports must reference allocated ranges')
    for url in service['health_urls']+[service['endpoint']]+([service['gateway_health_url']] if service.get('gateway_health_url') else []):
        from urllib.parse import urlsplit
        parsed = urlsplit(url)
        if parsed.scheme != 'http' or parsed.hostname not in ['127.0.0.1', 'localhost', network['ip']] or parsed.port not in ports:
            raise ValueError('Service URL must use allocated local port: '+url)
    from container_baseline import baseline_mounts, DEVICES, DEFAULT_CONTAINER
    mounts = baseline_mounts(bindings.get('mounts', []))
    pool=spec.get('container_pool')
    locations=[(pool['task_root'],False)]+[(str(Path(v).resolve()),True) for v in pool['model_paths']] if pool else [(str(root),False),(str(model_path.resolve()),True)]
    for path, read_only in locations:
        matching = [m for m in mounts if m['source'] == m['target'] and Path(path).is_relative_to(Path(m['source']))]
        if not read_only and any(m['read_only'] for m in matching):
            raise ValueError('Task workspace must be writable in container')
        if not matching:
            mounts.append({'source': path, 'target': path, 'read_only': read_only})
    if len({m['target'] for m in mounts}) != len(mounts):
        raise ValueError('Duplicate mount targets')
    for resource in [d.get(field) for d in bindings.get('benchmark', {}).get('datasets', {}).values()
                     for field in ('path','generation_manifest')]+[bindings.get('paths', {}).get('proxy_program')]:
        if resource and Path(resource).is_file() and not any(Path(resource).is_relative_to(Path(m['source'])) and m['source']==m['target'] for m in mounts):
            mounts.append({'source': resource, 'target': resource, 'read_only': True})
    if len({m['target'] for m in mounts}) != len(mounts):
        raise ValueError('Benchmark resource mount conflicts with existing target')
    devices = list(dep['container'].get('devices', DEVICES))
    required_cards = ['/dev/davinci'+str(i) for i in (pool['devices'] if pool else dep['npu_ids'])]
    if not set(required_cards).issubset(devices):
        raise ValueError('Container devices do not cover assigned model devices')
    container = {**DEFAULT_CONTAINER, **dep['container'], 'devices': devices, 'mounts': mounts}
    generated=list((child/'scripts').glob('start-*.sh'))
    if (child/'scripts/proxy.py').exists():
        generated.append(child/'scripts/proxy.py')
    fingerprints = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in generated}
    fingerprint = digest({'image': image, 'model': model, 'container': container,
                          'service': service, 'scripts': fingerprints})
    job_id = spec['job_id']+f'-{index:03}'
    execution = {'schema_version': 'resolved-1', 'job_id': job_id,
        'target': {'host': spec['target']['host'], 'container': job_id, 'work_dir': str(child), 'image': image},
        'model': model, 'container': container, 'deployment': {'service': service},
        'service_fingerprint': fingerprint,
        'benchmark': {**bindings.get('benchmark', {}), 'adapter': 'aisbench_cli', 'cases': spec.get('benchmark_cases', []),
                      'executable': tools.get('executable'),
                      'prefill_dp': dep['prefill_dp']},
        'execution': spec['execution'], 'cleanup': spec['cleanup']}
    if 'storage' in spec: execution['storage'] = spec['storage']
    if pool:
        execution['job_id']=pool['job_id']
        execution['parent_job_id']=pool.get('parent_job_id',pool['job_id'])
        execution['node_id']=pool.get('node_id')
        execution['target']['container']=pool['name']
        execution['container_fingerprint']=digest({'job_id':pool['job_id'],'plan':pool['plan_sha256'],'image':image,'container':container})
        host_key = hashlib.sha256(spec['target']['host'].encode()).hexdigest()[:16]
        execution['container_registry']=str(Path(pool['task_root'])/'container-pools'/host_key/(pool['name']+'.json'))
    save(child/'resolved.json', execution)
    save(child/'plan.json', {'execution_spec': execution, 'execution_sha256': digest(execution)})
    save(child/'scripts/service.json', service)
    return {'run_dir': str(child), 'execution_sha256': digest(execution),
            'service_fingerprint': fingerprint, 'scripts_sha256': fingerprints, 'tool_discovery': tools}
