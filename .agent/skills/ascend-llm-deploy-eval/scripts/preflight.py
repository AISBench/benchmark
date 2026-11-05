"""Verify user-selected resources and prepared inputs without mutating services."""

if __name__ == '__main__':
    raise SystemExit('Retired lifecycle entry. Use scripts/run.py --mode FULL|scripts; historical frozen jobs retain their own code.')
import json
from pathlib import Path
import re
import socket
import time


class PortUnavailable(RuntimeError):
    pass


def probe_port(port, timeout=60):
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", port))
            return
        except OSError as exc:
            if exc.errno not in (98, 10048) or time.monotonic() >= deadline:
                raise PortUnavailable(f"Port preflight failed: address=0.0.0.0 port={port} errno={exc.errno}") from exc
            time.sleep(1)
from stage_runtime import command, entry, read_plan, ready, save, state


CONTAINER_TMP = '/bench-tmp'


def check_ipc_path(directory=CONTAINER_TMP):
    length = len((directory.rstrip('/') + '/' + '0'*36).encode('utf-8'))
    if length > 107:
        raise ValueError(f'IPC socket path too long: {length} bytes > 107')
    return length


def _check(root, report):
    s=read_plan(root)
    evidence={}
    model=Path(s['model']['path'])
    service=s['deployment']['service']
    errors={}
    try:
        for filename in service.get('kv_endpoint_files',[]):
            config=json.loads(Path(filename).read_text())
            if not config.get('endpoint_list'):
                raise ValueError('Empty KV endpoint configuration: '+filename)
        report['checks']['kv_endpoints']='PASSED' if service.get('kv_endpoint_files') else 'NOT_APPLICABLE'
    except Exception as exc:
        report['checks']['kv_endpoints']='FAILED'
        errors['kv_endpoints']=repr(exc)
    try:
        evidence['ipc_path_bytes'] = check_ipc_path()
        if any(m['target'] in ['/bench-tmp','/bench-cache'] for m in s['container']['mounts']):
            raise ValueError('Reserved runtime mount target')
        report['checks']['ipc_path']='PASSED'
    except Exception as exc:
        report['checks']['ipc_path']='FAILED'
        errors['ipc_path']=repr(exc)
    try:
        report['checks']['storage']='PASSED'
        service=s['deployment']['service']
        if s.get('storage'):
            from disk_guard import check as disk_check
            report['storage']=disk_check(s['storage']['root'], s['storage']['subtasks'], admission=True)
        else:
            report['checks']['storage']='NOT_APPLICABLE'
        save(root/'scripts/service.json',service)
    except Exception as exc:
        report['checks']['storage']='FAILED'
        errors['storage']=repr(exc)
    try:
        report['checks']['image']='PASSED'
        evidence['image_id']=json.loads(command(['docker','image','inspect',s['target']['image']]))[0]['Id']
        save(root/'preflight-evidence.json',evidence)
    except Exception as exc:
        report['checks']['image']='FAILED'
        errors['image']=repr(exc)
    try:
        report['checks']['mounts']='PASSED'
        missing=[m['source'] for m in s['container']['mounts'] if not Path(m['source']).exists()]
        if missing: raise ValueError('Missing mounts: '+json.dumps(missing))
    except Exception as exc:
        report['checks']['mounts']='FAILED'
        errors['mounts']=repr(exc)
    try:
        report['checks']['devices']='PASSED'
        missing=[d for d in s['container']['devices'] if not Path(d).exists()]
        if missing: raise ValueError('Missing devices: '+json.dumps(missing))
    except Exception as exc:
        report['checks']['devices']='FAILED'
        errors['devices']=repr(exc)
    try:
        report['checks']['weights']='PASSED'
        model=Path(s['model']['path'])
        if not model.is_dir(): raise ValueError('Missing model directory')
        index=model/'model.safetensors.index.json'
        if index.exists():
            shards=set(json.loads(index.read_text())['weight_map'].values())
            for name in shards:
                path=(model/name).resolve()
                if not path.is_relative_to(model.resolve()) or not path.is_file(): raise ValueError('Missing weight shard: '+name)
            evidence['weight_shards']=len(shards)
    except Exception as exc:
        report['checks']['weights']='FAILED'
        errors['weights']=repr(exc)
    try:
        report['checks']['tokenizer']='PASSED'
        for name in ['tokenizer.json','tokenizer_config.json']:
            if not (model/name).is_file(): raise ValueError('Missing '+name)
    except Exception as exc:
        report['checks']['tokenizer']='FAILED'
        errors['tokenizer']=repr(exc)
    try:
        report['checks']['benchmark_tool']='PASSED'
        benchmark=s['benchmark']
        if not benchmark.get('executable'):
            raise ValueError('Image must provide ais_bench CLI')
        from aisbench_cli import dataset_for, check_dataset
        if not benchmark.get('datasets'): raise ValueError('Missing benchmark.datasets')
        for key in benchmark['datasets']: dataset_for(benchmark, {'dataset_id':key})
        for case in benchmark.get('cases', []):
            check_dataset(dataset_for(benchmark,case), case['formal_requests'])
    except Exception as exc:
        report['checks']['benchmark_tool']='FAILED'
        errors['benchmark_tool']=repr(exc)
    try:
        report['checks']['network']='PASSED'
        network=service.get('network')
        if network:
            from task_preflight import network_addresses
            addresses=network_addresses('dev',network['interface'])
            if not any(x.get('local')==network['ip'] for i in addresses for x in i.get('addr_info',[])):
                raise ValueError('Configured network address absent')
            evidence['network']=addresses
        else:
            report['checks']['network']='NOT_APPLICABLE'
    except Exception as exc:
        report['checks']['network']='FAILED'
        errors['network']=repr(exc)
    try:
        report['checks']['resource_or_reuse']='PASSED'
        if ready(root,s):
            evidence['reuse']='READY instance verified'
        else:
            if state(root)['service'] not in ['ABSENT','RELEASED']:
                raise ValueError('Existing service not READY; inspect/finalize before deployment')
            npu=command(['npu-smi','info']); evidence['npu']=npu
            free={int(x) for x in re.findall(r'No running processes found in NPU (\d+)',npu)}
            if not set(service['npu_ids'])<=free:
                save(root/'preflight-evidence.json',evidence)
                raise ValueError('Requested NPUs occupied or state unknown; do not touch other jobs')
            for port in service['ports']:
                try:
                    probe_port(port)
                except PortUnavailable as exc:
                    try:
                        evidence['port_conflict'] = command(['ss', '-ltnp', 'sport', '=', ':'+str(port)])
                    except Exception as detail:
                        evidence['port_conflict'] = repr(detail)
                    raise
            evidence['free_ports']=service['ports']
    except Exception as exc:
        report['checks']['resource_or_reuse']='FAILED'
        errors['resource_or_reuse']=repr(exc)
        port_failure = isinstance(exc, PortUnavailable)
    report['errors']=errors
    save(root/'preflight-evidence.json',evidence)
    if errors:
        if locals().get('port_failure') and set(errors)=={'resource_or_reuse'}:
            raise PortUnavailable('Preflight findings: '+json.dumps(errors))
        raise ValueError('Preflight findings: '+json.dumps(errors))
    report['current']=None
    save(root/'preflight-evidence.json',evidence)
    return {'status':'PASSED','message':'Selected dependencies and service/resources verified','artifacts':[str(root/'preflight-evidence.json')]}

def check(root):
    from datetime import datetime, timezone
    report={'time':datetime.now(timezone.utc).isoformat(), 'current':None,
            'checks':{name:'NOT_RUN' for name in ('storage','image','mounts','devices','weights','tokenizer','benchmark_tool','network','resource_or_reuse')}}
    try:
        result=_check(root, report)
        report['status']='PASSED'
        return result
    except Exception as exc:
        report['status']='FAILED'; report['error']=repr(exc)
        if report['current']: report['checks'][report['current']]='FAILED'
        raise
    finally:
        try: save(root/'preflight-report.json',report)
        except OSError: pass


if __name__=='__main__': raise SystemExit(entry('preflight'))
