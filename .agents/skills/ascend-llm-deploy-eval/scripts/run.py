"""Compact task entry. Generate once; FULL and manual execution use the same scripts.

Bindings must contain explicit network/ports for each deployment on every node.
This entry never imports or mutates historical frozen task directories.
"""
from mount_profiles import resolve_mounts
import argparse
import ast
import base64
import copy
import csv
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import sys
import shutil
import subprocess
import time
import urllib.request
import uuid
import tempfile
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, wait

from aisbench_cli import commands, require_supported_mode, write_config
from container_baseline import docker_run_args
from task_model import load_json, normalize, validate, validate_bindings, expand, higher_concurrency_blocker
from remote_node import transport_argv, is_local
from topology import compile_deployment
from shared_storage import verify_groups
from bundle_distribution import PROGRAM as DISTRIBUTE_PROGRAM


def save(path, value):
    if path.name=='status.json':
        from jsonschema import validate as schema_validate
        schema_validate(value,load_json(Path(__file__).parent.parent/'schemas/status.schema.json'))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def script(path, lines, *, strict=True):
    if path.exists():
        raise ValueError('Generated script path collision: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/usr/bin/env bash\n'+('set -euo pipefail' if strict else 'set -e')+'\n'+lines+'\n', encoding='utf-8', newline='\n')


def remote_root(binding, job_id):
    path = PurePosixPath(binding['work_dir'])/job_id
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('work_dir must be an absolute Linux path')
    return str(path)


def port_check_program(ports):
    # This is an availability check, not a reservation. A competing process can
    # bind after it exits; startup/health must still fail safely on conflicts.
    # Linux server restarts may leave TIME_WAIT after all owned processes exit.
    # REUSEADDR permits that normal restart, but deliberately never REUSEPORT.
    # listen also rejects an existing listener when both sockets use REUSEADDR.
    return ('import socket; sockets=[]\nfor port in '+repr(ports)+':\n'
            ' s=socket.socket(); sockets.append(s)\n'
            ' s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)\n'
            ' try: s.bind(("0.0.0.0",port)); s.listen(1)\n'
            ' except OSError as exc: raise RuntimeError("TCP port %s unavailable: %s" % (port,exc)) from exc')


def proxy_health_gate(urls, timeout):
    program = """import subprocess,time,sys,json
urls=json.loads(sys.argv[1])
deadline=time.monotonic()+float(sys.argv[2])
while True:
    healthy=True
    for url in urls:
        remaining=deadline-time.monotonic()
        if remaining<=0:
            sys.exit('Proxy startup blocked: model health timeout: '+url)
        limit=min(3.0,remaining)
        try:
            probe=subprocess.run(['curl','--noproxy','*','--silent','--show-error','--output','/dev/null','--write-out','%{http_code}','--max-time',str(limit),url],capture_output=True,text=True,timeout=limit)
            ok=probe.returncode==0 and probe.stdout=='200'
        except subprocess.TimeoutExpired:
            ok=False
        if not ok: healthy=False
    if healthy:
        print('Proxy model health dependencies ready',flush=True)
        break
    remaining=deadline-time.monotonic()
    if remaining<=0: sys.exit('Proxy startup blocked: model health timeout')
    time.sleep(min(1.0,remaining))
"""
    return shlex.join(['python3','-c',program,json.dumps(urls),str(timeout)])+' || exit $?\n'


def generate(spec, bindings, root):
    """Transactional generation: failure never leaves a partially executable task."""
    if root.exists():
        raise ValueError('Refusing to overwrite an existing task directory')
    root.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.generate-',dir=root.parent) as temporary:
        staging=Path(temporary)/'task'
        result=_generate(spec,bindings,staging)
        for attempt in range(4):
            try:
                staging.rename(root)
                break
            except PermissionError:
                if os.name!='nt' or attempt==3:raise
                time.sleep(0.1*(attempt+1))
    return result


def _generate(spec, bindings, root):
    """No SSH, Docker or model side effects; all unresolved bindings fail closed."""
    spec = copy.deepcopy(spec)
    validate_bindings(bindings)
    for dep in spec['deployments']:
        if dep['id'].lower()=='tests':
            raise ValueError('Deployment ID tests is reserved for generated test scripts')
    for dep in spec['deployments']:
        for instance in dep['instances']:
            if instance['role'] != 'proxy':
                if instance['launch'].get('preserve_reference') is False:
                    raise ValueError('Only preserve_reference launch is supported')
                instance['launch']['preserve_reference'] = True
    spec = normalize(spec)
    for slot,binding in bindings['nodes'].items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',slot):
            raise ValueError('Invalid physical node ID')
        if not re.fullmatch(r'(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9_.-]*',binding['host']):
            raise ValueError('Invalid SSH host')
    errors = validate(spec)
    if errors:
        raise ValueError('; '.join(errors))
    # Node aliases must not hide physical resource overlap between parallel branches.
    tasks_by_id={s['id']:s for s in spec['sub_tasks']}
    deps_by_id={d['id']:d for d in spec['deployments']}
    for stage in spec.get('stages',[]):
        reserved=set()
        for branch in stage['branches']:
            hosts={bindings['nodes'][i['placement']['node']]['host'].split('@')[-1].lower()
                   for sid in branch['sub_tasks']
                   for i in deps_by_id[tasks_by_id[sid]['deployment_id']]['instances']}
            if hosts & reserved: raise ValueError('Parallel branches share a bound host')
            reserved.update(hosts)
    if spec['state'] != 'ready':
        raise ValueError('Resolve draft before generating executable scripts')
    for case in spec['cases']:
        require_supported_mode(case)
    if spec['cleanup']['task_end'] != 'stop_container' or any(s['after_case']=='stop_service' for s in spec['sub_tasks']):
        raise ValueError('Compact runner requires task_end=stop_container; after_case supports keep_service/stop_container')
    if root.exists() and any(root.iterdir()):
        raise ValueError('Refusing to overwrite an existing task directory')
    root.mkdir(parents=True, exist_ok=True)
    save(root/'job_spec.json', spec)
    save(root/'bindings.json', bindings)
    helper = root/'.runtime'
    helper.mkdir(exist_ok=True)
    for filename in ('ownership_evidence.py','container_dependencies.py','aisbench_cli.py','validate_case.py','cache_reset.py','metric_evidence.py','test_evidence.py'):
        (helper/filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    (helper/'check_test.py').write_text('import sys\nfrom test_evidence import check\ncheck(*sys.argv[1:])\n',encoding='utf-8')
    plan = {'deployments': {}, 'tests': {}}
    resource_names=set()
    dependency_env={}
    for dep in spec['deployments']:
        dep = copy.deepcopy(dep)
        for instance in dep['instances']:
            if instance['role'] != 'proxy':
                if instance['launch'].get('preserve_reference') is False:
                    raise ValueError('Only preserve_reference launch is supported')
                instance['launch']['preserve_reference'] = True
        nodes = {slot: bindings['nodes'][slot]['deployments'][dep['id']]
                 for slot in {i['placement']['node'] for i in dep['instances']}}
        # Resolve explicitly authored values for comparison, never overwrite a conflicting template.
        for instance in dep['instances']:
            slot=instance['placement']['node'];node=nodes[slot];binding=bindings['nodes'][slot]
            ports=node['ports'][instance['id']]
            context={'network':node['network'],'ports':ports,'paths':binding.get('paths',{}),
                     'model':{'path':binding['model_paths'][dep['model']['id']],'served_name':dep['model']['served_name']}}
            expected={'HCCL_IF_IP':node['network']['ip'],'VLLM_HOST_IP':node['network']['ip'],
                'GLOO_SOCKET_IFNAME':node['network']['interface'],'TP_SOCKET_IFNAME':node['network']['interface'],
                'HCCL_SOCKET_IFNAME':node['network']['interface']}
            if 'hccl' in ports:expected['HCCL_IF_BASE_PORT']=ports['hccl']
            for key,value in instance['launch'].get('env',{}).items():
                if key in expected and str(expand(value,context))!=str(expected[key]):
                    raise ValueError('Confirmed environment conflicts with binding: '+instance['id']+'.'+key)
            for key,value in instance['launch'].get('args',{}).items():
                target={'data-parallel-address':node['network']['ip'],'data-parallel-rpc-port':ports.get('rpc')}
                if key in target and str(expand(value,context))!=str(target[key]):
                    raise ValueError('Confirmed argument conflicts with binding: '+instance['id']+'.'+key)
            if 'kv_transfer_config' in instance['launch']:
                instance['launch']['kv_transfer_config']=expand(instance['launch']['kv_transfer_config'],context)
        compiled = compile_deployment(dep, nodes)
        record = {'groups': {}, 'health_urls': [], 'endpoint': compiled['endpoint'], 'reset_urls':compiled['reset_urls'], 'metric_instances':compiled['metric_instances']}
        plan['deployments'][dep['id']] = record
        dependency_env[dep['id']]={}
        for group in compiled['groups'].values():
            slot = group['node']; binding = bindings['nodes'][slot]
            task = remote_root(binding, spec['job_id'])
            name = spec['job_id']+'-'+dep['id']+'-'+slot
            if name in resource_names:
                raise ValueError('Deployment/node resource identity collision: '+name)
            resource_names.add(name)
            q = shlex.quote
            owner = spec['job_id']
            identity = task+'/.runtime/'+dep['id']+'-'+slot+'.container-id'
            token_file = identity+'.token'
            guard = 'test "$(docker inspect -f '+q('{{ index .Config.Labels "bench.task" }}')+' '+q(name)+')" = '+q(owner)+ '\n'+\
                    'test "$(docker inspect -f '+q('{{.Id}}')+' '+q(name)+')" = "$(cat '+q(identity)+')"\n'+\
                    'test "$(docker inspect -f '+q('{{ index .Config.Labels "bench.execution" }}')+' '+q(name)+')" = "$(cat '+q(token_file)+')"'
            folder = root/'scripts'/dep['id']
            remote_scripts = task+'/scripts/'+dep['id']
            mounts = resolve_mounts(binding)
            model = binding['model_paths'][dep['model']['id']]
            for source in (task, model):
                matching=[m for m in mounts if m['target']==source]
                if matching and (matching[0]['source']!=source or (source==task and matching[0].get('read_only'))):
                    raise ValueError('Explicit mount conflicts with task/model binding: '+source)
                if not matching:
                    mounts.append({'source': source, 'target': source, 'read_only': source == model})
            if len({m['target'] for m in mounts}) != len(mounts):
                raise ValueError('Duplicate mount targets: '+slot)
            argv = ['docker', 'run', '-d', '--name', name, '--label', 'bench.task='+owner,
                    '--log-driver', 'none']+docker_run_args(dep['runtime']['container'])
            checks = []
            for device in dep['runtime']['container']['devices']:
                checks.append('test -e '+q(device)); argv += ['--device', device]
            for mount in mounts:
                checks.append('test -e '+q(mount['source']))
                argv += ['-v', mount['source']+':'+mount['target']+(':ro' if mount.get('read_only') else '')]
            argv += [dep['runtime']['image'], 'sleep', 'infinity']
            checks.insert(0,'if docker inspect '+q(name)+' >/dev/null 2>&1; then echo "Container already exists" >&2; exit 1; fi')
            checks.insert(0,'docker info >/dev/null')
            checks.append('npu-smi info')
            checks.append('command -v fuser >/dev/null')
            for device in group['npu_ids']:
                checks.append('RC=0; fuser '+q('/dev/davinci'+str(device))+' || RC=$?\ntest "$RC" -eq 1')
            port_check=port_check_program(group['ports'])
            checks.append(shlex.join(['python3','-c',port_check]))
            create=shlex.join(argv).replace('docker run ', 'docker run --label "bench.execution=$TOKEN" ',1)
            script(folder/(slot+'-container.sh'), '\n'.join(checks)+'\nTOKEN=${1:-$(python3 -c "import uuid; print(uuid.uuid4().hex)")}\n[[ "$TOKEN" =~ ^[a-f0-9]{32}$ ]]\nprintf "%s" "$TOKEN" > '+q(token_file)+'\n'+create+' > '+q(identity)+'\ncat '+q(identity))
            script(folder/(slot+'-stop.sh'),
                   'docker info >/dev/null\nif docker inspect '+q(name)+' >/dev/null 2>&1; then\n'+
                   'TOKEN=${1:-$(cat '+q(token_file)+')}\n[[ "$TOKEN" =~ ^[a-f0-9]{32}$ ]]\n'+
                   'if test "$(docker inspect -f '+q('{{ index .Config.Labels "bench.execution" }}')+' '+q(name)+')" != "$TOKEN"; then echo "Foreign execution; untouched"; exit 0; fi\n  '+
                   'test "$(docker inspect -f '+q('{{ index .Config.Labels "bench.task" }}')+' '+q(name)+')" = '+q(owner)+'\n  docker rm -f '+q(name)+'\nfi\n'+
                   'if docker inspect '+q(name)+' >/dev/null 2>&1; then exit 1; fi')
            starts = [guard, 'RUN=${1:-run01}', '[[ "$RUN" =~ ^run[0-9]+$ ]]',
                      'mkdir -p '+q(task+'/logs/'+dep['id']+'-')+'"$RUN"',
                      'printf "%s" "$RUN" > '+q(task+'/.runtime/'+dep['id']+'-'+slot+'.run')]
            probes = [guard,'RUN=$(cat '+q(task+'/.runtime/'+dep['id']+'-'+slot+'.run')+')','[[ "$RUN" =~ ^run[0-9]+$ ]]']
            ownership = ('python3 '+q(task+'/.runtime/ownership_evidence.py')+' '+
                         q(task+'/logs/'+dep['id']+'-')+'"$RUN"/'+q(slot+'-ownership.jsonl')+' '+
                         ' '.join(str(int(device)) for device in group['npu_ids'])+' || true')
            starts.append(ownership)
            probes.append(ownership)
            dependency_env[dep['id']][slot]={}
            for item in group['commands']:
                context = {'model': {'path': model, 'served_name':dep['model']['served_name']},
                           'paths':binding.get('paths', {}), 'network':nodes[slot]['network'],
                           'ports':nodes[slot]['ports'][item['name']], 'task_dir':task,
                           'run_dir':task+'/logs/'+dep['id']+'-manual/'+slot+'-'+item['name']}
                args = [json.dumps(expand(a['json'], context)) if isinstance(a,dict) else str(expand(a,context)) for a in item['argv']]
                env = expand(item['env'],context)
                if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', k) for k in env):
                    raise ValueError('Invalid environment name')
                if item['role']!='proxy':dependency_env[dep['id']][slot][item['name']]={k:str(v) for k,v in env.items()}
                filename = slot+'-'+item['name']+'.sh'
                gate=proxy_health_gate(item['startup_health_urls'],spec['execution']['startup_timeout_seconds']) if item['role']=='proxy' and spec['execution']['proxy_start_after_models_healthy'] else ''
                script(folder/filename, '# Execute inside container '+name+' on '+binding['host']+'\n'+expand(item['bootstrap'],context,shell=True)+'\n'+
                       '\n'.join('export '+k+'='+q(str(v)) for k,v in env.items())+'\n'+gate+('# @proxy-command\n' if item['role']=='proxy' else '')+'exec '+shlex.join(args), strict=False)
                pid = '/tmp/bench-'+slot+'-'+item['name']+'.pid'
                # The PID is written by the shell that execs the generated script.
                inner = 'echo $$ > '+q(pid)+'; exec bash '+q(remote_scripts+'/'+filename)
                log = task+'/logs/'+dep['id']+'-'
                starts.append('docker exec -d '+q(name)+' bash -c '+q(inner+' > "$1" 2>&1')+
                              ' bash '+q(log)+'"$RUN"/'+q(slot+'-'+item['name']+'.log'))
                probes.append(shlex.join(['docker','exec',name,'bash','-c',
                              'test -s '+q(pid)+' && kill -0 "$(cat '+q(pid)+')"']))
                probes.append('if grep -Eq '+q('EngineCore failed to start|Engine core initialization failed')+' '+
                              q(log)+'"$RUN"/'+q(slot+'-'+item['name']+'.log')+'; then echo "Fatal engine startup failure" >&2; exit 1; fi')
            dependency_log=task+'/logs/'+dep['id']+'-'
            script(folder/(slot+'-dependencies.sh'), guard+'\nRUN=${1:-run01}\n[[ "$RUN" =~ ^run[0-9]+$ ]]\nmkdir -p '+q(dependency_log)+'"$RUN"\n'+
                   shlex.join(['docker','exec',name,'python3',task+'/.runtime/container_dependencies.py',task,dep['id'],slot,str(spec['execution']['dependency_timeout_seconds'])])+
                   ' > '+q(dependency_log)+'"$RUN"/'+q(slot+'-dependencies.log')+' 2>&1')
            script(folder/(slot+'-start.sh'), '\n'.join(starts))
            script(folder/(slot+'-probe.sh'), '\n'.join(probes))
            record['groups'][slot] = {'host':binding['host'], 'root':task, 'container':name,
                'log_names':[slot+'-'+item['name']+'.log' for item in group['commands']], 'guard':guard}
            record['health_urls'] += group['health_urls']
        endpoint_slot=compiled['endpoint_group'].split('/')[0]
        record['endpoint_node']=endpoint_slot
        payload=json.dumps({'model':dep['model']['served_name'],'prompt':'Hello','max_tokens':1})
        script(root/'scripts'/dep['id']/'smoke.sh',shlex.join(['curl','--noproxy','*','--fail-with-body','--silent','--show-error',
            '--max-time',str(spec['execution']['smoke_timeout_seconds']),'-H','Content-Type: application/json','-d',payload,
            compiled['endpoint']+'/v1/completions']))
        for sub in (s for s in spec['sub_tasks'] if s['deployment_id']==dep['id']):
            case = next(c for c in spec['cases'] if c['id']==sub['case_id'])
            slot = compiled['endpoint_group'].split('/')[0]
            binding = bindings['nodes'][slot]; group = record['groups'][slot]
            folder = root/'scripts/tests'; folder.mkdir(parents=True,exist_ok=True)
            destination = folder/(sub['id']+'.config.py')
            if case.get('aisbench_config'):
                source=Path(binding['benchmark']['configs'][case['aisbench_config']])
                text=source.read_text(encoding='utf-8')
                if any(isinstance(n,ast.ImportFrom) and n.level for n in ast.walk(ast.parse(text))):
                    raise ValueError('Native config must use installed absolute imports, not relative file imports')
                destination.write_text(text,encoding='utf-8',newline='\n')
            else:
                config = write_config({'model':{'path':binding['model_paths'][dep['model']['id']],
                                  'served_name':dep['model']['served_name']},'benchmark':binding['benchmark']},
                                  case, {'endpoint':compiled['endpoint']}, folder, check_files=False)
                config.replace(destination)
                (folder/'dataset.json').unlink()
            target_config = group['root']+'/scripts/tests/'+destination.name
            text = destination.read_text(encoding='utf-8').replace(repr(str(folder/'outputs')), "'outputs'")
            destination.write_text(text,encoding='utf-8',newline='\n')
            argv = commands(binding['benchmark'].get('executable','ais_bench'), target_config,case,Path('OUTPUT'))[0][1]
            argv[argv.index('--work-dir')+1]='__RESULT__'
            # Shell result path is the sole run-time argument; all load settings are fixed.
            command = shlex.join(argv).replace('__RESULT__', '"$OUT/outputs"')
            script(folder/(sub['id']+'.sh'), group['guard']+'\nTRY=${1:-try01}\n[[ "$TRY" =~ ^try[0-9]+$ ]]\nOUT='+
                   shlex.quote(group['root']+'/results/'+sub['id']+'-')+'"$TRY"\n'+
                   'test ! -e "$OUT"\nmkdir -p "$OUT"\n: > "$OUT/aisbench.log"\n'+
                   shlex.join(['docker','exec',group['container'],'python3',group['root']+'/.runtime/check_test.py',group['root'],sub['id'],slot])+' \"$OUT\" before\n'+
                   'RC=0\ndocker exec '+shlex.quote(group['container'])+' bash -c '+
                   shlex.quote('OUT=$1; exec '+command)+' bash "$OUT" > "$OUT/aisbench.log" 2>&1 || RC=$?\n'+
                   shlex.join(['docker','exec',group['container'],'python3',group['root']+'/.runtime/check_test.py',group['root'],sub['id'],slot])+ ' "$OUT" after "$RC"')
            plan['tests'][sub['id']]={'node':slot,'deployment':dep['id'],'timeout':case['timeout_seconds']}
    save(root/'.runtime/dependency-env.json',dependency_env)
    save(root/'.runtime/dispatch.json',plan)
    files = {str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*')) if p.is_file()}
    save(root/'.runtime/manifest.json', {'files':files, 'generator':{
        p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}})
    save(root/'status.json', {'status':'SCRIPTS_READY','cases':{},'release':{}})
    return spec, plan


class ScriptFailure(RuntimeError):
    def __init__(self, relative, returncode):
        self.returncode = returncode
        super().__init__(f'{relative} exited {returncode}')


def retry_decision(error, phase):
    if phase == 'test':
        return False, 'Test failure or timeout; preserve result without replay'
    if isinstance(error, ValueError):
        return False, 'Invalid configuration'
    if isinstance(error, ScriptFailure):
        return error.returncode == 255, 'Transport failure' if error.returncode == 255 else 'Remote script rejected operation'
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired, ConnectionError)):
        return True, 'Infrastructure timeout or connection failure'
    return False, 'Unclassified failure; no automatic replay'


def invoke(group, relative, *args, timeout=240):
    argv = transport_argv(group['host'], ['bash',group['root']+'/'+relative,*args])
    logging.info('Executing %s on %s',relative,group['host'])
    result = subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
    if result.stdout: logging.info('%s',result.stdout.rstrip())
    if result.stderr: logging.info('%s',result.stderr.rstrip())
    if result.returncode:
        raise ScriptFailure(relative, result.returncode)
    return result.stdout


def distribute(root, plan):
    """Copy a single verified bundle per physical destination; shared paths reuse it."""
    manifest=load_json(root/'.runtime/manifest.json')
    if any(hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest for name,digest in manifest['files'].items()):
        raise ValueError('Generated bundle changed before distribution')
    bindings=load_json(root/'bindings.json')
    if bindings.get('shared_storage_groups'):
        def ssh(host,argv):
            result=subprocess.run(transport_argv(host,argv),capture_output=True,text=True,timeout=30)
            if result.returncode:raise RuntimeError('Shared storage probe failed: '+result.stderr)
            return result.stdout
        verify_groups(bindings,ssh,root/'.runtime/shared-storage.json')
    files={relative:base64.b64encode((root/relative).read_bytes()).decode() for relative in manifest['files']}
    files['.runtime/manifest.json']=base64.b64encode((root/'.runtime/manifest.json').read_bytes()).decode()
    seen=set()
    for dep in plan['deployments'].values():
        for group in dep['groups'].values():
            key=(group['host'],group['root'])
            if key in seen: continue
            seen.add(key)
            result=subprocess.run(transport_argv(group['host'],['python3','-c',DISTRIBUTE_PROGRAM]),
                input=json.dumps({'root':group['root'],'files':files,'hashes':manifest['files']}),
                capture_output=True,text=True,timeout=120)
            if result.returncode: raise RuntimeError('Bundle distribution failed: '+result.stderr)


def claim_execution(plan, token):
    """Persistent exclusive claim: different coordinators cannot replay a remote task."""
    program='''import os,pathlib,sys
path=pathlib.Path(sys.argv[1])/'.runtime/execution.claim'; token=sys.argv[2]
try:
 fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
except FileExistsError:
 if path.read_text()!=token: raise SystemExit('Task already claimed; use a new job ID')
else:
 with os.fdopen(fd,'w') as stream: stream.write(token)
'''
    seen=set()
    for dep in plan['deployments'].values():
        for group in dep['groups'].values():
            key=(group['host'],group['root'])
            if key in seen: continue
            seen.add(key)
            result=subprocess.run(transport_argv(group['host'],['python3','-c',program,group['root'],token]),capture_output=True,text=True,timeout=30)
            if result.returncode:raise RuntimeError('Execution claim failed: '+result.stderr)


def collection_access(root, group):
    """Prove directory sharing by a nonce, then select explicit local/SSH access."""
    marker='.runtime/collect-'+uuid.uuid4().hex
    token=uuid.uuid4().hex
    path=root/marker
    path.write_text(token)
    try:
        check=subprocess.run(transport_argv(group['host'],['cat',group['root']+'/'+marker]),
                             capture_output=True,text=True,timeout=30)
        if check.returncode==0 and check.stdout==token:
            return 'shared'
    finally:
        try:
            path.unlink()
        except OSError:
            logging.exception('Cannot remove collection probe %s',path)
    return 'local' if is_local(group['host']) else 'ssh'


def collect(root, plan):
    """Gather user artifacts only; never infer shared storage from matching strings."""
    seen=set()
    for dep in plan['deployments'].values():
        for group in dep['groups'].values():
            key=(group['host'],group['root'])
            if key in seen: continue
            seen.add(key)
            access=collection_access(root,group)
            if access=='shared':
                continue
            for category in ('logs','results'):
                source=group['root']+'/'+category
                probe=['test','-d',source]
                argv=transport_argv(group['host'],probe)
                checked=subprocess.run(argv,capture_output=True,timeout=30)
                if checked.returncode==1: continue  # Required files checked below, including shared paths.
                if checked.returncode: raise RuntimeError('Cannot collect '+source)
                destination=root/category;destination.mkdir(exist_ok=True)
                if access=='local':
                    if Path(source).resolve()!=destination.resolve():
                        shutil.copytree(source,destination,dirs_exist_ok=True)
                else:
                    result=subprocess.run(['scp','-r',group['host']+':'+shlex.quote(source+'/.') ,str(destination)],capture_output=True,text=True,timeout=600)
                    if result.returncode: raise RuntimeError('Artifact collection failed: '+result.stderr)
    expected=load_json(root/'status.json').get('expected_artifacts',[])
    missing=[name for name in expected if not (root/name).is_file()]
    if missing:raise RuntimeError('Missing required artifacts: '+', '.join(missing))


def check_process(pid):
    if os.name == 'nt':
        # os.kill(pid, 0) is not a non-destructive existence check on Windows.
        result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
                                capture_output=True, text=True, timeout=10)
        if result.returncode:
            raise PermissionError('Cannot inspect process list')
        if not any(len(row) > 1 and row[1] == str(pid)
                   for row in csv.reader(result.stdout.splitlines())):
            raise ProcessLookupError(pid)
    else:
        os.kill(pid, 0)


def finish_execution(root, status, subtasks, release, collect_artifacts, publish, lock, primary_error):
    """Attempt every finalization step without masking the original failure."""
    steps = {}
    failures = []

    def attempt(name, operation):
        try:
            operation()
            steps[name] = {'status':'PASSED'}
        except Exception as error:
            logging.exception('Finalization %s failed', name)
            steps[name] = {'status':'FAILED', 'error':repr(error)}
            failures.append(error)
            if status['status'] != 'INTERRUPTED':
                status['status'] = 'FAILED'

    attempt('release', release)
    if any(value != 'PASSED' for value in status['release'].values()) and status['status'] != 'INTERRUPTED':
        status['status'] = 'FAILED'

    def gather():
        try:
            collect_artifacts()
            status['artifacts'] = 'COLLECTED' if status['expected_artifacts'] else 'NOT_PRODUCED'
        except Exception as error:
            status['artifacts'] = str(error)
            raise

    attempt('collect', gather)
    for sub in subtasks:
        status['cases'].setdefault(sub['id'], 'NOT_RUN')
    attempt('publish', publish)
    attempt('unlock', lock.unlink)
    if steps['unlock']['status'] == 'FAILED':
        attempt('publish_after_unlock_failure', publish)
    report = {'status':status['status'], 'primary_error':repr(primary_error) if primary_error else None,
              'steps':steps}
    attempt('record', lambda: save(root/'.runtime/finalization.json', report))
    if failures and primary_error is None:
        raise RuntimeError('Task finalization failed; inspect coordinator log and .runtime/finalization.json') from failures[0]


def execute(root, spec, plan):
    """One foreground coordinator. Caller may manage its lifetime with a service manager."""
    lock=root/'.runtime/task.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as error:
        hint='unreadable or invalid PID'
        try:
            pid=int(lock.read_text())
            if pid<=0: raise ValueError('Invalid PID')
            try:
                check_process(pid)
                hint=f'PID {pid} exists (identity unverified)'
            except ProcessLookupError:
                hint=f'stale coordinator PID {pid}; remote resources may remain'
            except PermissionError:
                hint=f'PID {pid} cannot be inspected'
        except (OSError,ValueError,subprocess.TimeoutExpired):
            pass
        raise FileExistsError(f'Task execution lock exists: {lock}; {hint}. Verify coordinator and remote resources; retain evidence and use a new task directory. Do not blindly delete the lock.') from error
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    token=uuid.uuid4().hex
    status={'status':'RUNNING','cases':{},'release':{},'expected_artifacts':[], 'timeline':[]}
    active={}
    run_counter=0
    state_lock=threading.RLock()
    cancelled=threading.Event()
    pending=queue.Queue()
    owner=threading.get_ident()
    def drain():
        while True:
            try: snapshot=pending.get_nowait()
            except queue.Empty: break
            save(root/'status.json',snapshot)
    def publish():
        with state_lock: pending.put(copy.deepcopy(status))
        if threading.get_ident()==owner: drain()
    def event(kind, identity, value):
        with state_lock:
            status['timeline'].append({'time':time.time(),'kind':kind,'id':identity,'status':value})
        logging.info('%s %s %s',kind,identity,value)
        publish()
    def case_state(identity,value):
        with state_lock: status['cases'][identity]=value
        event('case',identity,value)
    def require_running():
        if cancelled.is_set(): raise RuntimeError('Stage cancelled')
    def release(owned=None):
        failures=[]
        with state_lock: entries=list(active.items())
        for (dep,slot),group in entries:
            if owned is not None and dep not in owned: continue
            try:
                invoke(group,f'scripts/{dep}/{slot}-stop.sh',token)
                with state_lock:
                    status['release'][dep+'/'+slot]='PASSED';del active[(dep,slot)]
            except Exception as exc:
                with state_lock: status['release'][dep+'/'+slot]=str(exc)
                failures.append(str(exc))
        publish()
        if failures: raise RuntimeError('Release not verified: '+'; '.join(failures))
    def branch_run(subtasks, branch_id):
        nonlocal run_counter
        owned={s['deployment_id'] for s in subtasks}
        run_number=0
        event('branch',branch_id,'RUNNING')
        try:
            for index,sub in enumerate(subtasks):
                if cancelled.is_set(): raise RuntimeError('Stage cancelled')
                with state_lock: previous={key:{'status':value} for key,value in status['cases'].items()}
                if higher_concurrency_blocker({**spec, 'sub_tasks':subtasks},sub,previous):
                    case_state(sub['id'],'SKIPPED');continue
                dep_id=sub['deployment_id'];dep=plan['deployments'][dep_id]
                case_state(sub['id'],'RUNNING')
                for attempt in range(1,spec['execution']['max_retries']+2):
                    phase='deployment'
                    try:
                        with state_lock: has_active=any(key[0] in owned for key in active)
                        if not has_active:
                            with state_lock:
                                run_counter+=1
                                run_number=run_counter
                            for slot,group in dep['groups'].items():
                                require_running()
                                with state_lock: active[(dep_id,slot)]=group
                                invoke(group,f'scripts/{dep_id}/{slot}-container.sh',token)
                            for slot,group in dep['groups'].items():
                                require_running()
                                with state_lock: status['expected_artifacts'].append(f'logs/{dep_id}-run{run_number:02}/{slot}-dependencies.log')
                                publish()
                                invoke(group,f'scripts/{dep_id}/{slot}-dependencies.sh',f'run{run_number:02}',timeout=spec['execution']['dependency_timeout_seconds']+30)
                            for slot,group in dep['groups'].items():
                                with state_lock: status['expected_artifacts'].extend(f'logs/{dep_id}-run{run_number:02}/{name}' for name in group['log_names'])
                                publish()
                                invoke(group,f'scripts/{dep_id}/{slot}-start.sh',f'run{run_number:02}')
                        deadline=time.monotonic()+spec['execution']['startup_timeout_seconds']
                        phase='health'
                        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                        while True:
                            require_running()
                            for slot,group in dep['groups'].items():
                                invoke(group,f'scripts/{dep_id}/{slot}-probe.sh')
                            try:
                                for url in dep['health_urls']:
                                    with opener.open(url,timeout=5) as response:
                                        if response.status!=200: raise RuntimeError('Health failed')
                                break
                            except Exception:
                                if time.monotonic()>=deadline: raise TimeoutError('Service readiness timed out')
                                time.sleep(2)
                        if spec['execution']['smoke']!='skip':
                            phase='smoke'
                            invoke(dep['groups'][dep['endpoint_node']],f'scripts/{dep_id}/smoke.sh',timeout=spec['execution']['smoke_timeout_seconds']+10)
                        test=plan['tests'][sub['id']]
                        with state_lock: status['expected_artifacts'].append(f"results/{sub['id']}-try{attempt:02}/aisbench.log")
                        publish()
                        require_running()
                        phase='test'
                        invoke(dep['groups'][test['node']],f"scripts/tests/{sub['id']}.sh",f'try{attempt:02}',timeout=test['timeout'])
                        with state_lock: status['expected_artifacts'].append(f"results/{sub['id']}-try{attempt:02}/validated.json")
                        case_state(sub['id'],'PASSED');break
                    except Exception as error:
                        logging.exception('Subtask %s attempt %s failed',sub['id'],attempt)
                        retryable,reason=retry_decision(error,phase)
                        retrying=retryable and attempt<=spec['execution']['max_retries'] and not cancelled.is_set()
                        with state_lock:
                            status.setdefault('attempt_failures',[]).append({'case':sub['id'],'attempt':attempt,'phase':phase,'retryable':retryable,'retrying':retrying,'reason':reason,'error':str(error)})
                        publish()
                        release(owned)
                        if not retrying:
                            case_state(sub['id'],'FAILED')
                            if 'stages' in spec: cancelled.set()
                            break
                next_dep=subtasks[index+1]['deployment_id'] if index+1<len(subtasks) else None
                if sub['after_case']!='keep_service' or next_dep!=dep_id:
                    release(owned)
        except BaseException as exc:
            cancelled.set()
            event('branch',branch_id,'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED')
            raise
        finally:
            release(owned)
        event('branch',branch_id,'PASSED' if all(status['cases'].get(s['id'])=='PASSED' for s in subtasks) else 'FAILED')
    try:
        distribute(root,plan)
        claim_execution(plan,token)
        # Verify the exact scripts before any deployment is started.
        manifest=load_json(root/'.runtime/manifest.json')
        for dep in plan['deployments'].values():
            for group in dep['groups'].values():
                for relative,digest in manifest['files'].items():
                    if relative.startswith('scripts/'):
                        command=['sha256sum',group['root']+'/'+relative]
                        result=subprocess.run(transport_argv(group['host'],command),capture_output=True,text=True,timeout=30)
                        if result.returncode or result.stdout.split()[0]!=digest:
                            raise ValueError('Bundle unavailable or changed on '+group['host']+': '+relative)
        stages=spec.get('stages') or [{'id':'serial','branches':[{'id':'serial','sub_tasks':[s['id'] for s in spec['sub_tasks']]}]}]
        tasks={s['id']:s for s in spec['sub_tasks']}
        for stage in stages:
            event('stage',stage['id'],'RUNNING')
            with ThreadPoolExecutor(max_workers=len(stage['branches'])) as pool:
                futures=[pool.submit(branch_run,[tasks[sid] for sid in branch['sub_tasks']],stage['id']+'/'+branch['id']) for branch in stage['branches']]
                try:
                    while not all(f.done() for f in futures):
                        drain()
                        wait(futures,timeout=0.05)
                    drain()
                    errors=[f.exception() for f in futures if f.exception() is not None]
                    if errors: raise next((e for e in errors if isinstance(e,KeyboardInterrupt)),errors[0])
                except BaseException:
                    cancelled.set()
                    wait(futures)
                    raise
            drain()
            stage_passed=all(status['cases'].get(sid)=='PASSED' for branch in stage['branches'] for sid in branch['sub_tasks'])
            if 'stages' in spec and not stage_passed:
                event('stage',stage['id'],'FAILED')
                raise RuntimeError('Stage failed: '+stage['id'])
            if 'stages' in spec:
                collect(root,plan)
            event('stage',stage['id'],'PASSED' if stage_passed else 'FAILED')
        status['status']='PASSED' if all(v=='PASSED' for v in status['cases'].values()) else 'FAILED'
    except BaseException as exc:
        status['status']='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED'
        for key,value in status['cases'].items():
            if value=='RUNNING':status['cases'][key]=status['status']
        logging.exception('Task stopped')
        try:
            drain()
            if 'stage' in locals(): event('stage',stage['id'],status['status'])
        except Exception:
            logging.exception('Cannot publish original task failure; preserving original exception')
        raise
    finally:
        finish_execution(root,status,spec['sub_tasks'],release,
                         lambda: collect(root,plan),publish,lock,sys.exc_info()[1])



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-spec',type=Path,required=True)
    parser.add_argument('--bindings',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--mode',choices=['FULL','scripts'],required=True)
    args=parser.parse_args()
    root=args.run_dir.resolve()
    spec,plan=generate(load_json(args.job_spec),load_json(args.bindings),root)
    if args.mode=='scripts':
        print('SCRIPTS_READY '+str(root));return 0
    (root/'logs').mkdir(exist_ok=True)
    logging.basicConfig(filename=root/'logs/coordinator.log',level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    def interrupted(signum,frame): raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    execute(root,spec,plan)
    return 0 if load_json(root/'status.json')['status']=='PASSED' else 2


if __name__=='__main__': raise SystemExit(main())
