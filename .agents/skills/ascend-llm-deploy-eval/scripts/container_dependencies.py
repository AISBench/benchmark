"""Bounded, container-local dependency evidence; never installs or starts services."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
from aisbench_cli import dataset_for, check_dataset, check_synthetic_support


def benchmark_check(root, spec, binding, sub):
    case=next(c for c in spec['cases'] if c['id']==sub['case_id'])
    executable=binding['benchmark'].get('executable','ais_bench')
    resolved=shutil.which(executable)
    if not resolved:raise ValueError('AISBench executable unavailable: '+executable)
    distributions=[]
    for dist in importlib.metadata.distributions():
        if any(e.group=='console_scripts' and e.name==Path(resolved).name for e in dist.entry_points):
            distributions.append({'name':dist.metadata['Name'],'version':dist.version})
    if not distributions:raise ValueError('No installed distribution owns AISBench entry point: '+resolved)
    evidence={'subtask':sub['id'],'executable':resolved,'distributions':distributions}
    if not case.get('aisbench_config'):
        data=dataset_for(binding['benchmark'],case)
        if data.get('type')=='aisbench_synthetic':
            evidence['dataset']=check_synthetic_support(case)
        else:
            check_dataset(data,case['formal_requests'])
            evidence['dataset']={'path':data['path'],'status':'CHECKED'}
    else:
        from mmengine.config import Config
        config=Config.fromfile(str(root/'scripts/tests'/(sub['id']+'.config.py')))
        from ais_bench.benchmark.datasets import CustomDataset
        evidence['datasets']=[]
        for data in config.get('datasets',[]):
            if data.get('type') is CustomDataset and Path(str(data.get('path',''))).suffix in ('.jsonl','.csv'):
                reader=data.get('reader_cfg',{});columns=reader.get('input_columns',[])
                if len(columns)==1 and reader.get('output_column'):
                    check_dataset({'path':data['path'],'input_column':columns[0],'output_column':reader['output_column']},case['formal_requests'])
                    evidence['datasets'].append({'path':data['path'],'status':'CHECKED'});continue
            evidence['datasets'].append({'abbr':data.get('abbr'),'status':'PENDING_NATIVE_LOADER_VALIDATION'})
    print(json.dumps(evidence),flush=True)


def proxy_check(root, dep, node):
    """Check the generated, fully bound proxy command without executing it."""
    for instance in dep['instances']:
        if instance['role']!='proxy' or instance['placement']['node']!=node:
            continue
        script=root/'scripts'/dep['id']/(node+'-'+instance['id']+'.sh')
        # A generator-owned marker prevents bootstrap exec lines from being selected.
        text=script.read_text()
        marker='# @proxy-command'
        lines=text.splitlines()
        positions=[i for i,line in enumerate(lines) if line.strip()==marker]
        if len(positions)!=1:
            raise ValueError('Missing or ambiguous proxy command marker: '+str(script))
        tail='\n'.join(lines[positions[0]+1:]).replace('\\\n','')
        line=next((line.strip() for line in tail.splitlines() if line.strip() and not line.lstrip().startswith('#')), '')
        try:
            command=shlex.split(line, comments=True)
        except ValueError as error:
            raise ValueError('Invalid generated proxy command in '+str(script)+': '+repr(line)) from error
        if len(command)<3 or command[:2]!=['exec','python3']:
            raise ValueError('Invalid generated proxy command in '+str(script)+': '+repr(line))
        program=Path(command[2])
        try:
            if not program.is_file():
                raise OSError('not a regular file')
            with program.open('rb') as source:
                source.read(1)
        except OSError as error:
            raise ValueError('Proxy program unavailable in actual container: '+str(program)) from error
        print(json.dumps({'proxy':instance['id'],'program':str(program),'status':'READABLE'}),flush=True)


def worker(root, dep_id, node):
    root=Path(root);spec=json.loads((root/'job_spec.json').read_text());bindings=json.loads((root/'bindings.json').read_text())
    dep=next(d for d in spec['deployments'] if d['id']==dep_id)
    proxy_check(root,dep,node)
    envs=json.loads((root/'.runtime/dependency-env.json').read_text())[dep_id][node]
    for instance,env in envs.items():
        print('MODEL '+instance+': image ENV + confirmed launch.env; bootstrap is NOT executed',flush=True)
        result=subprocess.run([sys.executable,'-c','import torch_npu; print(torch_npu.__file__); print(getattr(torch_npu,"__version__","unknown"))'],env={**os.environ,**env})
        if result.returncode:raise RuntimeError('Model dependency import failed; inspect container visibility and confirmed environment. Bootstrap-only environment is not included: '+instance)
    plan=json.loads((root/'.runtime/dispatch.json').read_text())
    for sub in spec['sub_tasks']:
        if sub['deployment_id']==dep_id and plan['tests'][sub['id']]['node']==node:
            benchmark_check(root,spec,bindings['nodes'][node],sub)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('root')
    parser.add_argument('deployment')
    parser.add_argument('node')
    parser.add_argument('timeout',type=int,nargs='?')
    args=parser.parse_args()
    if args.worker:
        if args.timeout is not None: parser.error('--worker does not accept a timeout')
        worker(args.root,args.deployment,args.node);return
    if args.timeout is None or args.timeout<=0: parser.error('positive timeout is required')
    timeout=args.timeout
    process=subprocess.Popen([sys.executable,__file__,'--worker',args.root,args.deployment,args.node],start_new_session=True)
    try:
        rc=process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        import signal
        os.killpg(process.pid,signal.SIGKILL);process.wait()
        raise TimeoutError('Container dependency preflight timed out')
    if rc:raise SystemExit(rc)

if __name__=='__main__':main()
