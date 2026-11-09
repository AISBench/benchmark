"""Bounded, container-local dependency evidence; never installs or starts services."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from aisbench_cli import dataset_for, check_dataset


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
        data=dataset_for(binding['benchmark'],case);check_dataset(data,case['formal_requests'])
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


def worker(root, dep_id, node):
    root=Path(root);spec=json.loads((root/'job_spec.json').read_text());bindings=json.loads((root/'bindings.json').read_text())
    dep=next(d for d in spec['deployments'] if d['id']==dep_id)
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
    if sys.argv[1]=='--worker':
        worker(*sys.argv[2:]);return
    timeout=int(sys.argv[4])
    process=subprocess.Popen([sys.executable,__file__,'--worker',*sys.argv[1:4]],start_new_session=True)
    try:
        rc=process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        import signal
        os.killpg(process.pid,signal.SIGKILL);process.wait()
        raise TimeoutError('Container dependency preflight timed out')
    if rc:raise SystemExit(rc)

if __name__=='__main__':main()
