"""Container-side test preconditions and evidence; no task coordination."""
import json
import urllib.request
from pathlib import Path
from aisbench_cli import dataset_for, check_dataset, validate_native_result, audit_synthetic_result
from validate_case import validate
from cache_reset import reset_instances, validate_reset
from metric_evidence import capture, summarize
from urllib.parse import urlsplit


def native_config_check(config, case, model, endpoint, model_path):
    """Check resolved native config without rewriting its adapters or load."""
    models=config.get('models',[]);datasets=config.get('datasets',[])
    if not models or not datasets:raise ValueError('Native models/datasets must be nonempty')
    url=urlsplit(endpoint)
    expected={'model':model['served_name'],'path':model_path,'host_ip':url.hostname,
              'host_port':url.port,'batch_size':case['concurrency'],
              'max_out_len':case['output_tokens'],'request_rate':case.get('request_rate',0)}
    for entry in models:
        for key,value in expected.items():
            if entry.get(key)!=value:raise ValueError('Native config conflicts with confirmed task: '+key)
        if 'url' in entry and entry['url'].rstrip('/')!=endpoint.rstrip('/'):
            raise ValueError('Native config endpoint mismatch')
    groups=[(m.get('abbr'),d.get('abbr')) for m in models for d in datasets]
    import re
    if any(not isinstance(v,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',v) or v in ('.','..') for pair in groups for v in pair) or len(set(groups))!=len(groups):
        raise ValueError('Native model/dataset abbreviations must form unique safe result groups')
    return groups


def check(task, sub_id, node, output, phase, returncode='0'):
    root=Path(task);folder=Path(output)
    spec=json.loads((root/'job_spec.json').read_text())
    bindings=json.loads((root/'bindings.json').read_text())
    sub=next(s for s in spec['sub_tasks'] if s['id']==sub_id)
    case=next(c for c in spec['cases'] if c['id']==sub['case_id'])
    service=json.loads((root/'.runtime/dispatch.json').read_text())['deployments'][sub['deployment_id']]
    if phase=='before':
        if case.get('aisbench_config'):
            from mmengine.config import Config
            deployment=next(d for d in spec['deployments'] if d['id']==sub['deployment_id'])
            groups=native_config_check(Config.fromfile(str(root/'scripts/tests'/(sub_id+'.config.py'))),case,
                deployment['model'],service['endpoint'],bindings['nodes'][node]['model_paths'][deployment['model']['id']])
            (folder/'native-expectations.json').write_text(json.dumps(groups))
        if not case.get('aisbench_config'):
            dataset=dataset_for(bindings['nodes'][node]['benchmark'],case)
            if dataset.get('type')!='aisbench_synthetic':
                check_dataset(dataset,case['formal_requests'])
        if case.get('reset_cache'):
            targets={i:url.rsplit('/reset_prefix_cache',1)[0] for i,url in enumerate(service['reset_urls'])}
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            def http(index,path,method='GET'):
                with opener.open(urllib.request.Request(targets[index]+path,method=method),timeout=30) as response:
                    return response.read().decode()
            reset_instances(folder,list(targets),http,targets)
        capture(service['metric_instances'],folder,'before')
        return
    if phase!='after':raise ValueError('Unknown evidence phase')
    (folder/'execution.json').write_text(json.dumps({'returncode':int(returncode)}))
    capture(service['metric_instances'],folder,'after')
    summarize(folder)
    if case.get('reset_cache'):
        validate_reset(folder)
    if case.get('aisbench_config'):
        if not (folder/'native-expectations.json').is_file():raise ValueError('Missing resolved native config expectations')
        validate_native_result(folder,case)
    else:
        dataset=dataset_for(bindings['nodes'][node]['benchmark'],case)
        if dataset.get('type')=='aisbench_synthetic' and int(returncode)==0:
            deployment=next(d for d in spec['deployments'] if d['id']==sub['deployment_id'])
            audit_synthetic_result(folder,case,bindings['nodes'][node]['model_paths'][deployment['model']['id']])
        validate(folder,case,input_length_policy='native_target' if dataset.get('type')=='aisbench_synthetic' else 'exact')
