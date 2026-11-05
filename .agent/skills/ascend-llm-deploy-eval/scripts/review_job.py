"""Read-only evidence for interactive job review; never repairs confirmed parameters."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from remote_node import transport_argv
from task_model import load_json, normalize, validate, validate_bindings
from topology import compile_deployment


PROBE = r'''
import json,pathlib,subprocess,sys
p=json.load(sys.stdin); result={'models':{},'errors':[],'paths':{}}
def command(argv):
 try:
  r=subprocess.run(argv,capture_output=True,text=True,timeout=30)
  if r.returncode: raise RuntimeError(r.stderr or r.stdout)
  return r.stdout
 except Exception as e:
  result['errors'].append(str(argv)+': '+str(e)); return None
result['hardware']=command(['npu-smi','info'])
addresses=command(['ip','-j','address','show'])
result['network']=json.loads(addresses) if addresses else []
for image in p['images']: command(['docker','image','inspect',image])
for name,path in p['models'].items():
 try:
  root=pathlib.Path(path); config=json.loads((root/'config.json').read_text())
  weights=any(root.glob('*.safetensors')) or any(root.glob('*.bin'))
  if not weights: raise ValueError('No weight files found')
  result['models'][name]={'path':path,'config':config,'weights_present':weights}
 except Exception as e: result['errors'].append(name+': '+str(e))
for path in p['paths']:
 result['paths'][path]={'present_on_host':pathlib.Path(path).exists(),'container_visibility':'REQUIRES_REVIEW'}
for path in p['mount_sources']:
 if not pathlib.Path(path).exists(): result['errors'].append('Missing mount source: '+path)
print(json.dumps(result))
'''


def assess(spec, bindings, evidence):
    errors=[]; review=[]
    for slot,observed in evidence.items():
        errors.extend(slot+': '+e for e in observed['errors'])
    for dep in spec['deployments']:
        compile_deployment(dep,{slot:bindings['nodes'][slot]['deployments'][dep['id']]
            for slot in {i['placement']['node'] for i in dep['instances']}})
        for item in dep['instances']:
            slot=item['placement']['node'];bound=bindings['nodes'][slot]['deployments'][dep['id']]['network']
            observed=evidence[slot]
            if not any(i['ifname']==bound['interface'] and any(a.get('local')==bound['ip'] for a in i.get('addr_info',[])) for i in observed['network']):
                errors.append(slot+': confirmed IP/interface not present on machine')
            if item['role']=='proxy':continue
            config=observed['models'].get(dep['model']['id'],{}).get('config',{})
            review.append({'deployment':dep['id'],'instance':item['id'],'model_config':config,
                'launch':item['launch'],'devices':item['placement']['devices'],
                'requires_agent_review':['architecture and quantization support in selected image',
                    'context length / rope scaling','parallelism and memory capacity','connector and proxy dependencies']})
    return {'status':'BLOCKED' if errors else 'REQUIRES_AGENT_REVIEW','errors':errors,'review':review,'nodes':evidence}


def review(spec, bindings):
    validate_bindings(bindings)
    spec=copy.deepcopy(spec)
    for dep in spec.get('deployments',[]):
        for instance in dep.get('instances',[]):
            if instance['role']!='proxy':
                if instance['launch'].get('preserve_reference') is False:
                    raise ValueError('Only confirmed-parameter preservation is supported')
                instance['launch']['preserve_reference']=True
    spec=normalize(spec)
    errors=validate(spec)
    if errors:raise ValueError('; '.join(errors))
    evidence={}
    for slot,binding in bindings['nodes'].items():
        deps=[d for d in spec['deployments'] if any(i['placement']['node']==slot for i in d['instances'])]
        payload={'models':{d['model']['id']:binding['model_paths'][d['model']['id']] for d in deps},
                 'images':list({d['runtime']['image'] for d in deps}),'paths':list(binding.get('paths',{}).values()),
                 'mount_sources':[m['source'] for m in binding.get('mounts',[])]}
        response=subprocess.run(transport_argv(binding['host'],['python3','-c',PROBE]),input=json.dumps(payload),capture_output=True,text=True,timeout=180)
        if response.returncode:raise RuntimeError('Review probe failed on '+slot+': '+response.stderr)
        evidence[slot]=json.loads(response.stdout)
    return assess(spec,bindings,evidence)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-spec',type=Path,required=True)
    parser.add_argument('--bindings',type=Path,required=True)
    args=parser.parse_args()
    report=review(load_json(args.job_spec),load_json(args.bindings))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    sys.exit(2 if report['errors'] else 0)
