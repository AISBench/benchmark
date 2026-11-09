"""Read-only evidence for interactive job review; never repairs confirmed parameters."""
from mount_profiles import resolve_mounts
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
import hashlib,json,pathlib,subprocess,sys
p=json.load(sys.stdin); result={'models':{},'errors':[],'paths':{},'images':{}}
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
for image in p['images']:
 raw=command(['docker','image','inspect',image])
 if raw:
  try:
   records=json.loads(raw)
   if not records: raise ValueError('Empty image inspection')
   result['images'][image]=[{'id':v['Id'],'repo_tags':v.get('RepoTags'),
       'repo_digests':v.get('RepoDigests'),'architecture':v.get('Architecture'),
       'source':'docker image inspect'} for v in records]
  except Exception as e: result['errors'].append(image+': '+str(e))
def metadata(file, parse=False):
 data=file.read_bytes()
 if not data: raise ValueError('Empty file: '+str(file))
 info={'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 if parse: info['content']=json.loads(data)
 return info
for name,path in p['models'].items():
 model={'path':path,'weights_present':False,'files':{},'shards':{},
        'verification_scope':'Metadata hashes and readable nonempty shards; no full shard hash or tensor validation'}
 result['models'][name]=model
 try:
  root=pathlib.Path(path)
  model['files']['config.json']=metadata(root/'config.json',True)
  model['config']=model['files']['config.json']['content']
  if not isinstance(model['config'],dict): raise ValueError('config.json must be an object')
  indexes=sorted(root.glob('*.index.json')); shards=set()
  for index in indexes:
   item=metadata(index,True); model['files'][index.name]=item
   mapping=item['content'].get('weight_map')
   if not isinstance(mapping,dict) or not mapping: raise ValueError('Invalid weight_map: '+index.name)
   for value in mapping.values():
    if not isinstance(value,str) or not value or pathlib.PurePosixPath(value).is_absolute() or '..' in pathlib.PurePosixPath(value).parts:
     raise ValueError('Invalid shard path in '+index.name)
    shards.add(value)
  if not indexes:
   shards={f.name for pattern in ('*.safetensors','*.bin') for f in root.glob(pattern)}
  if not shards: raise ValueError('No weight files found')
  model['indexed']=bool(indexes)
  for shard in sorted(shards):
   file=root/shard
   with file.open('rb') as stream:
    if not stream.read(1): raise ValueError('Empty shard: '+shard)
   model['shards'][shard]={'size_bytes':file.stat().st_size,'readable':True}
  model['weights_present']=True
  tokenizer_files=[f for pattern in ('tokenizer*','special_tokens_map.json','vocab.*','merges.txt') for f in root.glob(pattern) if f.is_file()]
  if not any(f.name in ('tokenizer.json','tokenizer.model','vocab.json','vocab.txt') or f.name.endswith('.model') for f in tokenizer_files):
   raise ValueError('No tokenizer vocabulary artifact found')
  for file in sorted(set(tokenizer_files)):
   item=metadata(file,file.suffix=='.json')
   item.pop('content',None)
   model['files'][file.name]=item
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
                 'mount_sources':[m['source'] for m in resolve_mounts(binding)]}
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
