"""Run a pinned upstream GSM8K generator in an isolated process and audit its output.

This prepares data only; it never sends prefix warmup requests or starts AISBench.
"""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import random
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_source(path):
    rows=[]
    with Path(path).open(encoding='utf-8-sig') as stream:
        for number,line in enumerate(stream,1):
            row=json.loads(line)
            if not isinstance(row,dict) or not isinstance(row.get('question'),str) or not row['question'] or not isinstance(row.get('answer'),str):
                raise ValueError(f'Invalid GSM8K row {number}')
            rows.append(row)
    if not rows: raise ValueError('Empty GSM8K source')
    return rows


def audit(path, tokenizer, count, target):
    lengths=[]; common=None
    for row in inspect_source(path):
        tokens=tokenizer.encode(row['question'],add_special_tokens=False)
        lengths.append(len(tokens))
        if common is None: common=tokens
        else:
            end=0
            while end<min(len(common),len(tokens)) and common[end]==tokens[end]: end+=1
            common=common[:end]
    if len(lengths)!=count: raise ValueError('Generated request count mismatch')
    return {'token_lengths':lengths,'target_tokens':target,
            'exact_lengths':all(n==target for n in lengths),
            'common_prefix_tokens':len(common or []),
            'template_overhead':'UNVERIFIED; validate actual service requests',
            'cache_hit_rate':'UNVERIFIED'}


def normalize_lengths(path, tokenizer, target):
    """Bounded decode/re-encode correction; refuse unresolved lengths."""
    rows=inspect_source(path); corrections=[]
    for index,row in enumerate(rows):
        original=len(tokenizer.encode(row['question'],add_special_tokens=False))
        for attempt in range(8):
            tokens=tokenizer.encode(row['question'],add_special_tokens=False)
            if len(tokens)==target: break
            if not tokens: raise ValueError('Empty generated token sequence')
            adjusted=tokens[:target] if len(tokens)>target else tokens+tokens[-1:]*(target-len(tokens))
            row['question']=tokenizer.decode(adjusted,skip_special_tokens=True)
        if len(tokenizer.encode(row['question'],add_special_tokens=False))!=target:
            raise ValueError(f'Cannot obtain exact token length for row {index}')
        if original!=target: corrections.append({'row':index,'before':original,'after':target})
    Path(path).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    return corrections


def generate(args):
    source=Path(args.source).resolve(); upstream=Path(args.upstream).resolve()
    output=Path(args.output).resolve()
    inspect_source(source)
    files={name:sha(upstream/name) for name in ('generate_dataset.py','data_picker.py')}
    if files['generate_dataset.py']!=args.generator_sha256 or files['data_picker.py']!=args.picker_sha256:
        raise ValueError('Pinned upstream source digest mismatch')
    if args.tokens<4 or args.count<1 or not 0<=args.prefix_ratio<1:
        raise ValueError('Invalid generation parameters')
    tokenizer_root=Path(args.tokenizer)
    tokenizer_files={p.name:sha(p) for p in sorted(tokenizer_root.iterdir())
                     if p.is_file() and (p.name.startswith('tokenizer') or p.name in
                     ('config.json','special_tokens_map.json','added_tokens.json','vocab.json','merges.txt'))}
    if not tokenizer_files: raise ValueError('No local tokenizer files to fingerprint')
    identity={'source':sha(source),'upstream':files,'adapter':sha(__file__),
              'tokenizer':tokenizer_files,'versions':{name:importlib.metadata.version(name)
              for name in ('transformers','tokenizers','torch')},
              'tokens':args.tokens,'count':args.count,'seed':args.seed,
              'ratio':args.prefix_ratio,'backend':getattr(args,'tokenizer_backend','auto')}
    cache_key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if output.exists() and any(output.iterdir()):
        manifest_path=output/'generation.json'
        if not manifest_path.is_file(): raise ValueError('Incomplete output; retain evidence and use a new directory')
        cached=json.loads(manifest_path.read_text(encoding='utf-8'))
        if cached.get('cache_key')!=cache_key or sha(cached['dataset'])!=cached['dataset_sha256']:
            raise ValueError('Existing generation identity or content mismatch')
        if not cached['audit']['exact_lengths']: raise ValueError('Cached lengths were not verified')
        return cached
    output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(upstream))
    spec=importlib.util.spec_from_file_location('pinned_gsm8k_generator',upstream/'generate_dataset.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Batch execution records a manifest instead of starting tqdm monitor threads.
    module.tqdm = None
    if getattr(args, 'tokenizer_backend', 'auto') == 'tokenizers':
        # DeepSeek V4's vLLM tokenizer uses this same base backend, then adds
        # its chat renderer. Raw input audit does not include that renderer.
        from transformers import TokenizersBackend
        module.AutoTokenizer = TokenizersBackend
    original=module.DataPicker
    picks=[]
    class SourcePicker(original):
        def __init__(self, jsonl_file, record_file='picked_ids.txt', prefix_flag=0):
            super().__init__(str(source),str(output/'picked_ids.txt'),prefix_flag)
        def pick_one(self):
            value=super().pick_one()
            picks.append({'question_sha256':hashlib.sha256(value.encode()).hexdigest() if value is not None else None,
                          'prefix':bool(self.prefix_flag)})
            return value
    module.DataPicker=SourcePicker
    random.seed(args.seed);module.torch.manual_seed(args.seed)
    source_digest=sha(source)
    prefix,dataset=module.create_multi_prefix_dataset(args.tokenizer,args.tokens,args.count,
        str(output),int(args.prefix_ratio>0),1,args.prefix_ratio,args.seed,1)
    tokenizer=module.AutoTokenizer.from_pretrained(args.tokenizer)
    corrections=normalize_lengths(dataset,tokenizer,args.tokens)
    evidence=audit(dataset,tokenizer,args.count,args.tokens)
    if sha(source)!=source_digest: raise ValueError('Source changed during generation')
    manifest={'cache_key':cache_key,'identity':identity,'source_sha256':source_digest,'upstream_files':files,
        'parameters':vars(args),'dataset':str(dataset),'dataset_sha256':sha(dataset),
        'prefix_dataset':prefix,'prefix_test':False,'native_warmups':1,
        'picks':picks,'length_corrections':corrections,'audit':evidence}
    (output/'generation.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','upstream','output','tokenizer','generator-sha256','picker-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--tokens',type=int,required=True)
    parser.add_argument('--count',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--prefix-ratio',type=float,required=True)
    parser.add_argument('--tokenizer-backend',choices=('auto','tokenizers'),default='auto')
    result=generate(parser.parse_args())
    print(json.dumps({'dataset':result['dataset'],'audit':result['audit']}))
