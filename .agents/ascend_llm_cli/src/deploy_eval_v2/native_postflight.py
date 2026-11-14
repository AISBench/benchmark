"""Bounded diagnostic audit of actual synthetic request text in tester Python."""
import argparse
import hashlib
import json
from pathlib import Path
from .durable import atomic_json
from .json_io import load_json,unique_object,reject_constant
from .native_results import LINE_LIMIT
from .aisbench_cli import synthetic_config


def audit(folder,case,model_path,*,tokenizer=None):
    root=Path(folder).resolve()
    report=dict(schema_version='1.0',status='UNVERIFIED',target=synthetic_config(case),
                scope='formal request text; excludes warmup',policy='diagnostic_only; no load or data changes')
    try:
        paths=iter((root/'outputs').glob('*/performances/*/*_details.jsonl'))
        path=next(paths)
        if next(paths,None) is not None or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('detail inventory')
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=False,local_files_only=True)
        count=total=0;low=None;high=0;common=None;checksum=hashlib.sha256()
        with path.open('rb') as stream:
            while line:=stream.readline(LINE_LIMIT+1):
                if len(line)>LINE_LIMIT:
                    raise ValueError('row limit')
                checksum.update(line)
                if not line.strip(): continue
                row=json.loads(line,object_pairs_hook=unique_object,parse_constant=reject_constant)
                if not isinstance(row.get('input'),str):
                    raise ValueError('input text absent')
                tokens=tokenizer.encode(row['input'],add_special_tokens=False)
                if len(tokens)>2**20:
                    raise ValueError('tokenized row limit')
                count+=1;total+=len(tokens);high=max(high,len(tokens));low=len(tokens) if low is None else min(low,len(tokens))
                if count>case['formal_requests']:
                    raise ValueError('request count')
                if common is None:
                    common=tokens
                else:
                    stop=0
                    for left,right in zip(common,tokens):
                        if left!=right: break
                        stop+=1
                    common=common[:stop]
        if count!=case['formal_requests']:
            raise ValueError('request count')
        report.update(status='MEASURED',details_sha256=checksum.hexdigest(),request_count=count,
            input_tokens=dict(min=low,max=high,mean=total/count),exact_input_lengths=low==high==case['input_tokens'],
            common_prefix_tokens=len(common) if count>1 else None,
            note='Observed text prefix is not a server cache hit ratio')
    except Exception:
        report['reason']='NATIVE_TEXT_OR_TOKENIZER_EVIDENCE_UNAVAILABLE'
    atomic_json(root/'synthetic-evidence.json',report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
    args=parser.parse_args();value=load_json(args.request,65536)
    audit(value['folder'],value['case'],value['model_path'])


if __name__=='__main__': main()
