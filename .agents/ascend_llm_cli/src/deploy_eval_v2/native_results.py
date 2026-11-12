"""Streaming native AISBench request evidence validation.

Semantics migrated from baseline aisbench_cli.validate_native_result at commit
74a7feffb76c32b23658d37ea681784b3a9b2474; frozen helper remains unchanged.
"""
import json
import hashlib
import math
from pathlib import Path
from .durable import atomic_json
from .json_io import load_json,unique_object,reject_constant
from .process_guard import remaining,Cancelled
from .protocol import ProtocolError

LINE_LIMIT=1024*1024
GROUP_LIMIT=128


def validate_native_result(folder,case,*,deadline=None,cancelled=None):
    folder=Path(folder)
    def check():
        if deadline is not None:
            remaining(deadline)
        if cancelled is not None and cancelled.is_set():
            raise Cancelled('Result validation cancelled')
    def confined(path):
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
            raise ProtocolError('INVALID_INPUT: result evidence escapes operation')
    check()
    execution=load_json(folder/'execution.json',65536)
    if type(execution.get('returncode')) is not int or execution['returncode']!=0:
        raise ProtocolError('INVALID_INPUT: AISBench exited unsuccessfully')
    expected=load_json(folder/'native-expectations.json',65536)
    if (not isinstance(expected,list) or not 1<=len(expected)<=GROUP_LIMIT or
            any(not isinstance(pair,list) or len(pair)!=2 or any(not isinstance(v,str) or len(v)>256 for v in pair) for pair in expected)):
        raise ProtocolError('INVALID_INPUT: native result expectations')
    unique_expected={tuple(pair) for pair in expected}
    if len(unique_expected)!=len(expected):
        raise ProtocolError('INVALID_INPUT: duplicate native result expectations')
    expected=unique_expected
    groups={}
    for path in (folder/'outputs').glob('*/performances/*/*_details.jsonl'):
        check();confined(path)
        group=(path.parent.name,path.name.removesuffix('_details.jsonl'))
        if group in groups or group not in expected or len(groups)>=GROUP_LIMIT:
            raise ProtocolError('INVALID_INPUT: native result groups differ from config')
        seen=set()
        checksum=hashlib.sha256()
        timing_count=0;max_latency=0
        timings=[]
        token_stats={key:dict(count=0,total=0) for key in ('input_tokens','output_tokens')}
        with path.open('rb') as source:
            while True:
                check()
                line=source.readline(LINE_LIMIT+1)
                if not line:
                    break
                checksum.update(line)
                if len(line)>LINE_LIMIT:
                    raise ProtocolError('INVALID_INPUT: native detail row exceeds 1 MiB limit')
                if not line.strip():
                    continue
                try:
                    row=json.loads(line,object_pairs_hook=unique_object,parse_constant=reject_constant)
                    identifier=row['id']
                    if (type(identifier) not in (int,str) or (isinstance(identifier,str) and len(identifier)>256) or
                            identifier in seen or row.get('success') is not True or row.get('error_info')):
                        raise ValueError('request evidence')
                except (ValueError,KeyError,TypeError) as error:
                    raise ProtocolError('INVALID_INPUT: failed, duplicate or malformed native request') from error
                seen.add(identifier)
                start,end=row.get('start_time'),row.get('end_time')
                if (type(start) in (int,float) and type(end) in (int,float) and
                        math.isfinite(start) and math.isfinite(end) and end>start):
                    timing_count+=1;max_latency=max(max_latency,end-start)
                    if timings is not None:
                        if len(timings)<100000:
                            timings.append((start,end))
                        else:
                            timings=None  # Bounded optional timing analysis; request validation continues.
                for key,stat in token_stats.items():
                    if type(row.get(key)) is int and row[key]>=0:
                        stat['count']+=1;stat['total']+=row[key]
                if len(seen)>case['formal_requests']:
                    raise ProtocolError('INVALID_INPUT: native request count exceeds plan')
        if len(seen)!=case['formal_requests']:
            raise ProtocolError('INVALID_INPUT: native request count mismatch')
        groups[group]=dict(model=group[0],dataset=group[1],requests=len(seen),
                           details_ref=path.relative_to(folder).as_posix(),details_sha256=checksum.hexdigest(),
                           timing_count=timing_count,max_request_seconds=max_latency if timing_count==len(seen) else None,
                           observed_tokens=token_stats)
        if timings is not None and timing_count==len(seen):
            from .performance_evidence import timing_window
            groups[group]['timing_window']=timing_window(timings,case.get('concurrency',1))
    if set(groups)!=expected:
        raise ProtocolError('INVALID_INPUT: missing native request groups')
    log_found=False
    for path in (folder/'outputs').glob('*/logs/**/*'):
        check()
        if path.is_file():
            confined(path);log_found=True;break
    if not log_found:
        raise ProtocolError('INVALID_INPUT: native worker logs missing')
    result=dict(schema_version='1.0',status='PASSED',mode='perf',request_groups=list(groups.values()),
                evidence_scope='native request success/count and worker log presence',performance_validity='UNVERIFIED')
    atomic_json(folder/'validated.json',result)
    return result
