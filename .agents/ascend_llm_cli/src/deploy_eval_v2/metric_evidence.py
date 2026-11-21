"""Bounded per-instance Prometheus evidence, adapted from the frozen baseline.

Source semantics: 74a7feffb76c32b23658d37ea681784b3a9b2474,
scripts/metric_evidence.py. Failed reads are never interpreted as zero counters.
"""
import hashlib
import math
import os
from pathlib import Path
import re
import time
from .durable import atomic_json
from .http_probe import request
from .json_io import load_json
from .process_guard import Cancelled,remaining
from .protocol import ProtocolError


def capture(instances,folder,phase,deadline,cancelled,*,fetch=request):
    if phase not in ('before','after') or len(instances)>128:
        raise ProtocolError('INVALID_INPUT: metric capture')
    folder=Path(folder)
    records=[]
    for index,instance in enumerate(instances):
        if cancelled.is_set():
            raise Cancelled('Metric capture cancelled')
        remaining(deadline)
        record=dict(instance,started_unix=time.time())
        try:
            status,raw=fetch(instance['url'],min(deadline,time.monotonic()+5),cancelled,max_bytes=65536)
            if status!=200 or len(raw)>65536:
                raise ValueError('metric HTTP response')
            raw.decode('utf-8')
            reference=f'metrics-{phase}-{index}.prom'
            with (folder/reference).open('xb') as stream:
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
            record.update(status='CAPTURED',artifact_ref=reference,sha256=hashlib.sha256(raw).hexdigest())
        except Cancelled:
            raise
        except Exception:
            record.update(status='UNVERIFIED',error_code='METRIC_CAPTURE_UNVERIFIED')
        record['finished_unix']=time.time()
        records.append(record)
    result=dict(schema_version='1.0',phase=phase,instances=records,
                window='native CLI including native warmups',routing_coverage='UNVERIFIED')
    atomic_json(folder/('metrics-'+phase+'.json'),result)
    return result


def counters(text,metric):
    result={}
    for line in text.splitlines():
        match=re.fullmatch(re.escape(metric)+r'(\{.*\})?\s+(\S+)(?:\s+\S+)?',line)
        if match:
            value=float(match[2]);labels=match[1] or ''
            if labels in result or not math.isfinite(value) or value<0:
                raise ValueError('invalid counter series')
            result[labels]=value
    if not result:
        raise ValueError('missing counter')
    return result


def summarize(folder,case=None):
    folder=Path(folder)
    before=load_json(folder/'metrics-before.json')
    after=load_json(folder/'metrics-after.json')
    def index(report):
        result={}
        for item in report['instances']:
            if item['id'] in result:
                raise ValueError('duplicate instance')
            result[item['id']]=item
        return result
    def raw(item):
        if item['status']!='CAPTURED':
            raise ValueError('missing capture')
        path=folder/item['artifact_ref']
        if path.is_symlink() or path.parent.resolve()!=folder.resolve():
            raise ValueError('metric path')
        with path.open('rb') as stream:
            data=stream.read(65537)
        if len(data)>65536 or hashlib.sha256(data).hexdigest()!=item['sha256']:
            raise ValueError('metric evidence changed')
        return data.decode('utf-8')
    records=[]
    complete=False
    try:
        initial,final=index(before),index(after)
        if initial.keys()!=final.keys():
            raise ValueError('instance coverage changed')
        for key,old in initial.items():
            if old['role'] not in ('prefill','mixed'):
                continue
            record=dict(id=key,status='UNVERIFIED')
            try:
                new=final[key]
                if any(old.get(field)!=new.get(field) for field in ('role','url','dp')):
                    raise ValueError('metric binding changed')
                previous,current=raw(old),raw(new)
                deltas={}
                for kind in ('queries','hits'):
                    a=counters(previous,'vllm:prefix_cache_'+kind+'_total')
                    b=counters(current,'vllm:prefix_cache_'+kind+'_total')
                    if a.keys()!=b.keys():
                        raise ValueError('series changed')
                    deltas[kind]={label:b[label]-a[label] for label in a}
                    if any(value<0 for value in deltas[kind].values()):
                        raise ValueError('counter reset')
                if deltas['queries'].keys()!=deltas['hits'].keys() or any(deltas['hits'][label]>value for label,value in deltas['queries'].items()):
                    raise ValueError('counter relationship invalid')
                record.update(status='VERIFIED',queries=sum(deltas['queries'].values()),hits=sum(deltas['hits'].values()),deltas=deltas)
                record['routing_coverage']='VERIFIED' if old.get('dp',1)==1 and sum(deltas['queries'].values())>0 else 'UNVERIFIED'
                if record['routing_coverage']!='VERIFIED':
                    record['routing_reason']='DP_SERIES_IDENTITY_UNAVAILABLE' if old.get('dp',1)>1 else 'NO_OBSERVED_QUERIES'
            except Exception:
                record['error_code']='PREFIX_METRIC_UNVERIFIED'
            records.append(record)
        complete=bool(records) and all(item['status']=='VERIFIED' for item in records)
    except Exception:
        pass
    report=dict(schema_version='1.0',status='VERIFIED' if complete else 'UNVERIFIED',instances=records,
                window=before['window'],warmup_route_coverage='UNVERIFIED',performance_validity='UNVERIFIED')
    report['routing_coverage']='VERIFIED' if complete and all(item.get('routing_coverage')=='VERIFIED' for item in records) else 'UNVERIFIED'
    if case is not None and case['num_warmups']==0:
        report['warmup_route_coverage']='NOT_REQUESTED'
    else:
        report['warmup_reason']='COMBINED_NATIVE_WINDOW_DOES_NOT_ISOLATE_WARMUP'
    if complete:
        queries=sum(item['queries'] for item in records);hits=sum(item['hits'] for item in records)
        report.update(queries=queries,hits=hits,hit_ratio=hits/queries if queries else None)
    atomic_json(folder/'prefix-evidence.json',report)
    return report
