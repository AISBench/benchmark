"""Per-instance raw Prometheus evidence; failed reads never imply zero counters."""
import json
import math
import re
from pathlib import Path
import time
import urllib.request


def capture(instances, folder, phase, fetch=None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def read(url):
        with opener.open(url, timeout=10) as response:
            return response.read().decode('utf-8')
    fetch = fetch or read
    records = []
    for index, instance in enumerate(instances):
        record = {**instance, 'started_unix': time.time()}
        try:
            raw = fetch(instance['url'])
            path = folder / f'metrics-{phase}-{index}.prom'
            path.write_text(raw, encoding='utf-8')
            record.update(status='CAPTURED', path=str(path))
        except Exception as exc:
            record.update(status='UNVERIFIED', error=repr(exc))
        record['finished_unix'] = time.time()
        records.append(record)
    report = {'phase': phase, 'window': 'native CLI including native warmups',
              'instances': records, 'routing_coverage': 'UNVERIFIED'}
    (folder / f'metrics-{phase}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def counters(text, metric):
    values={}
    for line in text.splitlines():
        match=re.fullmatch(re.escape(metric)+r'(\{.*\})?\s+(\S+)(?:\s+\S+)?',line)
        if match:
            value=float(match[2]);labels=match[1] or ''
            if labels in values or not math.isfinite(value) or value<0:
                raise ValueError('Invalid or duplicate counter series')
            values[labels]=value
    if not values: raise ValueError('Missing metric '+metric)
    return values


def summarize(folder):
    before=json.loads((folder/'metrics-before.json').read_text())
    after=json.loads((folder/'metrics-after.json').read_text())
    previous={r['id']:r for r in before['instances']}
    records=[]
    for final in after['instances']:
        if final.get('role') not in ('prefill','mixed'): continue
        record={'id':final['id'],'status':'UNVERIFIED'}
        try:
            initial=previous[final['id']]
            if initial['status']!='CAPTURED' or final['status']!='CAPTURED':
                raise ValueError('Incomplete metric capture')
            deltas={}
            for kind in ('queries','hits'):
                metric='vllm:prefix_cache_'+kind+'_total'
                old=counters(Path(initial['path']).read_text(),metric)
                new=counters(Path(final['path']).read_text(),metric)
                if old.keys()!=new.keys(): raise ValueError('Counter series changed')
                delta={k:new[k]-old[k] for k in old}
                if any(v<0 for v in delta.values()): raise ValueError('Counter reset')
                deltas[kind]=delta
            if deltas['queries'].keys()!=deltas['hits'].keys(): raise ValueError('Counter label mismatch')
            if any(deltas['hits'][k]>v for k,v in deltas['queries'].items()): raise ValueError('Hits exceed queries')
            record.update(status='VERIFIED',deltas=deltas,
                          queries=sum(deltas['queries'].values()),hits=sum(deltas['hits'].values()))
        except Exception as exc:
            record['error']=repr(exc)
        records.append(record)
    complete=bool(records) and len(records)==sum(r.get('role') in ('prefill','mixed') for r in before['instances']) and all(r['status']=='VERIFIED' for r in records)
    report={'instances':records,'window':before['window'],'status':'VERIFIED' if complete else 'UNVERIFIED',
            'warmup_route_coverage':'UNVERIFIED','aggregation':'sum(hit deltas) / sum(query deltas)'}
    if complete:
        queries=sum(r['queries'] for r in records);hits=sum(r['hits'] for r in records)
        report.update(queries=queries,hits=hits,hit_ratio=hits/queries if queries else None)
    (folder/'prefix-evidence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
