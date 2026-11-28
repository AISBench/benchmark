"""Explicit per-instance reset; baseline cache_reset.py semantics, commit 74a7feff.

Never send reset through a guessed service/proxy URL. All targets and authority
are frozen, all instances must be idle before the first mutating request.
"""
import hashlib
import json
import math
from pathlib import Path
import re
from .durable import atomic_json
from .http_probe import request
from .json_io import unique_object,reject_constant
from .process_guard import remaining,Cancelled
from .protocol import ProtocolError,now


def require_idle(text):
    for metric in ('num_requests_running','num_requests_waiting'):
        values=[float(v) for v in re.findall(r'^vllm:'+metric+r'(?:\{[^\n]*\})?\s+(\S+)',text,re.M)]
        if not values or any(not math.isfinite(v) or v!=0 for v in values):
            raise ProtocolError('INVALID_INPUT: cache reset target not verified idle')


def reset_instances(service,folder,deadline,cancelled,*,fetch=request):
    targets=service.get('reset_targets',[])
    if not service['permissions']['reset_cache'] or not targets:
        raise ProtocolError('INVALID_INPUT: explicit cache reset authority and targets required')
    folder=Path(folder)
    report=dict(schema_version='1.0',status='FAILED',scope='local_prefix_cache',time=now(),instances=[])
    def save(index,phase,raw):
        name=f'reset-{index}-{phase}.txt'
        (folder/name).write_bytes(raw)
        return dict(artifact_ref=name,sha256=hashlib.sha256(raw).hexdigest())
    try:
        for index,target in enumerate(targets):
            remaining(deadline)
            item=dict(id=target['id'],url=target['url'],status='CHECKING_IDLE')
            report['instances'].append(item)
            status,raw=fetch(target['metrics_url'],deadline,cancelled,max_bytes=65536)
            item['idle_evidence']=save(index,'idle',raw)
            if status!=200:
                raise ProtocolError('INVALID_INPUT: cache reset metrics unavailable')
            require_idle(raw.decode('utf-8'))
            item['status']='IDLE'
        for index,(target,item) in enumerate(zip(targets,report['instances'])):
            remaining(deadline)
            item['status']='UNKNOWN_OUTCOME'
            # Persist intent before HTTP; never automatically retry a lost response.
            atomic_json(folder/'cache-reset.json',report)
            status,raw=fetch(target['url'],deadline,cancelled,payload={},max_bytes=8192)
            item['response_evidence']=save(index,'response',raw)
            response=json.loads(raw,object_pairs_hook=unique_object,parse_constant=reject_constant)
            if status!=200 or not isinstance(response,dict) or response.get('success') is not True:
                item['status']='FAILED'
                raise ProtocolError('INVALID_INPUT: cache reset not confirmed')
            item['status']='VERIFIED'
        report['status']='VERIFIED'
        return report
    finally:
        atomic_json(folder/'cache-reset.json',report)
