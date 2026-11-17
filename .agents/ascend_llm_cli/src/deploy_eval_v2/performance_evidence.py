"""Validate native summary evidence; unknown formats never become zero metrics.

Field/unit contract inspected at AISBench/benchmark aff1554354fbf19dccbdf2199cc3c22119f815a2.
No upstream implementation code is imported or copied.
"""
import hashlib
import math
from pathlib import Path
from .durable import atomic_json
from .json_io import load_json
from .process_guard import remaining,Cancelled


def timing_window(timings,concurrency):
    events=sorted((stamp,index,kind) for index,pair in enumerate(timings) for kind,stamp in enumerate(pair))
    active=0;peak=0;reached=0;first=None;last=None
    for stamp,_,kind in events:
        active+=1 if kind==0 else -1
        peak=max(peak,active)
        if active==concurrency:
            if kind==0:
                reached+=1
                if reached==2: first=stamp
            last=stamp
    result=dict(total_seconds=max(end for _,end in timings)-min(start for start,_ in timings),peak_concurrency=peak)
    if first is not None and last is not None and last>first:
        result.update(stable_seconds=last-first,stable_requests=sum(first<=start<=last for start,_ in timings))
    return result


def summarize(folder,case,native,*,deadline=None,cancelled=None):
    root=Path(folder).resolve();records=[]
    stage='stable' if case.get('steady_state') else 'total'
    def check():
        if deadline is not None:
            remaining(deadline)
        if cancelled is not None and cancelled.is_set():
            raise Cancelled('Performance evidence cancelled')
    def number(value,unit=None):
        if unit:
            if not isinstance(value,str) or not value.endswith(' '+unit):
                raise ValueError('SUMMARY_UNITS_UNRECOGNIZED')
            value=value[:-len(unit)-1]
        if isinstance(value,bool):
            raise ValueError('SUMMARY_VALUE_INVALID')
        value=float(value)
        if not math.isfinite(value) or value<0:
            raise ValueError('SUMMARY_VALUE_INVALID')
        return value
    for group in native['request_groups']:
        check()
        item=dict(model=group['model'],dataset=group['dataset'],stage=stage,status='UNVERIFIED')
        try:
            details=root/group['details_ref']
            if details.is_symlink() or not details.resolve().is_relative_to(root):
                raise ValueError('DETAIL_PATH_INVALID')
            hasher=hashlib.sha256()
            with details.open('rb') as stream:
                for block in iter(lambda:stream.read(65536),b''):
                    check();hasher.update(block)
            if hasher.hexdigest()!=group['details_sha256']:
                raise ValueError('DETAIL_EVIDENCE_CHANGED')
            path=details.with_name(group['dataset']+'.json')
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('SUMMARY_PATH_INVALID')
            summary=load_json(path,1024*1024)
            def metric(key,unit=None):
                return number(summary[key][stage],unit)
            duration=metric('Benchmark Duration','ms')/1000
            total=metric('Total Requests');success=metric('Success Requests');failed=metric('Failed Requests')
            concurrency=metric('Max Concurrency')
            throughput=metric('Request Throughput','req/s')
            if (not duration>0 or any(value!=int(value) for value in (total,success,failed,concurrency)) or
                    not 0<total<=group['requests'] or success!=total or failed!=0 or concurrency!=case['concurrency']):
                raise ValueError('SUMMARY_COUNT_OR_WINDOW_INVALID')
            if not case.get('steady_state') and total!=group['requests']:
                raise ValueError('SUMMARY_COUNT_MISMATCH')
            if not math.isclose(throughput,success/duration,rel_tol=.01,abs_tol=.001):
                raise ValueError('SUMMARY_THROUGHPUT_INCONSISTENT')
            item.update(duration_seconds=duration,requests=int(total),request_throughput=throughput,
                        summary_ref=path.relative_to(root).as_posix())
            window=group.get('timing_window')
            if window and not case.get('steady_state') and not math.isclose(duration,window['total_seconds'],rel_tol=.01,abs_tol=.001):
                raise ValueError('SUMMARY_TIMING_MISMATCH')
            if case.get('steady_state'):
                latency=group.get('max_request_seconds')
                if latency is None or not window:
                    raise ValueError('FULL_REQUEST_TIMING_UNAVAILABLE')
                if ('stable_seconds' not in window or window['peak_concurrency']!=case['concurrency'] or
                        total!=window['stable_requests'] or not math.isclose(duration,window['stable_seconds'],rel_tol=.01,abs_tol=.001)):
                    raise ValueError('SUMMARY_TIMING_MISMATCH')
                if latency*3>=duration:
                    raise ValueError('STEADY_WINDOW_TOO_SHORT')
            item['status']='VERIFIED'
        except (OSError,ValueError,KeyError,TypeError,OverflowError) as error:
            if isinstance(error,TimeoutError):
                raise
            # Stable short codes; no raw source data or unbounded exception text.
            reason=str(error)
            item['reason']=reason if reason in {
                'SUMMARY_UNITS_UNRECOGNIZED','SUMMARY_VALUE_INVALID','DETAIL_PATH_INVALID','DETAIL_EVIDENCE_CHANGED',
                'SUMMARY_PATH_INVALID','SUMMARY_COUNT_OR_WINDOW_INVALID','SUMMARY_COUNT_MISMATCH',
                'SUMMARY_THROUGHPUT_INCONSISTENT','FULL_REQUEST_TIMING_UNAVAILABLE','STEADY_WINDOW_TOO_SHORT',
                'SUMMARY_TIMING_MISMATCH'} else 'NATIVE_SUMMARY_UNAVAILABLE_OR_UNRECOGNIZED'
        records.append(item)
    report=dict(schema_version='1.0',status='VERIFIED' if records and all(i['status']=='VERIFIED' for i in records) else 'UNVERIFIED',
                groups=records,scope='native request performance; not prefix routing coverage')
    atomic_json(root/'performance-evidence.json',report)
    return report
