"""One AISBench attempt inside the owned container; always release its client process group."""

if __name__ == '__main__':
    raise SystemExit('Retired lifecycle entry. Use scripts/run.py --mode FULL|scripts; historical frozen jobs retain their own code.')
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from service_process import members,ticks
from validate_case import validate
from cache_reset import reset_instances
from task_events import emit
from aisbench_cli import write_config, commands
from metric_evidence import capture, summarize


def run(root,folder):
    spec_path=root/'resolved.json'
    spec=json.loads(spec_path.read_text())
    case=json.loads((folder/'case.json').read_text())
    if case.get('aisbench_config'):
        raise ValueError('Native config requires the unified run.py entry; legacy adapter cannot preserve it')
    context = {**case.get('_event_context', {'subcase_id':case['id']}), 'attempt':case.get('_attempt',1)}
    def event(stage, status, **values):
        emit(root, stage, status, **context, **values)
    service=json.loads((root/'scripts/service.json').read_text())
    reset_targets={i:url.rsplit('/reset_prefix_cache',1)[0] for i,url in enumerate(service['reset_urls'])} if 'reset_urls' in service else {p:f'http://127.0.0.1:{p}' for p in service['reset_ports']}
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def http(port,path,method='GET'):
        with opener.open(urllib.request.Request(reset_targets[port]+path,method=method),timeout=30) as r:
            return r.read().decode()
    case=dict(case,prefill_dp=spec['benchmark']['prefill_dp'],reset_ports=list(reset_targets) if case.get('reset_cache', False) else [])
    (folder/'case.json').write_text(json.dumps(case,indent=2))
    process=None; errors=[]; result=None; failure=None; started=time.monotonic()
    current_stage = 'cache_reset'
    try:
        config=write_config(spec,case,service,folder)
        event('cache_reset', 'doing' if case['reset_cache'] else 'skipped')
        if case['reset_cache']:
            if not reset_targets: raise ValueError('Cache reset requested but no reset endpoint configured')
            reset_instances(folder, list(reset_targets), http, reset_targets)
            event('cache_reset', 'done')
        for current_stage, argv in commands(spec['benchmark']['executable'], config, case, folder):
            capture(service.get('metric_instances', []), folder, 'before')
            event(current_stage, 'doing')
            (folder/(current_stage+'-command.json')).write_text(json.dumps(argv))
            with (folder/(current_stage+'.log')).open('w') as log:
                process=subprocess.Popen(argv,cwd=folder,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                (folder/'client.json').write_text(json.dumps({'pid':process.pid,'start_ticks':ticks(process.pid)}))
                deadline=time.monotonic()+case['timeout_seconds']
                while True:
                    if spec.get('storage'):
                        from disk_guard import check as disk_check
                        disk_check(spec['storage']['root'], spec['storage']['subtasks'], folder.parent)
                    remaining=deadline-time.monotonic()
                    if remaining<=0: raise subprocess.TimeoutExpired(argv,case['timeout_seconds'])
                    try:
                        code=process.wait(timeout=min(30,remaining)); break
                    except subprocess.TimeoutExpired:
                        continue
            (folder/'execution.json').write_text(json.dumps({'argv':argv,'returncode':code}),encoding='utf-8')
            if code: raise RuntimeError(f'Native CLI exited {code}; see {current_stage}.log')
            if members(process.pid): raise RuntimeError('Native CLI left child processes running')
            event(current_stage, 'done')
        event('validate', 'doing')
        current_stage = 'validate'
        result=validate(folder,case)
        event('validate', 'done')
    except Exception as exc:
        failure=repr(exc)
        event(current_stage, 'fail', error=failure)
    finally:
        event('client_cleanup', 'doing')
        if process is not None and members(process.pid):
            # This group was created by this invocation and cannot be confused with another client.
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); process.wait(timeout=10)
            deadline=time.monotonic()+10
            while members(process.pid) and time.monotonic()<deadline: time.sleep(.2)
            if members(process.pid):
                os.killpg(process.pid,signal.SIGKILL); time.sleep(1)
                if members(process.pid): errors.append('Client process group remains')
        (folder/'cleanup.json').write_text(json.dumps({'released':not errors,'errors':errors,'level':'case'},indent=2))
        try:
            capture(service.get('metric_instances', []), folder, 'after')
            summarize(folder)
        except Exception as exc:
            event('metrics', 'fail', error=repr(exc))
    event('client_cleanup', 'done' if not errors else 'fail')
    logs=sorted(str(path) for path in (folder/'outputs').glob('*/logs/**/*') if path.is_file())
    (folder/'log-index.json').write_text(json.dumps({'console':str(folder/'benchmark.log'),'native_logs':logs,'work_dir':str(folder/'outputs')},indent=2),encoding='utf-8')
    summary={'status':'PASSED' if result and not errors else 'FAILED','error':failure,'cleanup_errors':errors,'seconds':time.monotonic()-started,'log_index':str(folder/'log-index.json')}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)
    return 0 if summary['status']=='PASSED' else 2

if __name__=='__main__': raise SystemExit(run(Path(sys.argv[1]),Path(sys.argv[2])))
