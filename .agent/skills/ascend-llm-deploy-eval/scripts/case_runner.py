"""One AISBench attempt inside the owned container; always release its client/links."""
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
from tool_links import cleanup as cleanup_links


def run(root,folder):
    spec_path=root/'resolved.json'
    spec=json.loads(spec_path.read_text())
    case=json.loads((folder/'case.json').read_text())
    context = {**case.get('_event_context', {'subcase_id':case['id']}), 'attempt':case.get('_attempt',1)}
    def event(stage, status, **values):
        emit(root, stage, status, **context, **values)
    service=json.loads((root/'scripts/service.json').read_text())
    reset_targets={i:url.rsplit('/reset_prefix_cache',1)[0] for i,url in enumerate(service['reset_urls'])} if 'reset_urls' in service else {p:f'http://127.0.0.1:{p}' for p in service['reset_ports']}
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def http(port,path,method='GET'):
        with opener.open(urllib.request.Request(reset_targets[port]+path,method=method),timeout=30) as r:
            return r.read().decode()
    case=dict(case,prefill_dp=spec['benchmark']['prefill_dp'],reset_ports=list(reset_targets))
    (folder/'case.json').write_text(json.dumps(case,indent=2))
    work=Path(spec['benchmark']['aisbench_work_path'])
    links=[work/'ais_bench/datasets/gsm8k/test.jsonl',work/'ais_bench/benchmark/configs/models/vllm_api/vllm_api_chat_temp.py']
    # Existing files are never ours to remove. Stop without retry if ownership unclear.
    for link in links:
        if link.exists() or link.is_symlink():
            (folder/'cleanup.json').write_text(json.dumps({'released':False,'error':'Existing link/file: '+str(link)}))
            raise ValueError('Existing tool link/file; reconcile before running')
    tool=folder/'tool'; datasets=folder/'datasets'; datasets.mkdir()
    from urllib.parse import urlsplit
    endpoint=urlsplit(service['endpoint'])
    config={'DATASET_PATH':str(datasets),'WORK_PATH':str(work),'MODEL_NAME':spec['model']['served_name'],
        'MODEL_PATH':spec['model']['path'],'HOST_IP':endpoint.hostname,'HOST_PORT':str(endpoint.port),
        'URL':'','API_KEY':'','DEFAULT_PERFORMANCE_TEST':'stable_stage' if case.get('steady_state', False) else 'default_perf','OUTPUT_DIR':'./outputs',
        'POD_INFO':[service['prefill_url'].split('://',1)[-1] if 'prefill_url' in service else f"127.0.0.1:{service['prefill_port']}"]}
    config.pop('DEFAULT_PERFORMANCE_TEST')
    config_path=folder/'tool-config.json'
    config_path.write_text(json.dumps({'case':case,'service':config,'dp':spec['benchmark']['prefill_dp'],'npu_num':len(service['npu_ids'])},indent=2))
    process=None; errors=[]; result=None; failure=None; started=time.monotonic()
    current_stage = 'cache_reset'
    try:
        event('cache_reset', 'doing' if case['reset_cache'] else 'skipped')
        if case['reset_cache']:
            reset_instances(folder, list(reset_targets), http, reset_targets)
            event('cache_reset', 'done')
        for port in reset_targets:
            (folder/f'metrics-{port}-before.txt').write_text(http(port,'/metrics'))
        current_stage = 'testing_detail'
        argv=['python3',str(Path(spec['benchmark']['tool_dir'])/'aisbench_test.py')]
        (folder/'command.json').write_text(json.dumps(argv))
        with (folder/'runner-diagnostics.log').open('w') as log:
            process=subprocess.Popen(argv,cwd=tool,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ, 'BENCH_CASE_CONFIG':str(config_path), 'BENCH_LINK_MANIFEST':str(folder/'tool-links.json'), 'BENCH_DATASET_CACHE':str(Path(spec.get('storage', {}).get('root', root))/'dataset-cache'), 'BENCH_EVENT_ROOT':str(root), 'BENCH_EVENT_CONTEXT':json.dumps(context),
                     'PYTHONPATH':str(Path(__file__).parent)+os.pathsep+os.environ.get('PYTHONPATH','')})
            identity={'pid':process.pid,'start_ticks':ticks(process.pid)}
            (folder/'client.json').write_text(json.dumps(identity))
            deadline=time.monotonic()+case['timeout_seconds']
            while True:
                if spec.get('storage'):
                    from disk_guard import check as disk_check
                    disk_check(spec['storage']['root'], spec['storage']['subtasks'], tool.resolve().parent.parent)
                remaining=deadline-time.monotonic()
                if remaining<=0: raise subprocess.TimeoutExpired(argv,case['timeout_seconds'])
                try:
                    code=process.wait(timeout=min(30,remaining)); break
                except subprocess.TimeoutExpired:
                    continue
        if code == 0:
            (folder/'runner-diagnostics.log').unlink()
        if code: raise RuntimeError(f'Benchmark exited {code}; see tool/aisbench.log and runner-diagnostics.log')
        for port in reset_targets:
            (folder/f'metrics-{port}-after.txt').write_text(http(port,'/metrics'))
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
        errors.extend(cleanup_links(links, folder/'tool-links.json'))
        (folder/'cleanup.json').write_text(json.dumps({'released':not errors,'errors':errors,'level':'case'},indent=2))
    event('client_cleanup', 'done' if not errors else 'fail')
    summary={'status':'PASSED' if result and not errors else 'FAILED','error':failure,'cleanup_errors':errors,'seconds':time.monotonic()-started}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)
    return 0 if summary['status']=='PASSED' else 2

if __name__=='__main__': raise SystemExit(run(Path(sys.argv[1]),Path(sys.argv[2])))
