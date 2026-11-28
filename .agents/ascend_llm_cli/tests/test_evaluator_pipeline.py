"""Local pipeline integration; benchmark/preflight are explicit fixture processes.

Uses real loopback health, child processes, native files and result validators.
It does not emulate or claim an AISBench/model acceptance run.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from deploy_eval_v2.evaluate import execute_external_case
from deploy_eval_v2.process_guard import run_logged
from deploy_eval_v2.plan_model import normalize_external_plan
from test_plan_model import external_plan, eval_bindings


class EvaluatorPipelineTests(unittest.TestCase):
    def test_real_health_child_native_validation_pipeline(self):
        class Health(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(b'healthy')
            def log_message(self,*args): pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Health)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary); folder=root/'case'
                plan=external_plan();plan['cases'][0].update(concurrency=2,formal_requests=20,num_warmups=0)
                plan['service']['endpoint']='http://127.0.0.1:'+str(server.server_port)
                plan,_=normalize_external_plan(plan)
                bound=eval_bindings()
                tester=dict(bound['testers']['tester'],python=sys.executable,aisbench=sys.executable,
                            environment={'V2_TEST_BOUND_ENV':'verified'},model_path=str(root),datasets={'synthetic':{'type':'jsonl'}})
                script=root/'fixture.py'
                script.write_text('''import json,sys,os
assert os.environ["V2_TEST_BOUND_ENV"] == "verified"
from pathlib import Path
root=Path(sys.argv[1])
if sys.argv[2]=='preflight':
 (root/'preflight.json').write_text('{"status":"VERIFIED"}')
else:
 p=root/'outputs/run/performances/service/dataset_details.jsonl'; p.parent.mkdir(parents=True)
 p.write_text('\\n'.join(json.dumps(dict(id=i,success=True,start_time=i*.5,end_time=i*.5+.8)) for i in range(20)))
 values={'Benchmark Duration':'10300 ms','Total Requests':20,'Success Requests':20,'Failed Requests':0,'Max Concurrency':2,'Request Throughput':'1.9417 req/s'}
 p.with_name('dataset.json').write_text(json.dumps({k:{'total':v} for k,v in values.items()}))
 log=root/'outputs/run/logs/worker.log';log.parent.mkdir(parents=True);log.write_text('fixture workload')
print('fixture only')
''',encoding='utf-8')
                def prepare(*args,**kwargs):
                    folder.mkdir()
                    (folder/'native-expectations.json').write_text('[["service","dataset"]]')
                    return plan['cases'][0],tester,[sys.executable,str(script),str(folder),'benchmark']
                calls=[]
                def run(argv,log,deadline,cancelled,env=None):
                    calls.append(log.name)
                    if log.name=='preflight.log':
                        argv=[sys.executable,str(script),str(folder),'preflight']
                    return run_logged(argv,log,deadline,cancelled,env=env)
                with patch('deploy_eval_v2.evaluate.require_linux'),patch('deploy_eval_v2.evaluate.prepare_case',side_effect=prepare),patch('deploy_eval_v2.evaluate.run_logged',side_effect=run):
                    result=execute_external_case(plan,bound,'c',folder,threading.Event(),verified_local_host='example.invalid')
                self.assertEqual(calls,['preflight.log','aisbench.log'])
                self.assertEqual(result['execution_status'],'SUCCEEDED')
                self.assertEqual(result['performance_validity'],'VERIFIED')
                self.assertTrue((folder/'health.json').exists())
                self.assertEqual(json.loads((folder/'validated.json').read_text())['request_groups'][0]['requests'],20)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)
