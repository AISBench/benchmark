"""Real Linux socket/subprocess integration, with explicitly simulated stages."""
import json
import os
from pathlib import Path
import socket
import socketserver
import subprocess
import sys
import tempfile
import time
import unittest
from deploy_eval_v2.remote_client import relay


@unittest.skipUnless(os.name=='posix' and hasattr(socket,'AF_UNIX') and
                     hasattr(socketserver,'ThreadingUnixStreamServer'),'requires Linux Unix sockets')
class LinuxSocketIntegrationTests(unittest.TestCase):
    def test_disposable_client_reconnect_replay_eval_release(self):
        with tempfile.TemporaryDirectory(prefix='v2-') as folder:
            root=Path(folder)
            with (root/'server.log').open('wb') as log:
                process=subprocess.Popen([sys.executable,'-m','deploy_eval_v2.server',
                    '--root',folder,'--task-id','t','--fake'],stdout=log,stderr=log,start_new_session=True)
                try:
                    deadline=time.monotonic()+10
                    while not (root/'.runtime/control.sock').exists():
                        if process.poll() is not None or time.monotonic()>deadline:
                            self.fail('Runtime did not expose its socket; inspect server log')
                        time.sleep(.02)
                    def query(kind,**fields):
                        return relay(root,dict(protocol_version='1.0',task_id='t',kind=kind,**fields))
                    deploy=dict(protocol_version='1.0',task_id='t',kind='deploy',request_id='d',
                                expected_version=0,plan_ref='frozen',wait_ms=1000)
                    # Separate client exits; runtime must remain alive for the next client.
                    request=root/'request.json';request.write_text(json.dumps(deploy)+'\n',encoding='utf-8')
                    with request.open('rb') as source,(root/'client.json').open('wb') as output:
                        code=subprocess.run([sys.executable,'-m','deploy_eval_v2.remote_client','--root',folder],
                            stdin=source,stdout=output,stderr=log,timeout=10).returncode
                    self.assertEqual(code,0)
                    response=json.loads((root/'client.json').read_text())
                    self.assertEqual(response['events'][-1]['type'],'deployment.ready')
                    self.assertIsNone(process.poll())
                    replay=relay(root,deploy)
                    self.assertEqual(replay['operation_id'],response['operation_id'])
                    for kind,event,fields in [('eval','evaluation.completed',dict(case_id='c')),
                                              ('release','release.completed',{})]:
                        status=query('get-status')
                        result=query(kind,request_id=kind,expected_version=status['task_version'],
                                     service_id='fake-service',wait_ms=1000,**fields)
                        self.assertEqual(result['events'][-1]['type'],event)
                    self.assertEqual(query('get-status')['task_state'],'RELEASED')
                    waited=query('wait-event',after_seq=result['next_cursor'],wait_ms=10,limit=10)
                    self.assertEqual(waited['status'],'WAIT_TIMEOUT')
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill();process.wait(timeout=5)
