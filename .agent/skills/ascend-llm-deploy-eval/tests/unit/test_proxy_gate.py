import copy,json,os,sys,tempfile,threading,time,subprocess,unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from test_compact_entry import inputs
import run as entry

class GateTests(unittest.TestCase):
    def test_generated_cross_node_gate_and_opt_out(self):
        spec,bindings=inputs();dep=spec['deployments'][0]
        decode=next(i for i in dep['instances'] if i['role']=='decode');decode['placement']['node']='node_b'
        bindings['nodes']['node_b']=copy.deepcopy(bindings['nodes']['node_a'])
        bindings['nodes']['node_b']['host']='root@192.0.2.2'
        bindings['nodes']['node_b']['deployments'][dep['id']]['network']['ip']='192.0.2.2'
        for enabled in (True,False):
            spec['execution']['proxy_start_after_models_healthy']=enabled
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'job';norm,_=entry.generate(spec,bindings,root)
                scripts=[f.read_text() for f in (root/'scripts'/dep['id']).glob('*.sh') if 'exec python3 /proxy.py' in f.read_text()]
                self.assertEqual(len(scripts),1)
                self.assertEqual('Proxy startup blocked' in scripts[0],enabled)
                if enabled:
                    self.assertIn('192.0.2.1',scripts[0]);self.assertIn('192.0.2.2',scripts[0])
                self.assertEqual(norm['execution']['proxy_start_after_models_healthy'],enabled)

    @unittest.skipUnless(sys.platform.startswith('linux'),'Bash/curl behavior')
    def test_generated_gate_waits_for_both_models_and_times_out(self):
        ready=[False,False]
        servers=[]
        for index in range(2):
            def handler(n):
                class H(BaseHTTPRequestHandler):
                    def do_GET(self):self.send_response(200 if ready[n] else 503);self.end_headers()
                    def log_message(self,*args):pass
                return H
            server=ThreadingHTTPServer(('127.0.0.1',0),handler(index));servers.append(server)
            threading.Thread(target=server.serve_forever,daemon=True).start()
        urls=['http://127.0.0.1:%s/health'%s.server_port for s in servers]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);mark=root/'started';script=root/'proxy.sh'
                def generated_proxy(timeout,label):
                    spec,bindings=inputs();dep=spec['deployments'][0]
                    spec['execution']['startup_timeout_seconds']=timeout
                    node=bindings['nodes']['node_a'];node['deployments'][dep['id']]['network']['ip']='127.0.0.1'
                    proxy=root/'proxy.py';proxy.write_text('from pathlib import Path\nPath('+repr(str(mark))+').touch()\n')
                    node['paths']['proxy_program']=str(proxy)
                    models=[i for i in dep['instances'] if i['role']!='proxy']
                    for instance,server in zip(models,servers):
                        node['deployments'][dep['id']]['ports'][instance['id']]['http']=server.server_port
                    target=root/label;entry.generate(spec,bindings,target)
                    return next(f for f in (target/'scripts'/dep['id']).glob('*.sh') if 'Proxy startup blocked' in f.read_text())
                script=generated_proxy(3,'ready')
                env={**os.environ,'http_proxy':'http://127.0.0.1:1','HTTP_PROXY':'http://127.0.0.1:1','NO_PROXY':''}
                proc=subprocess.Popen(['bash',str(script)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
                try:
                    time.sleep(.25);self.assertFalse(mark.exists());ready[0]=True
                    time.sleep(.25);self.assertFalse(mark.exists());ready[1]=True
                    out,err=proc.communicate(timeout=5);self.assertEqual(proc.returncode,0,err);self.assertTrue(mark.exists())
                finally:
                    if proc.poll() is None:proc.kill();proc.wait()
                mark.unlink();ready[1]=False
                script=generated_proxy(1,'timeout')
                result=subprocess.run(['bash',str(script)],capture_output=True,timeout=3)
                self.assertNotEqual(result.returncode,0);self.assertFalse(mark.exists());self.assertIn(b'Proxy startup blocked',result.stderr)
        finally:
            for s in servers:s.shutdown();s.server_close()
