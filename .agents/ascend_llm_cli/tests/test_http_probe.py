from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import threading
import time
import unittest
from deploy_eval_v2.http_probe import request,wait_health,smoke_completion
from deploy_eval_v2.process_guard import Cancelled
from deploy_eval_v2.protocol import ProtocolError


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass
    def do_GET(self):
        self.send_response(302 if self.path=='/redirect' else 200)
        self.end_headers()
        try:
            if self.path=='/slow':
                for _ in range(100):
                    self.wfile.write(b'x');self.wfile.flush();time.sleep(.02)
            else:
                self.wfile.write(b'x'*100 if self.path=='/large' else b'ok')
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):
            pass
    def do_POST(self):
        data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.last_payload=data
        if self.path.startswith('/delayed/'):
            time.sleep(10.2)
        self.send_response(200);self.end_headers()
        self.wfile.write(json.dumps({'choices':[{'text':'Hello'}]}).encode())


class HttpProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.server.daemon_threads=True
        cls.worker=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.worker.start()
        cls.url='http://127.0.0.1:'+str(cls.server.server_port)
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.worker.join()
    def test_real_loopback_health_and_smoke(self):
        flag=threading.Event()
        evidence=wait_health([self.url+'/health'],time.monotonic()+3,flag)
        self.assertEqual(evidence[0]['status'],200)
        result=smoke_completion(self.url,'model',time.monotonic()+3,flag)
        self.assertEqual(result['status'],'VERIFIED')
        self.assertEqual(self.server.last_payload['max_tokens'],16)
        self.assertEqual(self.server.last_payload['model'],'model')
    def test_body_budget_redirect_and_cancel(self):
        flag=threading.Event()
        with self.assertRaisesRegex(ProtocolError,'budget'):
            request(self.url+'/large',time.monotonic()+3,flag,max_bytes=16)
        with self.assertRaisesRegex(ProtocolError,'redirect'):
            wait_health([self.url+'/redirect'],time.monotonic()+3,flag)
        flag.set()
        with self.assertRaises(Cancelled):
            request(self.url+'/health',time.monotonic()+3,flag)
    def test_dripping_response_obeys_total_deadline(self):
        start=time.monotonic()
        with self.assertRaises(TimeoutError):
            request(self.url+'/slow',start+.15,threading.Event())
        self.assertLess(time.monotonic()-start,1)

    def test_smoke_uses_reviewed_budget_beyond_ten_seconds(self):
        result=smoke_completion(self.url+'/delayed','model',time.monotonic()+15,threading.Event())
        self.assertEqual(result['status'],'VERIFIED')
