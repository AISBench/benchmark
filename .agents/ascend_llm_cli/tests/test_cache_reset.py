import tempfile
import threading
import time
import unittest
from pathlib import Path
from deploy_eval_v2.cache_reset import reset_instances
from deploy_eval_v2.json_io import load_json
from deploy_eval_v2.protocol import ProtocolError


IDLE=b'vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n'


class CacheResetTests(unittest.TestCase):
    def service(self):
        return dict(permissions={'reset_cache':True},reset_targets=[dict(id=str(i),
            url=f'http://node-{i}/reset_prefix_cache',metrics_url=f'http://node-{i}/metrics') for i in range(2)])

    def test_all_idle_checked_before_reset_and_evidence_saved(self):
        calls=[]
        def fetch(url,*args,**kwargs):
            calls.append(url)
            return 200,IDLE if url.endswith('/metrics') else b'{"success":true}'
        with tempfile.TemporaryDirectory() as folder:
            report=reset_instances(self.service(),folder,time.monotonic()+5,threading.Event(),fetch=fetch)
            self.assertEqual(report['status'],'VERIFIED')
            self.assertTrue(all(url.endswith('/metrics') for url in calls[:2]))
            self.assertEqual(load_json(Path(folder)/'cache-reset.json'),report)
            self.assertTrue(all('sha256' in item['response_evidence'] for item in report['instances']))

    def test_busy_instance_blocks_every_mutation(self):
        calls=[]
        def fetch(url,*args,**kwargs):
            calls.append(url)
            return 200,IDLE if 'node-0' in url else IDLE.replace(b'running 0',b'running 1')
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ProtocolError):
                reset_instances(self.service(),folder,time.monotonic()+5,threading.Event(),fetch=fetch)
            self.assertTrue(all(url.endswith('/metrics') for url in calls))
            self.assertEqual(load_json(Path(folder)/'cache-reset.json')['status'],'FAILED')

    def test_lost_response_is_not_retried_and_does_not_reset_next_target(self):
        posts=[]
        def fetch(url,*args,**kwargs):
            if url.endswith('/metrics'):
                return 200,IDLE
            posts.append(url)
            raise OSError('lost response')
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(OSError):
                reset_instances(self.service(),folder,time.monotonic()+5,threading.Event(),fetch=fetch)
            report=load_json(Path(folder)/'cache-reset.json')
            self.assertEqual(len(posts),1)
            self.assertEqual(report['instances'][0]['status'],'UNKNOWN_OUTCOME')
            self.assertEqual(report['instances'][1]['status'],'IDLE')

    def test_no_permission_no_http(self):
        service=self.service();service['permissions']['reset_cache']=False
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ProtocolError):
                reset_instances(service,folder,time.monotonic()+5,threading.Event(),
                                fetch=lambda *a,**kw:self.fail('unauthorized HTTP'))
