from pathlib import Path
import tempfile
import threading
import time
import unittest
from deploy_eval_v2.metric_evidence import capture,summarize


def payload(queries,hits):
    return (f'vllm:prefix_cache_queries_total{{model="m"}} {queries}\n'
            f'vllm:prefix_cache_hits_total{{model="m"}} {hits}\n').encode()


class MetricEvidenceTests(unittest.TestCase):
    def test_zero_queries_and_multi_dp_do_not_prove_routing_coverage(self):
        for dp,queries,expected in [(1,0,'UNVERIFIED'),(1,10,'VERIFIED'),(2,10,'UNVERIFIED')]:
            with tempfile.TemporaryDirectory() as folder:
                instances=[dict(id='p',role='prefill',url='http://example.invalid/metrics',dp=dp)]
                flag=threading.Event()
                capture(instances,folder,'before',time.monotonic()+5,flag,fetch=lambda *a,**kw:(200,payload(0,0)))
                capture(instances,folder,'after',time.monotonic()+5,flag,fetch=lambda *a,**kw:(200,payload(queries,0)))
                report=summarize(folder,{'num_warmups':0})
                self.assertEqual(report['routing_coverage'],expected)
                self.assertEqual(report['warmup_route_coverage'],'NOT_REQUESTED')
                self.assertEqual(summarize(folder,{'num_warmups':1})['warmup_route_coverage'],'UNVERIFIED')

    def run_case(self,folder,first,last):
        instances=[dict(id='p',role='prefill',url='http://example.invalid/metrics',dp=1)]
        flag=threading.Event()
        capture(instances,folder,'before',time.monotonic()+5,flag,fetch=lambda *a,**kw:(200,first))
        capture(instances,folder,'after',time.monotonic()+5,flag,fetch=lambda *a,**kw:(200,last))
        return summarize(folder)

    def test_counter_delta_does_not_claim_overall_performance_validity(self):
        with tempfile.TemporaryDirectory() as folder:
            result=self.run_case(folder,payload(10,3),payload(30,8))
            self.assertEqual(result['status'],'VERIFIED')
            self.assertEqual(result['hit_ratio'],.25)
            self.assertEqual(result['performance_validity'],'UNVERIFIED')

    def test_missing_reset_or_inconsistent_counters_are_not_zeroes(self):
        for last in (b'',payload(5,1),payload(11,20),payload('NaN',4)):
            with tempfile.TemporaryDirectory() as folder:
                result=self.run_case(folder,payload(10,3),last)
                self.assertEqual(result['status'],'UNVERIFIED')
                self.assertNotIn('hit_ratio',result)

    def test_changed_evidence_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_case(folder,payload(10,3),payload(30,8))
            (Path(folder)/'metrics-after-0.prom').write_bytes(payload(100,50))
            self.assertEqual(summarize(folder)['status'],'UNVERIFIED')
