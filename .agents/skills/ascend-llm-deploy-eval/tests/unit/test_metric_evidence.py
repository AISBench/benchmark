import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from metric_evidence import capture, summarize


class MetricEvidenceTests(unittest.TestCase):
    def test_aggregate_uses_counter_weights_and_rejects_reset(self):
        def raw(q,h):
            return f'vllm:prefix_cache_queries_total{{engine="0"}} {q}\nvllm:prefix_cache_hits_total{{engine="0"}} {h}\n'
        instances=[{'id':'p1','role':'prefill','url':'p1'}, {'id':'p2','role':'prefill','url':'p2'}]
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            capture(instances,folder,'before',lambda url:raw(10,5))
            capture(instances,folder,'after',lambda url:raw(110,55) if url=='p1' else raw(910,95))
            result=summarize(folder)
            self.assertEqual(result['hit_ratio'],0.14)
            self.assertEqual(result['warmup_route_coverage'],'UNVERIFIED')
            capture(instances,folder,'after',lambda url:raw(0,0))
            result=summarize(folder)
            self.assertEqual(result['status'],'UNVERIFIED')
            self.assertNotIn('hit_ratio',result)

    def test_partial_failure_preserves_other_instances(self):
        def fetch(url):
            if url=='failed': raise OSError('offline')
            return 'vllm:prefix_cache_hits_total{engine="0"} 123\n'
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            report=capture([{'id':'p1','url':'ok'}, {'id':'p2','url':'failed'},
                            {'id':'p3','url':'ok'}],folder,'before',fetch)
            self.assertEqual([r['status'] for r in report['instances']],
                             ['CAPTURED','UNVERIFIED','CAPTURED'])
            self.assertIn('123',Path(report['instances'][2]['path']).read_text())
            self.assertEqual(json.loads((folder/'metrics-before.json').read_text())['routing_coverage'],'UNVERIFIED')
