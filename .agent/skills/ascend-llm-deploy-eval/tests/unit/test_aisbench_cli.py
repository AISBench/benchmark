import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from aisbench_cli import write_scenario, commands

class NativeCLITests(unittest.TestCase):
    def test_scenario_maps_load_and_uses_repository_parser(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); corpus=root/'corpus.jsonl';corpus.write_text('{"question":"example"}')
            spec={'benchmark':{'corpus_path':str(corpus),'block_size':16,'prefill_dp':2},'model':{'path':str(root),'served_name':'model'}}
            case=dict(id='c1',formal_requests=8,input_tokens=1024,output_tokens=32,concurrency=2,prefix_ratio=.5,timeout_seconds=60,steady_state=True)
            service={'endpoint':'http://127.0.0.1:8000','prefill_url':'http://127.0.0.1:8001'}
            path=write_scenario(spec,case,service,root);data=json.loads(path.read_text())
            self.assertEqual(data['aisbench']['model']['batch_size'],2)
            self.assertEqual(data['service']['metrics_url'],'http://127.0.0.1:8001/metrics')
            self.assertEqual(data['requests']['count'],8)
            self.assertIn('summarizers.perf.stable_stage', (root/'aisbench-config.py').read_text())
            self.assertEqual([stage for stage,_ in commands('/bin/ais-bench-prefix-cache',path)],['prepare','benchmark'])
            repo=Path(__file__).resolve().parents[5]
            plugin=repo/'plugins/prefix_cache'
            if plugin.is_dir():
                sys.path.insert(0,str(plugin))
                from ais_bench_prefix_cache.scenario import load_scenario
                parsed=load_scenario(path)
                self.assertEqual(parsed.dp_size,2)
            case['reset_cache']=False
            with self.assertRaisesRegex(ValueError,'reset_cache'): write_scenario(spec,case,service,root)
            corpus.unlink()
            with self.assertRaisesRegex(ValueError,'corpus'): write_scenario(spec,case,service,root)
