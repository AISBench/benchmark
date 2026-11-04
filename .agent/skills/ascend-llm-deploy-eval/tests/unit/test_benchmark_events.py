import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from prepare_benchmark_tool import prepare
from task_events import read_events


class BenchmarkEventTests(unittest.TestCase):
    def test_wrapper_failure_closes_actual_warmup_stage(self):
        source=Path(__file__).resolve().parents[2]/'assets/aisbench-prefix'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            manifest=prepare(source,root/'tool')
            self.assertFalse((root/'tool/aisbench_test.py').exists())
            code=Path(manifest['aisbench_test.py']['source']).read_text(encoding='utf-8')
            compile(code,'aisbench_test.py','exec')
            hooks=code.split('import os, errno',1)[0]
            scope={'subprocess':subprocess}
            exec(hooks,scope)
            with patch.dict(os.environ,{'BENCH_EVENT_ROOT':str(root),'BENCH_EVENT_CONTEXT':json.dumps({'subcase_id':'s1','attempt':2})}):
                scope['_benchmark_event']('warmup','doing')
                with self.assertRaises(subprocess.CalledProcessError):
                    scope['_benchmark_run']([sys.executable,'-c','raise SystemExit(2)'],check=True)
            events=read_events(root)['events']
            self.assertEqual([(e['stage'],e['status']) for e in events],[('warmup','doing'),('warmup','fail')])
            self.assertTrue(all(e['attempt']==2 for e in events))
