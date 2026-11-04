import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import ast
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[2]/'assets/aisbench-prefix'
class RuntimeConfigTests(unittest.TestCase):
    def test_resolved_config_and_standalone(self):
        with patch.object(sys,'path',[str(SOURCE),*sys.path]),patch.dict(os.environ,{},clear=True):
            self.assertEqual(runpy.run_path(str(SOURCE/'runtime_config.py'))['case_arguments'](),{})
            with tempfile.TemporaryDirectory() as tmp:
                file=Path(tmp)/'config.json'
                for ratio in (0.9,1.0):
                    case=dict(input_tokens=131072,output_tokens=1024,formal_requests=400,concurrency=100,request_rate=0,prefix_num=1,prefix_ratio=ratio,seed=1,steady_state=True)
                    file.write_text(json.dumps({'case':case,'service':{'MODEL_PATH':'/weights'},'dp':8,'npu_num':8}))
                    os.environ['BENCH_CASE_CONFIG']=str(file)
                    config=runpy.run_path(str(SOURCE/'runtime_config.py')); args=config['case_arguments']()
                    self.assertEqual(config['DEFAULT_PERFORMANCE_TEST'],'stable_stage')
                    self.assertEqual((args['prefix_num'],args['repeat_rate'],args['concurrency'],args['output_len']),(1,str(ratio),'100','1024'))
    def test_child_failure_propagates_and_emits_fail(self):
        tree=ast.parse((SOURCE/'aisbench_test.py').read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_benchmark_run')
        events=[]
        scope={'subprocess':subprocess,'_event_stage':'warmup','_benchmark_event':lambda *a:events.append(a)}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'runtime','exec'),scope)
        with patch.object(subprocess,'run',side_effect=subprocess.CalledProcessError(7,'child')):
            with self.assertRaises(subprocess.CalledProcessError): scope['_benchmark_run'](['child'],check=True)
        self.assertEqual(events,[('warmup','fail')])
if __name__=='__main__': unittest.main()
