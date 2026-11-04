import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from aisbench_cli import write_config, commands, dataset_for, check_dataset

class NativeCLITests(unittest.TestCase):
    def test_existing_dataset_standard_config_and_logging(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'dataset.jsonl';data.write_text('\n'.join(json.dumps({'question':'prompt','answer':''}) for _ in range(8)))
            before=data.read_bytes()
            spec={'benchmark':{'datasets':{'default':{'path':str(data)}}},'model':{'path':str(root),'served_name':'model'}}
            case=dict(id='c1',formal_requests=8,output_tokens=32,concurrency=2,timeout_seconds=60,steady_state=True,num_warmups=3)
            path=write_config(spec,case,{'endpoint':'http://127.0.0.1:8000'},root)
            text=path.read_text();ast.parse(text)
            self.assertNotIn('ais_bench_prefix_cache',text)
            self.assertIn('batch_size=2',text)
            self.assertIn('summarizers.perf.stable_stage',text)
            self.assertEqual(data.read_bytes(),before)
            argv=commands('/bin/ais_bench',path,case,root)[0][1]
            self.assertEqual(argv[0],'/bin/ais_bench')
            self.assertEqual(argv[argv.index('--num-warmups')+1],'3')
            self.assertEqual(argv[argv.index('--work-dir')+1],str(root/'outputs'))
            self.assertNotIn('--debug',argv)
            case['warmup']=False
            self.assertEqual(commands('ais_bench',path,case,root)[0][1][-1],'0')
            with self.assertRaisesRegex(ValueError,'Missing.*binding'): dataset_for(spec['benchmark'],{'dataset_id':'unknown'})
            with self.assertRaisesRegex(ValueError,'requested'): check_dataset({'path':str(data),'input_column':'question','output_column':'answer'},9)
            data.write_text('{"question":"missing answer"}')
            with self.assertRaisesRegex(ValueError,'columns'): write_config(spec,case,{'endpoint':'http://localhost:8000'},root)

    def test_csv_and_custom_columns(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data.csv';path.write_text('prompt,label\nhello,world\n')
            dataset=dataset_for({'datasets':{'custom':{'path':str(path),'input_column':'prompt','output_column':'label'}}},{'dataset_id':'custom'})
            check_dataset(dataset,1)
