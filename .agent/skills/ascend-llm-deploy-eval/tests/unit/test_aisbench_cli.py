import ast
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from aisbench_cli import write_config, commands, dataset_for, check_dataset

class NativeCLITests(unittest.TestCase):
    def test_generated_binding_checks_evidence_and_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data.jsonl';data.write_text('{"question":"abcd","answer":""}\n')
            manifest=root/'generation.json'
            manifest.write_text(json.dumps({'dataset_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
                'audit':{'exact_lengths':True,'target_tokens':4,'token_lengths':[4]},
                'parameters':{'prefix_ratio':0.2}}))
            benchmark={'datasets':{'default':{'path':str(data),'generation_manifest':str(manifest)}}}
            case={'input_tokens':4,'shared_prefix_ratio':0.2,'formal_requests':1}
            dataset_for(benchmark,case)
            with self.assertRaisesRegex(ValueError,'prefix target'):
                dataset_for(benchmark,dict(case,shared_prefix_ratio=0))
            with self.assertRaisesRegex(ValueError,'input length'):
                dataset_for(benchmark,dict(case,input_tokens=5))
            data.write_text('changed')
            with self.assertRaisesRegex(ValueError,'digest'):
                dataset_for(benchmark,case)

    def test_existing_dataset_standard_config_and_logging(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'dataset.jsonl';data.write_text('\n'.join(json.dumps({'question':'prompt','answer':''}) for _ in range(8)))
            before=data.read_bytes()
            spec={'benchmark':{'datasets':{'default':{'path':str(data)}}},'model':{'path':str(root),'served_name':'model'}}
            case=dict(id='c1',formal_requests=8,output_tokens=32,concurrency=2,timeout_seconds=60,steady_state=True,num_warmups=3)
            path=write_config(spec,case,{'endpoint':'http://127.0.0.1:8000'},root)
            text=path.read_text();ast.parse(text)
            # Inspect the generated model configuration without importing AISBench.
            tree=ast.parse(text)
            models=next(n.value for n in tree.body if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='models' for t in n.targets))
            fields={k.arg:k.value for k in models.elts[0].keywords}
            # Native retry counts total attempts; zero would suppress all HTTP.
            self.assertEqual(ast.literal_eval(fields['retry']),1)
            from urllib.parse import urljoin
            self.assertEqual(urljoin(ast.literal_eval(fields['url']), 'v1/completions'),
                             'http://127.0.0.1:8000/v1/completions')
            self.assertEqual(ast.literal_eval(fields['max_out_len']),32)
            self.assertFalse(ast.literal_eval(fields['generation_kwargs'])['ignore_eos'])
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
