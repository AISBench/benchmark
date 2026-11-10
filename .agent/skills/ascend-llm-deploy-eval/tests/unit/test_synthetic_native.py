import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from test_compact_entry import inputs
import run as entry
from task_model import validate_bindings
from aisbench_cli import dataset_for, write_config, synthetic_config, audit_synthetic_result

class SyntheticNativeTests(unittest.TestCase):
    def test_native_generation_needs_no_external_file_or_manifest(self):
        spec,bindings=inputs();case=spec['cases'][0]
        case.update(input_tokens=64,output_tokens=8,formal_requests=4,concurrency=2,shared_prefix_ratio=0.9)
        bindings['nodes']['node_a']['benchmark']['datasets']={case.get('dataset_id','default'):{'type':'aisbench_synthetic'}}
        validate_bindings(bindings)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'job'
            with patch.object(entry.subprocess,'run',side_effect=AssertionError('no remote call in scripts mode')):
                _,plan=entry.generate(spec,bindings,root)
            config=next(root.glob('scripts/tests/*.config.py')).read_text()
            tree=ast.parse(config)
            datasets=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='datasets' for t in n.targets))
            fields={k.arg:k.value for k in datasets.elts[0].keywords}
            self.assertEqual(fields['type'].id,'SyntheticDataset')
            native=ast.literal_eval(fields['config'])
            self.assertEqual(native['TokenIdConfig'],{'RequestSize':64,'PrefixLen':57})
            self.assertEqual(native['RequestCount'],4)
            self.assertNotIn('path',fields)
            self.assertNotIn('generation_manifest',config)
            self.assertNotIn('generate_gsm8k',config)
            self.assertTrue(plan['tests'])

    def test_synthetic_validation_and_rounding(self):
        case={'input_tokens':65536,'formal_requests':4,'shared_prefix_ratio':0.9}
        self.assertEqual(synthetic_config(case)['TokenIdConfig']['PrefixLen'],58982)
        for ratio in (0,1):
            self.assertEqual(synthetic_config(dict(case,shared_prefix_ratio=ratio))['TokenIdConfig']['PrefixLen'],65536*ratio)
        for changes in ({'input_tokens':None},{'input_tokens':True},{'formal_requests':0},{'shared_prefix_ratio':float('nan')},{'shared_prefix_ratio':1.01}):
            with self.assertRaises(ValueError):synthetic_config(dict(case,**changes))
        with self.assertRaisesRegex(ValueError,'external paths'):
            dataset_for({'datasets':{'default':{'type':'aisbench_synthetic','path':'/data.jsonl'}}},case)
        spec,bindings=inputs()
        bindings['nodes']['node_a']['benchmark']['datasets']={'default':{'type':'aisbench_synthetic','generation_manifest':'/wrong.json'}}
        with self.assertRaises(Exception):validate_bindings(bindings)

    def test_request_text_audit_records_actual_prefix_without_rewriting(self):
        class Tokenizer:
            def encode(self,text,add_special_tokens=False):return list(text)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'outputs/time/performances/service';folder.mkdir(parents=True)
            details=folder/'dataset_details.jsonl'
            details.write_text('\n'.join(json.dumps({'input':s}) for s in ['abcX','abcY']))
            original=details.read_bytes()
            result=audit_synthetic_result(root,{'input_tokens':5,'formal_requests':2,'shared_prefix_ratio':0.5},'/model',Tokenizer())
            self.assertEqual(result['common_prefix_tokens'],3)
            self.assertEqual(result['target_prefix_tokens'],2)
            self.assertFalse(result['exact_input_lengths'])
            self.assertEqual(details.read_bytes(),original)
            details.write_text('{"prediction":"missing input"}')
            with self.assertRaisesRegex(ValueError,'input text'):
                audit_synthetic_result(root,{'input_tokens':5,'formal_requests':2},'/model',Tokenizer())
            self.assertEqual(json.loads((root/'synthetic-evidence.json').read_text())['status'],'UNVERIFIED')

    def test_direct_config_can_be_built_without_reading_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            spec={'benchmark':{'datasets':{'default':{'type':'aisbench_synthetic'}}},'model':{'path':'/model','served_name':'model'}}
            case={'input_tokens':32,'output_tokens':8,'formal_requests':2,'concurrency':2,'shared_prefix_ratio':0}
            path=write_config(spec,case,{'endpoint':'http://localhost:8000'},Path(directory))
            self.assertIn('SyntheticDataset',path.read_text())
