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
            result=audit_synthetic_result(root,{'input_tokens':5,'formal_requests':2},'/model',Tokenizer())
            self.assertIn('input text',result['error'])
            self.assertEqual(json.loads((root/'synthetic-evidence.json').read_text())['status'],'UNVERIFIED')

    def test_direct_config_can_be_built_without_reading_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            spec={'benchmark':{'datasets':{'default':{'type':'aisbench_synthetic'}}},'model':{'path':'/model','served_name':'model'}}
            case={'input_tokens':32,'output_tokens':8,'formal_requests':2,'concurrency':2,'shared_prefix_ratio':0}
            path=write_config(spec,case,{'endpoint':'http://localhost:8000'},Path(directory))
            self.assertIn('SyntheticDataset',path.read_text())

    def test_native_after_phase_accepts_target_drift_but_rejects_failed_requests(self):
        import test_evidence
        from validate_case import validate
        spec,bindings=inputs();case=spec['cases'][0]
        case.update(input_tokens=65536,output_tokens=1024,formal_requests=2,concurrency=2,shared_prefix_ratio=0.9)
        bindings['nodes']['node_a']['benchmark']['datasets']={case.get('dataset_id','default'):{'type':'aisbench_synthetic'}}
        with tempfile.TemporaryDirectory() as directory:
            task=Path(directory)/'task';entry.generate(spec,bindings,task)
            folder=task/'results/test';details=folder/'outputs/run/performances/service/dataset_details.jsonl'
            details.parent.mkdir(parents=True)
            log=folder/'outputs/run/logs/infer/service/dataset.out';log.parent.mkdir(parents=True);log.write_text('worker')
            rows=[dict(id=i,success=True,error_info='',input_tokens=67000+i,output_tokens=1024) for i in range(2)]
            def save():details.write_text('\n'.join(json.dumps(row) for row in rows))
            save()
            with patch.object(test_evidence,'capture'),patch.object(test_evidence,'summarize'),patch.object(test_evidence,'audit_synthetic_result',return_value={'status':'UNVERIFIED'}):
                test_evidence.check(task,spec['sub_tasks'][0]['id'],'node_a',folder,'after')
                result=json.loads((folder/'validated.json').read_text())
                self.assertEqual(result['status'],'PASSED')
                self.assertEqual(result['input_length'],{'policy':'native_target','target':65536,'exact_match':False})
                # The external explicit-length contract is unchanged.
                with self.assertRaisesRegex(ValueError,'expectation'):validate(folder,case)
                rows[0]['success']=False;save()
                with self.assertRaisesRegex(ValueError,'request failed'):
                    test_evidence.check(task,spec['sub_tasks'][0]['id'],'node_a',folder,'after')

    def test_default_request_multiplier_reaches_native_config_and_cli_with_steady_state(self):
        for concurrency in (64,128):
            for steady in (False,True):
                with self.subTest(concurrency=concurrency,steady=steady),tempfile.TemporaryDirectory() as directory:
                    spec,bindings=inputs();case=spec['cases'][0]
                    case.update(input_tokens=65536,output_tokens=1024,concurrency=concurrency,steady_state=steady,shared_prefix_ratio=0.9)
                    case.pop('formal_requests',None)
                    bindings['nodes']['node_a']['benchmark']['datasets']={case.get('dataset_id','default'):{'type':'aisbench_synthetic'}}
                    root=Path(directory)/'task';normalized,_=entry.generate(spec,bindings,root)
                    self.assertEqual(normalized['cases'][0]['formal_requests'],concurrency*4)
                    self.assertEqual(normalized['cases'][0]['num_warmups'],1)
                    self.assertEqual(normalized['cases'][0]['steady_state'],steady)
                    text=next(root.glob('scripts/tests/*.config.py')).read_text()
                    self.assertIn("'RequestCount': "+str(concurrency*4),text)
                    self.assertIn('summarizers.perf.'+('stable_stage' if steady else 'default_perf'),text)
                    script=next(root.glob('scripts/tests/*.sh')).read_text()
                    self.assertIn('--num-prompts '+str(concurrency*4),script)
                    self.assertIn('--num-warmups 1',script)

    def test_explicit_request_count_is_preserved_and_missing_steady_defaults_false(self):
        from task_model import normalize
        spec,_=inputs();case=spec['cases'][0]
        case.update(concurrency=64,formal_requests=128);case.pop('steady_state',None)
        result=normalize(spec)['cases'][0]
        self.assertEqual(result['formal_requests'],128)
        self.assertFalse(result['steady_state'])
