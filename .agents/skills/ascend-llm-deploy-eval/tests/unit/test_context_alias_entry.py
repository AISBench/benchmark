import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from test_compact_entry import inputs, PACKAGE
from task_model import normalize, validate

class ContextAliasEntryTests(unittest.TestCase):
    def spec(self,key='max_model_len'):
        spec,binding=inputs()
        for item in spec['deployments'][0]['instances']:
            if item['role']=='proxy':continue
            args=item['launch']['args'];args.pop('max-model-len',None);args[key]=1024
        spec['cases'][0].update(input_tokens=896,output_tokens=128)
        return spec,binding
    def test_public_scripts_entry_alias_preserved(self):
        for key in ('max_model_len','max-model-len'):
            spec,binding=self.spec(key)
            self.assertEqual(validate(normalize(spec)),[])
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);(root/'spec.json').write_text(json.dumps(spec));(root/'binding.json').write_text(json.dumps(binding))
                result=subprocess.run([sys.executable,str(PACKAGE/'scripts/run.py'),'--job-spec',str(root/'spec.json'),'--bindings',str(root/'binding.json'),'--run-dir',str(root/'output'),'--mode','scripts'],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                dep=spec['deployments'][0]
                for item in dep['instances']:
                    if item['role']=='proxy':continue
                    script=(root/'output/scripts'/dep['id']/('node_a-'+item['id']+'.sh')).read_text()
                    self.assertIn('--'+key,script)
                    self.assertNotIn('--'+('max-model-len' if key=='max_model_len' else 'max_model_len'),script)
    def test_actual_overflow_and_duplicate_rejected(self):
        for key in ('max_model_len','max-model-len'):
            spec,_=self.spec(key);spec['cases'][0]['input_tokens']=897
            self.assertTrue(any('Context limit exceeded' in e for e in validate(normalize(spec))))
        for second in (1024,2048):
            spec,_=self.spec()
            spec['deployments'][0]['instances'][0]['launch']['args']['max-model-len']=second
            self.assertTrue(any('Duplicate max-model-len' in e for e in validate(normalize(spec))))
if __name__=='__main__':unittest.main()
