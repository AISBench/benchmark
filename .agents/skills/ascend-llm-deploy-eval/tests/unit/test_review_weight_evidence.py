import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from review_job import PROBE

class ProbeWeightsTests(unittest.TestCase):
    def run_probe(self, root):
        payload={'models':{'m':str(root)},'images':['selected'],'paths':[],'mount_sources':[]}
        def command(args,**kwargs):
            output = json.dumps([{'Id':'sha256:actual','RepoTags':['tag']}]) if args[0]=='docker' else ('[]' if args[0]=='ip' else 'healthy')
            return subprocess.CompletedProcess(args,0,output,'')
        out=io.StringIO()
        with patch('sys.stdin',io.StringIO(json.dumps(payload))),patch('subprocess.run',side_effect=command),contextlib.redirect_stdout(out):
            exec(PROBE,{})
        return json.loads(out.getvalue())
    def seed(self, root):
        (root/'config.json').write_text('{"model_type":"example"}')
        (root/'tokenizer.json').write_text('{}')
        (root/'one.safetensors').write_bytes(b'weight')
    def test_unindexed_and_image_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.seed(root);r=self.run_probe(root)
            self.assertEqual(r['errors'],[])
            self.assertEqual(r['images']['selected'][0]['id'],'sha256:actual')
            self.assertFalse(r['models']['m']['indexed'])
            self.assertEqual(len(r['models']['m']['files']['tokenizer.json']['sha256']),64)
            self.assertNotIn('content',r['models']['m']['files']['tokenizer.json'])
    def test_missing_index_shard_is_blocking(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.seed(root)
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'missing.safetensors'}}))
            r=self.run_probe(root)
            self.assertTrue(r['errors']);self.assertFalse(r['models']['m']['weights_present'])
    def test_empty_shard_and_absent_tokenizer(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.seed(root);(root/'one.safetensors').write_bytes(b'')
            self.assertIn('Empty shard',self.run_probe(root)['errors'][0])
            (root/'one.safetensors').write_bytes(b'x');(root/'tokenizer.json').unlink()
            self.assertIn('tokenizer',self.run_probe(root)['errors'][0])
    def test_complete_index(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.seed(root)
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'one.safetensors'}}))
            r=self.run_probe(root);self.assertEqual(r['errors'],[])
            self.assertTrue(r['models']['m']['indexed'])
    def test_malformed_config(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.seed(root);(root/'config.json').write_text('broken')
            self.assertTrue(self.run_probe(root)['errors'])
if __name__=='__main__':unittest.main()
