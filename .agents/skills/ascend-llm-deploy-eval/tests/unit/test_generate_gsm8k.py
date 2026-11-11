import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from generate_gsm8k import inspect_source, audit, normalize_lengths


class CharacterTokenizer:
    def encode(self,text,add_special_tokens=False): return list(text)
    def decode(self,tokens,skip_special_tokens=True): return ''.join(tokens)


class GeneratorAuditTests(unittest.TestCase):
    def test_audit_records_mismatch_and_common_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.jsonl'
            path.write_text('\n'.join(json.dumps({'question':q,'answer':''}) for q in ('abcd','abxyz')))
            before=path.read_bytes()
            report=audit(path,CharacterTokenizer(),2,4)
            self.assertFalse(report['exact_lengths'])
            self.assertEqual(report['token_lengths'],[4,5])
            self.assertEqual(report['common_prefix_tokens'],2)
            self.assertEqual(report['cache_hit_rate'],'UNVERIFIED')
            self.assertEqual(path.read_bytes(),before)
            with self.assertRaisesRegex(ValueError,'count'): audit(path,CharacterTokenizer(),3,4)
            self.assertEqual(normalize_lengths(path,CharacterTokenizer(),4),[{'row':1,'before':5,'after':4}])
            self.assertTrue(audit(path,CharacterTokenizer(),2,4)['exact_lengths'])

    def test_source_checks_every_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.jsonl'
            path.write_text('{"question":"ok","answer":"ok"}\n{"question":"bad"}\n')
            with self.assertRaisesRegex(ValueError,'row 2'): inspect_source(path)
