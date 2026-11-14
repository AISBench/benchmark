from pathlib import Path
import tempfile
import unittest
from deploy_eval_v2.artifacts import read_result
from deploy_eval_v2.protocol import encode, ProtocolError


class ArtifactTests(unittest.TestCase):
    def test_persisted_json_bytes_match_event_content_digest(self):
        from deploy_eval_v2.durable import atomic_json
        from deploy_eval_v2.hashes import digest
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            value={'z':1,'a':{'second':'中文','first':True}}
            atomic_json(root/'results/stage.json',value)
            response=read_result(root,'t','results/stage.json',expected_sha256=digest(value))
            self.assertEqual(response['verified_sha256'],digest(value))

    def test_result_digest_checked_before_returning_partial_content(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'results').mkdir()
            path=root/'results/a.json';raw=('中文'*4000).encode();path.write_bytes(raw)
            checksum=hashlib.sha256(raw).hexdigest()
            result=read_result(root,'t','results/a.json',max_bytes=1024,expected_sha256=checksum)
            self.assertEqual(result['verified_sha256'],checksum)
            self.assertLessEqual(len(encode(result)),1024)
            path.write_bytes(raw+b'changed')
            with self.assertRaisesRegex(ProtocolError,'digest mismatch'):
                read_result(root,'t','results/a.json',offset=result['next_offset'],expected_sha256=checksum)

    def test_tail_and_line_range_are_bounded_and_resumable(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'logs').mkdir()
            text='first\n'+'中文\\"'*3000+'\nlast\n'
            (root/'logs/a').write_bytes(text.encode())
            tail=read_result(root,'t','logs/a',tail_bytes=7)
            self.assertTrue(tail['content'].endswith('last\n'))
            offset,line=0,1
            parts=[]
            while True:
                result=read_result(root,'t','logs/a',max_bytes=1024,line_start=2,line_end=2,scan_offset=offset,scan_line=line)
                self.assertLessEqual(len(encode(result)),1024)
                parts.append(result['content'])
                offset=result['locator']['next_scan_offset'];line=result['locator']['next_scan_line']
                if not result['truncated']:
                    break
            self.assertEqual(''.join(parts),'中文\\"'*3000+'\n')

    def test_keyword_crosses_scan_boundary_without_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'logs').mkdir()
            (root/'logs/a').write_bytes(b'x'*(1024*1024-2)+'needle中文\n'.encode())
            result=read_result(root,'t','logs/a',keyword='needle中文')
            self.assertFalse(result['locator']['matched'])
            result=read_result(root,'t','logs/a',keyword='needle中文',scan_offset=result['locator']['next_scan_offset'])
            self.assertTrue(result['locator']['matched'])
            self.assertTrue(result['content'].startswith('needle中文'))

    def test_invalid_locator_combination_is_rejected(self):
        with self.assertRaises(ProtocolError):
            read_result('/unused','t','logs/a',tail_bytes=10,keyword='x')

    def test_keyword_resume_does_not_skip_undelivered_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'logs').mkdir()
            (root/'logs/a').write_bytes(b'needle'+b'x'*4000+b'needle-end')
            first=read_result(root,'t','logs/a',keyword='needle',max_bytes=1024)
            second=read_result(root,'t','logs/a',keyword='needle',max_bytes=1024,
                               scan_offset=first['locator']['next_scan_offset'])
            self.assertTrue(second['content'].startswith('needle-end'))
    def test_lossless_escaped_unicode_pagination(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'logs').mkdir()
            source = '中文\\"\n'*10000
            (root/'logs/a').write_bytes(source.encode('utf-8'))
            offset, chunks = 0, []
            while True:
                result = read_result(root, 't', 'logs/a', offset)
                self.assertLessEqual(len(encode(result)), 8192)
                chunks.append(result['content'])
                offset = result['next_offset']
                if not result['truncated']:
                    break
            self.assertEqual(''.join(chunks), source)
            with self.assertRaises(ProtocolError):
                read_result(root, 't', 'logs/a', 1)
            with self.assertRaises(ProtocolError):
                read_result(root, 't', '../secret')
