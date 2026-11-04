import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tool_links import record, cleanup


class LinkCleanupTests(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root/'target'
            target.touch()
            try:
                (root/'probe').symlink_to(target)
            except OSError as exc:
                if getattr(exc, 'winerror', None) == 1314:
                    self.skipTest('Windows account lacks symbolic-link permission')
                raise

    def test_cached_dataset_and_two_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root/'dataset-cache'; cache.mkdir()
            data = cache/'data.jsonl'; data.write_text('retained')
            link = root/'test.jsonl'
            for case in ('p50', 'p90'):
                attempt = root/case; attempt.mkdir()
                manifest = attempt/'tool-links.json'
                with patch.dict(os.environ, BENCH_LINK_MANIFEST=str(manifest)):
                    link.symlink_to(data); record(link)
                    # Warmup-to-formal replacement updates ownership.
                    link.unlink(); link.symlink_to(data); record(link)
                self.assertEqual(cleanup([link], manifest), [])
                self.assertFalse(link.is_symlink())
                self.assertEqual(data.read_text(), 'retained')

    def test_unknown_or_changed_links_and_files_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = root/'links.json'
            target = root/'data'; target.write_text('data')
            other = root/'other'; other.write_text('other')
            link = root/'link'; link.symlink_to(target)
            self.assertTrue(cleanup([link], manifest))
            with patch.dict(os.environ, BENCH_LINK_MANIFEST=str(manifest)):
                record(link)
            link.unlink(); link.symlink_to(other)
            self.assertTrue(cleanup([link], manifest))
            self.assertTrue(link.is_symlink())
            link.unlink(); link.write_text('foreign file')
            self.assertTrue(cleanup([link], manifest))
            self.assertEqual(link.read_text(), 'foreign file')
            link.unlink()
            self.assertEqual(cleanup([link], manifest), [])


if __name__ == '__main__':
    unittest.main()
